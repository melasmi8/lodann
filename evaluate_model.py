# Author: Mehdi Elasmi
# evaluate_model
from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import tensorflow as tf
import timeit
import json
import Geo
import PDEcoeff_functions
import DiscreteSetting
import util


def _layers(model):
    # layers of a network after the input: ("rescale", scale, offset) or ("dense", W, b, activation)
    out = []
    for layer in model.layers[1:]:
        if isinstance(layer, tf.keras.layers.Rescaling):
            out.append(("rescale", np.asarray(layer.scale, np.float32), np.asarray(layer.offset, np.float32)))
        elif isinstance(layer, tf.keras.layers.Dense):
            out.append(("dense", layer.kernel.numpy(), layer.bias.numpy(), layer.activation))
        else:
            raise NotImplementedError("batched evaluation: layer type " + type(layer).__name__)
    return out


def batched_outputs(models, inputs, max_bytes=2.5E8):
    """Outputs of several networks of the same architecture: {index: model}, {index: input array
    (rows, d)} -> {index: output (rows, k)}. Networks whose inputs have the same shape are evaluated
    together, their weights stacked, so that each layer is one batched matrix product for the whole
    group (instead of one product per network). The rows are processed in chunks of bounded
    memory (max_bytes for the activations of a group)."""
    groups = {}
    for index, X in inputs.items():
        groups.setdefault(np.shape(X), []).append(index)
    out = {}
    for shape, idx in groups.items():
        layers = [_layers(models[i]) for i in idx]
        stacked = []
        for l, first in enumerate(layers[0]):
            if first[0] == "rescale":
                # input scaling (one value per input) or output scaling (one value): (group, 1, size)
                stacked.append(("rescale",
                                tf.constant(np.stack([np.atleast_1d(L[l][1]) for L in layers])[:, None, :]),
                                tf.constant(np.stack([np.atleast_1d(L[l][2]) for L in layers])[:, None, :])))
            else:
                stacked.append(("dense", tf.constant(np.stack([L[l][1] for L in layers])),
                                tf.constant(np.stack([L[l][2] for L in layers])[:, None, :]), first[3]))
        width = max(np.shape(L[1])[-1] for L in stacked if L[0] == "dense")
        X = np.stack([inputs[i] for i in idx]).astype(np.float32)
        chunk = max(1, int(max_bytes/(4*len(idx)*width)))

        @tf.function
        def forward(x):
            # compiled graph of the whole group, one batched matrix product per layer
            for L in stacked:
                if L[0] == "rescale":
                    x = x*L[1] + L[2]
                else:
                    x = L[3](tf.matmul(x, L[1]) + L[2])
            return x
        res = []
        for r0 in range(0, shape[0], chunk):
            res.append(forward(tf.constant(X[:, r0:r0+chunk, :])).numpy())
        Y = np.concatenate(res, axis=1)
        for g, i in enumerate(idx):
            out[i] = Y[g]
    return out


def evaluateModel(loadPath, evalCoeff="train_coeff", timings=None, batched=None, corrections_out=None):
    """Corrected basis functions of the trained networks for the coefficients evalCoeff. If a dict
    timings is given, it receives the times [s] for loading the models and constraint bases
    ("load", once), evaluating the networks ("networks") and assembling the bases ("assembly")
    for all evaluation coefficients. batched: all networks with inputs of the same shape are
    evaluated together (batched_outputs), otherwise one after the other. Default (None): batched
    if a GPU is available (on a CPU both are about equally fast, on a GPU the batched evaluation
    makes better use of the device)."""
    with open(loadPath+"config.json", 'r') as configFile:
        config = json.load(configFile)
    # import parameters to create the same objects used in training
    coeff_fun = config["coeffFunction"]
    epsilon_min = config["epsilon_min"]
    epsilon_max = config["epsilon_max"]
    N_coeff = config["N_coeff"]
    domain = np.array(config["domain"])
    N_coarse = np.array(config["N_coarse"])
    N_fine_per_element = np.array(config["N_fine_per_element"])
    ell = config["ell"]
    if os.path.exists(loadPath+"trainParameter_Iter"+str(0)+".txt"):
        # one row per coefficient (vector valued parameters) or one value per coefficient
        train_coeff = np.reshape(np.loadtxt(
            loadPath+"trainParameter_Iter"+str(0)+".txt"), (N_coeff, -1))
        if np.shape(train_coeff)[1] == 1:
            train_coeff = train_coeff[:, 0]
    else:
        train_coeff = np.linspace(
            epsilon_min, epsilon_max, N_coeff, dtype=np.float32)

    # evaluation coefficients
    if str(evalCoeff) == "train_coeff":
        eval_coeff = train_coeff
    else:
        eval_coeff = evalCoeff
    N_eval = np.size(eval_coeff, 0)
    print("eval coeff = ", eval_coeff if np.ndim(eval_coeff) == 1 else np.shape(eval_coeff))
    #########################
    if not os.path.exists(loadPath+"EvalCoeff"):
        os.makedirs(loadPath+"EvalCoeff")
    # Discretization
    elementwise = config.get("elementwise", False)
    Mesh = Geo.GEO(domain, N_coarse, N_fine_per_element,
                   ell, elementwise=elementwise)

    # coefficient function with the same settings as in training
    PDEcoeff = PDEcoeff_functions.make_PDEcoeff(
        coeff_fun, domain, **config["coeff_kwargs"])
    FE = DiscreteSetting.FE_Nodal(Mesh, PDEcoeff, train_coeff, eval_coeff)
    # projection onto the interpolation constraint (if used in training)
    exact_constraint = config.get("exact_constraint", False)
    #########################

    # inputs of the network of one correction problem for all evaluation coefficients
    def network_input(index):
        X = Mesh.vertices_fine[Mesh.dofF[index], :]
        N = np.size(X, 0)
        # parameters the network depends on (all for a scalar parameter)
        params = FE.parameterMatrix(eval_coeff)[:, FE.localParameters(index)]
        return np.concatenate((np.tile(X, (N_eval, 1)), np.repeat(params, N, 0)), axis=1)

    # corrections from the network outputs: localization and projection onto the constraint
    def network_correction(index, out, W):
        # returns an array (N_eval, patch nodes, outputs)
        X = Mesh.vertices_fine[Mesh.dofF[index], :]
        N = np.size(X, 0)
        corr = util.localization(X, X[0, :], X[-1, :])[None, :, :] * \
            np.reshape(out, (N_eval, N, -1))
        if W is not None:
            corr = corr - np.einsum('nr,crk->cnk', W,
                                    np.einsum('nr,cnk->crk', W, corr))
        return corr

    # Evaluate models and build G: for each evaluation coefficient, the nodal basis functions minus
    # the corrections, summed over the elements for element-wise problems (eqs. (8) and (20))
    start = timeit.default_timer()
    if elementwise:
        indices = range(np.prod(N_coarse))
        needed = list(indices)
    else:
        indices = range(np.prod(N_coarse+1))
        # basis functions of global boundary nodes are not used by the solver (homogeneous
        # Dirichlet condition) and keep the classical nodal basis function if not trained
        needed = [i for i in indices if i not in util.boundaryNodes(N_coarse)]
    trained = [i for i in indices if os.path.exists(
        loadPath+"correctedHatModel"+str(i)+".h5")]
    missing = [i for i in needed if i not in trained]
    if len(missing) > 0:
        raise FileNotFoundError("no trained model for the correction problems " + str(missing) +
                                " in " + loadPath)
    t0 = timeit.default_timer()
    models = {index: util.loadModel(loadPath, index)[0] for index in trained}
    Ws = {index: FE.constraintBasis(index) if exact_constraint else None for index in trained}
    t_load = timeit.default_timer() - t0
    t0 = timeit.default_timer()
    inputs = {index: network_input(index) for index in trained}
    if batched is None:
        batched = len(tf.config.list_physical_devices("GPU")) > 0
    if batched:
        outputs = batched_outputs(models, inputs)
    else:
        outputs = {index: models[index](inputs[index]).numpy() for index in trained}
    corrections = {index: network_correction(index, outputs[index], Ws[index]) for index in trained}
    if corrections_out is not None:
        # corrections of the single problems {index: (N_eval, patch nodes, outputs)}, e.g. for
        # error estimates
        corrections_out.update(corrections)
    t_networks = timeit.default_timer() - t0
    t0 = timeit.default_timer()
    G_PINN = FE.assembleBases(corrections, N_eval)
    t_assembly = timeit.default_timer() - t0
    if timings is not None:
        timings.update(load=t_load, networks=t_networks, assembly=t_assembly, N_eval=N_eval)
    stop = timeit.default_timer()
    print("total evaluation time: %.2f" % (stop - start), "s")
    return FE, PDEcoeff, eval_coeff, G_PINN
