# Author: Mehdi Elasmi
# Shared training driver for the examples, see examples/*/train_model*.py for the configurations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import math
import timeit
import multiprocessing
import shutil
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import tensorflow as tf
import Geo
import DiscreteSetting
import LODANN
import PDEcoeff_functions
import util
import figures

DEFAULTS = {
    # coefficient: class PDEcoeff_<coeff_fun> in PDEcoeff_functions with the arguments coeff_kwargs
    "coeff_fun": "front",
    "coeff_kwargs": {},
    # domain, localization, and discretization
    "domain": [[0, 1], [0, 1]],
    "N_coarse": [6, 6],  # nbr of elements
    "N_fine_per_element": [6, 6],
    "ell": 1,
    "elementwise": False,
    # parameters of the coefficient (train: equidistant, test: random)
    "epsilon_min": 0.1,
    "epsilon_max": 1.0,
    "N_coeff": 40,
    "N_test": 25,
    "test_seed": 2024,
    "withData": False,
    # neural network and optimizers
    "model_class": "LODANN",
    "epochs": 30001,
    "freq_log": 100,
    "width_net": 128,
    "depth_net": 8,
    "initial_lr": 1E-3,
    "decay_rate": 0.9,
    "decay_steps": 1000,
    "lr_sa": 1E-2,  # learning rate for the self adaptive part
    "decay_rate_sa": 0.9,
    "decay_steps_sa": 1000,
    # interpolation constraint enforced exactly by a projection of the network output (True) or
    # penalized with self-adaptive weights as in the paper (False)
    "exact_constraint": True,
    # inputs mapped to [-1, 1] and output scaled to the size of the correction (layers of the model)
    "normalize": True,
    # early stopping of Adam (see LODANN.train_model): "plateau" stops when the loss has not
    # decreased by more than es_rel_tol (relative) during es_patience epochs (not before
    # es_min_epochs); "paper": criterion of the original code; "none"
    "early_stopping": "plateau",
    "es_patience": 2000,
    "es_rel_tol": 1E-4,
    "es_min_epochs": 0,
    # training stages: 0: Adam (epochs) only, as in the paper; 1: Adam followed by fine tuning;
    # 2: fine tuning of an already trained case (loaded from the results path, results are written
    # to <case>/fineTuned/)
    "mode": 1,
    # fine tuning: "lbfgs" (L-BFGS with finetune_epochs iterations, memory lbfgs_maxcor) or "adam"
    # (finetune_epochs steps with constant learning rates finetune_lr, finetune_lr_sa)
    "finetune_optimizer": "lbfgs",
    "finetune_epochs": 2000,
    "finetune_lr": 1E-4,
    "finetune_lr_sa": 1E-4,
    "lbfgs_maxcor": 50,
    # number of coefficients per Adam step (0: all) and used by the L-BFGS fine tuning (0: all);
    # mini-batches are needed for many (e.g. vector valued) parameters
    "coeff_batch": 0,
    "lbfgs_coeffs": 0,
    # seed of the random training parameters (vector valued parameters only)
    "param_seed": 1,
    # precision of the L-BFGS fine tuning ("float64" or "float32")
    "lbfgs_precision": "float64",
    # transfer learning (N_iterations > 1 retrains starting from the previous iteration)
    "transferLearning": False,
    "N_iterations": 1,
    # training of the correction problems in parallel processes
    "parallelize": False,
    "n_workers": None,  # None: number of cpus
    # basis functions to train: None for all, except global boundary nodes if skip_boundary_nodes
    "indices": None,
    # with homogeneous Dirichlet conditions, the basis functions of global boundary nodes are
    # not used by the solver and need not be trained
    "skip_boundary_nodes": True,
    # final time of the time-dependent problem in the evaluation (solve_parabolic.py, make_figures.py)
    "T_final": 1.0,
    # further parameters of the time-dependent problem of an example (read by problem() in its
    # solve_parabolic.py, e.g. physical units), do not affect the training
    "problem_kwargs": {},
    "assemble_GModel": True,
    "plots": True,
    "result_root": "results/",
}


def make_config(**kwargs):
    unknown = set(kwargs) - set(DEFAULTS)
    assert not unknown, "unknown config keys: " + str(unknown)
    cfg = dict(DEFAULTS)
    cfg.update(kwargs)
    return cfg


def config_from_argv(cfg, argv=None):
    """Overrides config entries from command line arguments key=value, e.g.
    python train_model.py epochs=1001 width_net=64 elementwise=True ell=2
    Values are parsed as Python literals (numbers, True/False, lists, None), otherwise strings."""
    import ast
    import sys
    cfg = dict(cfg)
    for arg in (sys.argv[1:] if argv is None else argv):
        key, sep, value = arg.partition("=")
        assert sep == "=" and key in DEFAULTS, "unknown argument " + arg + \
            ", expected key=value with key in training.DEFAULTS"
        try:
            cfg[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            cfg[key] = value
        print("config override: ", key, "=", cfg[key])
    return cfg


def case_name(cfg):
    if cfg["elementwise"]:
        typeComputation = "element"
    else:
        typeComputation = "nodal"
    DataExt = "withData" if cfg["withData"] else "withoutData"
    coarse = "_c" + str(np.array(cfg["N_coarse"]))
    return "case_domain_"+str(np.array(cfg["domain"]).tolist())+"_w"+str(cfg["width_net"])+"_d"+str(cfg["depth_net"]) + \
        "_p"+str(cfg["epsilon_min"])+"to"+str(cfg["epsilon_max"])+coarse +\
        "_f"+str(np.array(cfg["N_fine_per_element"]))+"_l"+str(cfg["ell"])+"_coeff" + \
        str(cfg["N_coeff"])+"_"+typeComputation+"_"+DataExt+"/"


def results_path(cfg):
    # folder of the trained models of a configuration (mode 2: fine tuned models)
    path = cfg["result_root"] + case_name(cfg)
    return path + "fineTuned/" if cfg["mode"] == 2 else path


def output_size(cfg):
    return 2**len(cfg["domain"]) if cfg["elementwise"] else 1


def layer_sizes(cfg):
    return [len(cfg["N_coarse"])+1]+cfg["depth_net"]*[cfg["width_net"]]+[output_size(cfg)]


def build_setting(cfg, param_coeff, test_coeff):
    domain = np.array(cfg["domain"])
    N_coarse = np.array(cfg["N_coarse"])
    N_fine_per_element = np.array(cfg["N_fine_per_element"])
    PDEcoeff = PDEcoeff_functions.make_PDEcoeff(
        cfg["coeff_fun"], domain, **cfg["coeff_kwargs"])
    Mesh = Geo.GEO(domain, N_coarse, N_fine_per_element, cfg["ell"],
                   np_dtype=np.float32, elementwise=cfg["elementwise"])
    FE = DiscreteSetting.FE_Nodal(
        Mesh, PDEcoeff, param_coeff, test_coeff, withData=cfg["withData"])
    return Mesh, PDEcoeff, FE


# context of the training processes (set by init_context, also in the worker processes)
_CTX = {}


def init_context(cfg, param_coeff, test_coeff, resultPath, loadPath):
    # several processes may share a GPU
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # GPU already initialized
    Mesh, PDEcoeff, FE = build_setting(cfg, param_coeff, test_coeff)
    _CTX.update(cfg=cfg, Mesh=Mesh, PDEcoeff=PDEcoeff, FE=FE,
                resultPath=resultPath, loadPath=loadPath, A_plot=None)
    if cfg["plots"]:
        # coefficient of the plotted (last) training parameter
        _CTX["A_plot"] = PDEcoeff(Mesh.Mids, [param_coeff[-1]]).flatten()


def train_one(index):
    cfg, FE = _CTX["cfg"], _CTX["FE"]
    resultPath, loadPath = _CTX["resultPath"], _CTX["loadPath"]
    mode = cfg["mode"]
    ModelClass = LODANN.LODANN
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=cfg["initial_lr"],
                                                                 decay_steps=cfg["decay_steps"],
                                                                 decay_rate=cfg["decay_rate"],
                                                                 staircase=False)
    lr_schedule_sa = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=cfg["lr_sa"],
                                                                    decay_steps=cfg["decay_steps_sa"],
                                                                    decay_rate=cfg["decay_rate_sa"],
                                                                    staircase=False)
    if cfg["transferLearning"] or mode == 2:
        pretrained_model = util.loadModel(loadPath, index)
        print("+ load pretrained model ", index)
    else:
        pretrained_model = []
    options = dict(withTransformInput=False, resultPath=resultPath, DTYPE=tf.float32, mode="train",
                   variableCoeffParam=False, exact_constraint=cfg["exact_constraint"], normalize=cfg["normalize"],
                   coeff_batch=cfg["coeff_batch"], lbfgs_coeffs=cfg["lbfgs_coeffs"])
    net = ModelClass(FE, layer_sizes(cfg), lr_schedule, lr_schedule_sa, [], [], index,
                     model=pretrained_model, **options)
    if mode != 2:
        # train
        print("  -> Start training model for correction problem " + str(index))
        net.train_model(cfg["epochs"], cfg["freq_log"], **early_stopping(cfg))
    if mode != 0:
        # fine tune the model trained above (mode 1) or the loaded one (mode 2)
        print("  - start Fine Tuning (" + cfg["finetune_optimizer"] + ") for Model ", index)
        if cfg["finetune_optimizer"] == "lbfgs":
            net.lbfgs(cfg["finetune_epochs"], cfg["freq_log"], cfg["lbfgs_maxcor"],
                      precision=cfg["lbfgs_precision"])
        elif cfg["finetune_optimizer"] == "adam":
            start_model = net.exportModel() if mode == 1 else pretrained_model
            net = ModelClass(FE, layer_sizes(cfg), cfg["finetune_lr"], cfg["finetune_lr_sa"], [], [], index,
                             model=start_model, **options)
            net.train_model(cfg["finetune_epochs"], cfg["freq_log"], **early_stopping(cfg))
        else:
            raise ValueError("finetune_optimizer must be lbfgs or adam")
        print("  - end Fine Tuning for Model ", index)
    # save model
    net.saveModel()
    if cfg["plots"]:
        plot_result(net, index)


def early_stopping(cfg):
    return dict(early_stopping=cfg["early_stopping"], es_patience=cfg["es_patience"],
                es_rel_tol=cfg["es_rel_tol"], es_min_epochs=cfg["es_min_epochs"])


def plot_result(net, index):
    # quick look: trained correction for the last training coefficient, the exact discrete
    # correction of the same problem (element-wise: the LOD corrector of eq. (7)) and the difference
    cfg, resultPath, A_plot = _CTX["cfg"], _CTX["resultPath"], _CTX["A_plot"]
    ref = None if A_plot is None else _CTX["FE"].patchCorrector(A_plot, index)
    phi = net.u_model(net.Xfinec_tf, net.Cfinex_tf, net.locc_tf).numpy()
    N = np.size(net.Xfine, 0)
    k = net.N_coeff-1
    Xf = net.Xfine
    nx, ny = np.size(np.unique(Xf[:, 0])), np.size(np.unique(Xf[:, 1]))
    extent = (Xf[:, 0].min(), Xf[:, 0].max(), Xf[:, 1].min(), Xf[:, 1].max())
    kind = "element" if cfg["elementwise"] else "node"
    for i in range(output_size(cfg)):
        V = np.reshape(phi[k*N:(k+1)*N, i], (ny, nx))
        panels = [figures.field_panel(V, "network correction (%s %d, output %d)" % (kind, index, i),
                                      cmap="div", extent=extent)]
        if ref is not None:
            R = np.reshape(ref[:, i], (ny, nx))
            m = max(np.abs(V).max(), np.abs(R).max())
            panels = [figures.field_panel(V, "network correction (%s %d, output %d)" % (kind, index, i),
                                          cmap="div", vmin=-m, vmax=m, extent=extent),
                      figures.field_panel(R, "exact discrete correction", cmap="div", vmin=-m,
                                          vmax=m, extent=extent),
                      figures.field_panel(V-R, "difference", cmap="div", extent=extent)]
        figures.save_png(resultPath+"correction_"+str(index)+"_"+str(i)+".png", panels)


def assemble_GModel(resultPath, index_vec):
    # all trained correctors in one model with a common input (output i <-> index_vec[i])
    print("-> Assemble Corrector Model: ")
    start = timeit.default_timer()
    models = [util.loadModel(resultPath, i)[0] for i in index_vec]
    if len(set(model.input_shape[-1] for model in models)) > 1:
        # networks with local parameters have different inputs
        print("Note: no corrector model assembled (the networks have different inputs)")
        return
    input = models[0].layers[0].input
    out = []
    for model in models:
        x = input
        for layer in model.layers[1:]:
            x = layer(x)
        out.append(x)
    if len(out) > 1:
        out = tf.keras.layers.concatenate(out, name="concatenated_models")
    else:
        out = out[0]
    Gmodel = tf.keras.models.Model(
        inputs=[input], outputs=[out], name="corrector_model")
    Gmodel.compile()
    util.save_h5(Gmodel, resultPath+"GModel.h5")
    # optional diagram of the model, requires the graphviz program (dot) and a pydot version that
    # works with keras 2.14 (pydot < 3)
    if shutil.which("dot") is None:
        print("Note: corrector_model.png not created (graphviz is not installed); "
              "this does not affect the results, GModel.h5 is saved")
    else:
        try:
            tf.keras.utils.plot_model(Gmodel, to_file=resultPath +
                                      "corrector_model.png", show_shapes=False, show_dtype=False)
        except Exception as e:
            print("Note: corrector_model.png not created (" + repr(e) + "); "
                  "this does not affect the results, GModel.h5 is saved")
    stop = timeit.default_timer()
    print("assembling time: ", stop - start)


def to_json(obj):
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_training(cfg):
    cfg = dict(cfg)
    N_coarse = np.array(cfg["N_coarse"])
    mode = cfg["mode"]
    N_iterations = cfg["N_iterations"]
    if N_iterations > 1:
        cfg["parallelize"] = False

    # test coefficients (before creating the coefficient function, which may reseed numpy)
    np.random.seed(cfg["test_seed"])
    test_coeff = cfg["epsilon_min"] + \
        (cfg["epsilon_max"] - cfg["epsilon_min"])*np.random.random(cfg["N_test"])
    # training parameters: equidistant for a scalar parameter, Latin hypercube samples in
    # [epsilon_min, epsilon_max]^n_par for vector valued parameters
    n_par = getattr(PDEcoeff_functions.make_PDEcoeff(
        cfg["coeff_fun"], np.array(cfg["domain"]), **cfg["coeff_kwargs"]), "n_par", 1)
    lo, hi = cfg["epsilon_min"], cfg["epsilon_max"]
    if n_par > 1:
        param_coeff = (lo + (hi-lo)*qmc.LatinHypercube(d=n_par, seed=cfg["param_seed"]).random(
            cfg["N_coeff"])).astype(np.float32)
        test_coeff = lo + (hi-lo)*qmc.LatinHypercube(d=n_par, seed=cfg["test_seed"]).random(
            cfg["N_test"])
    else:
        param_coeff = np.linspace(lo, hi, cfg["N_coeff"], dtype=np.float32)

    # results path and config file
    loadPath = cfg["result_root"] + case_name(cfg)
    resultPath = results_path(cfg)
    for path in [resultPath, resultPath+"TrainCoeff", resultPath+"TestCoeff"]:
        os.makedirs(path, exist_ok=True)

    # basis functions to train
    if cfg["elementwise"]:
        N_BF = np.prod(N_coarse)
    else:
        N_BF = np.prod(N_coarse+1)
    if cfg["indices"] is not None:
        index_vec = list(cfg["indices"])
    else:
        index_vec = list(range(N_BF))
        if cfg["skip_boundary_nodes"] and not cfg["elementwise"]:
            index_vec = [i for i in index_vec if i not in util.boundaryNodes(N_coarse)]
    print("train basis functions: ", index_vec)

    config = dict(cfg, coeffFunction=cfg["coeff_fun"], schedule="ExponentialDecay",
                  trained_indices=index_vec)
    with open(resultPath+"config.json", 'w') as f:
        json.dump(to_json(config), f)

    # Train models for correction problems
    start = timeit.default_timer()
    if cfg["parallelize"]:
        np.savetxt(resultPath+"trainParameter_Iter"+str(0)+".txt", param_coeff)
        print("Train param coeff: shape(", np.shape(
            param_coeff), ")\n", param_coeff)
        n_workers = cfg["n_workers"] or multiprocessing.cpu_count()
        print("-> Start training for "+str(len(index_vec)) +
              " correction problems with " + str(n_workers)+" processes")
        # spawn instead of fork: forking a process with an initialized tensorflow is unsafe
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers, initializer=init_context,
                      initargs=(cfg, param_coeff, test_coeff, resultPath, loadPath)) as pool:
            pool.map(train_one, index_vec, chunksize=1)
        with open(resultPath+"training_times.json", "w") as f:
            json.dump({"total": timeit.default_timer() - start, "parallel_workers": n_workers}, f)
    else:
        for samples_i in range(N_iterations):
            np.savetxt(resultPath+"trainParameter_Iter" +
                       str(samples_i)+".txt", param_coeff)
            print("Train param coeff nr", str(samples_i),
                  ": shape(", np.shape(param_coeff), ")\n", param_coeff)
            init_context(cfg, param_coeff, test_coeff, resultPath, loadPath)
            times = {}
            for i, index in enumerate(index_vec):
                start_iter = timeit.default_timer()
                print("-> Training for correction problem's type " +
                      str(i+1)+"/"+str(len(index_vec)))
                train_one(index)
                stop_iter = timeit.default_timer()
                times[int(index)] = stop_iter - start_iter
                print("training time for problem " +
                      str(i+1), ": ", stop_iter - start_iter)
            # offline cost, used by make_figures.py
            with open(resultPath+"training_times.json", "w") as f:
                json.dump({"per_network": times, "total": sum(times.values())}, f)
            stop = timeit.default_timer()
            print("total training time until batch index",
                  str(samples_i), ": ", stop - start)
            cfg = dict(cfg, transferLearning=True)

    if cfg["assemble_GModel"]:
        assemble_GModel(resultPath, index_vec)

    if cfg["plots"]:
        _, PDEcoeff, FE = build_setting(cfg, param_coeff, test_coeff)
        plot_coeff = param_coeff[:16]  # at most 16 coefficients
        cols = math.ceil(np.sqrt(np.shape(plot_coeff)[0]))
        rows = math.ceil(np.shape(plot_coeff)[0]/cols)
        fig = plt.figure(figsize=(4*cols, 3*rows))
        PDEcoeff_functions.plot_coeff(
            PDEcoeff, FE.Mids, plot_coeff, cols, fig)
        plt.savefig(resultPath+"TrainCoeff/train_coefficients")
        plt.close()
    return resultPath
