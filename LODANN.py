# Author: Mehdi Elasmi
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spp
from matplotlib import gridspec
import math
import datetime
import scipy.optimize
import util
import DiscreteSetting
import tensorflow as tf


def my_custom_activation(x, coeff):
    return tf.math.tanh(coeff * x)


class LODANN(DiscreteSetting.FE_Nodal):
    model_name = "PINNLOD"

    def __init__(self, fe, layers, lr, lr_sa, train_data, labels, index, withTransformInput, model, resultPath, DTYPE, mode="train", variableCoeffParam=False,
                 exact_constraint=False, normalize=False, coeff_batch=0, lbfgs_coeffs=0):
        """exact_constraint: the network output is projected onto the corrections satisfying the
        interpolation constraint (instead of penalizing it with self-adaptive weights).
        normalize: inputs mapped to [-1, 1] and output scaled to the expected size of the
        correction by fixed layers of the (saved) model.
        coeff_batch: number of coefficients per Adam step (0: all, otherwise random mini-batches).
        lbfgs_coeffs: number of coefficients used by the L-BFGS fine tuning (0: all)."""
        super().__init__(fe.geo, fe.PDEcoeff, fe.coeff_parameters,
                         fe.test_parameters, fe.withData, fe.np_dtype)
        tf.random.set_seed(73)
        self.mode = mode
        self.exact_constraint = exact_constraint
        self.normalize = normalize
        self.coeff_batch = coeff_batch
        self.lbfgs_coeffs = lbfgs_coeffs
        self.variableCoeffParam = variableCoeffParam
        if mode == "train":
            if self.withData:
                print("Training is performed with data")
            else:
                print("Training is performed without data")
            self.lr = lr
            self.lr_sa = lr_sa
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.optimizer_weights = tf.keras.optimizers.Adam(
                learning_rate=lr_sa)
            # Xavier/Glorot normal initialization: a seeded initializer returns the same values
            # on every call, hence a separate seed for each layer
            self.kernel_initializer = lambda k: tf.keras.initializers.glorot_normal(
                seed=73+2*k)
            self.bias_initializer = lambda k: tf.keras.initializers.glorot_normal(
                seed=73+2*k+1)
            self.activation = tf.keras.activations.tanh  # my_custom_activation
            self.layerType = tf.keras.layers.Dense  # myDenseLayer.myDense
            self.withTransformInput = withTransformInput
        self.layers = layers
        self.dtype = DTYPE
        self.resultPath = resultPath
        if resultPath and not os.path.exists(resultPath):
            os.makedirs(resultPath)
        if self.dim == 2:
            if np.size(index) == 2:
                self.index = index[0]*self.N_coarse[1]+index[1]
            else:
                self.index = index
        elif self.dim == 1:
            self.index = index
        # inputs and parameters restricted to element and dof
        dofC, dofF, PatchC, PatchF = self.dofC[self.index], self.dofF[
            self.index], self.PatchC[self.index], self.PatchF[self.index]
        self.N_patches = np.size(PatchC)
        N_v_pp_fine = self.N_v_pp_fine[self.index]
        self.Xcoarse = self.vertices_coarse[dofC, :]
        self.Xfine = self.vertices_fine[dofF, :]

        print("Patch domain: ", self.Xfine[0, :], ",", self.Xfine[-1, :])
        # localization function at the patch nodes (exactly zero on the patch boundary)
        self.locFun = util.localization(
            self.Xfine, self.Xfine[0, :], self.Xfine[-1, :])

        if self.variableCoeffParam:
            self.Cfine = self.coeff_parameters[:, dofF]
            self.Cfull = self.coeff_parameters
        else:
            # parameters (one row per coefficient) and the ones this network depends on: all for a
            # scalar parameter, for vector parameters those of the regions intersecting the patch
            self.par_idx = self.localParameters(self.index)
            self.Cfull = self.parameterMatrix(self.coeff_parameters)
            self.Cfine = self.Cfull[:, self.par_idx]
            layers = [self.dim + np.size(self.Cfine, 1)] + list(layers[1:])
            self.layers = layers
        if True:
            fig = plt.figure(figsize=(20, 25))
            # at most 16 coefficients
            self.plotCoeff(self.coeff_parameters[:16], np.floor(
                np.sqrt(min(self.N_coeff, 16))), fig, "Train")

        I_patch = spp.csr_matrix(self.patchInterpolation(self.index))
        if self.exact_constraint:
            self.W = self.constraintBasis(self.index)
            self.W_tf = tf.constant(self.W, dtype=DTYPE)

        # coarse functions to correct: the shape functions of the element K = index (element-wise,
        # eq. (7)) or the nodal basis function index
        shapeFunctions = self.patchBasisFunctions(self.index)[dofF, :]
        N_out = np.size(shapeFunctions, 1)
        N_constr = I_patch.shape[0]

        if mode == "train":
            if self.withData:
                if self.elementwise:
                    # LOD correctors of eq. (7)
                    trainData_index = self.exactBlocks(self.Cfull)
                else:
                    trainData_index = [self.TrainData[i][dofF, :]
                                       for i in range(self.N_coeff)]
                    trainData_index = np.vstack(trainData_index)[:, [self.index]]
            # input grid and system matrices (block-diagonal over the coefficients, sparse)
            self.Xcoarsec = np.tile(self.Xcoarse, (self.N_coeff, 1))
            self.Xfinec = np.tile(self.Xfine, (self.N_coeff, 1))
            if self.variableCoeffParam:
                self.Cfinex = np.reshape(
                    self.Cfine, [self.N_coeff*N_v_pp_fine, 1], "C")
                Sfinec = self.systemMatrixDOF(
                    self.coeff_parameters, self.index)
                RHSc = self.rhsBlocks(self.coeff_parameters)
            else:
                self.Cfinex = np.repeat(self.Cfine, N_v_pp_fine, 0)
                S_blocks = self.systemMatrixBlocks(self.Cfull, self.index)
                Sfinec = spp.block_diag(S_blocks, format="csr")
                RHSc = self.rhsBlocks(self.Cfull)
            Ic = spp.kron(spp.eye(self.N_coeff), I_patch, format="csr")
            ShFc = np.tile(shapeFunctions, (self.N_coeff, 1))
            # convert to tensors
            self.Ic_tf = util.to_sparse_tensor(Ic, DTYPE)
            self.Labels_tf = tf.convert_to_tensor(labels, dtype=DTYPE)
            self.Cfinex_tf = tf.convert_to_tensor(self.Cfinex, dtype=DTYPE)
            self.Xfinec_tf = tf.convert_to_tensor(self.Xfinec, dtype=DTYPE)
            self.locc_tf = tf.convert_to_tensor(
                np.tile(self.locFun, (self.N_coeff, 1)), dtype=DTYPE)
            self.Sfinec_tf = util.to_sparse_tensor(Sfinec, DTYPE)
            self.ShFc_tf = tf.convert_to_tensor(ShFc, dtype=DTYPE)
            self.RHSc_tf = tf.convert_to_tensor(RHSc, dtype=DTYPE)
            # per coefficient numpy/scipy sources, used by the L-BFGS fine tuning in double
            # precision and the mini-batches
            n = N_v_pp_fine
            self.train_parts = dict(C=self.Cfine, I=I_patch, RHS=np.reshape(RHSc, (self.N_coeff, n, -1)),
                                    S=S_blocks if not self.variableCoeffParam else None,
                                    data=None if not self.withData else np.reshape(
                                        trainData_index, (self.N_coeff, n, -1)))
            if self.coeff_batch > 0:
                self.setup_batches(I_patch, N_out)
            if self.withData:
                self.TrainData_tf = tf.constant(trainData_index, dtype=DTYPE)

            self.mse_res_log = []
            self.mse_mid_log = []
            self.mse_data_log = []
            self.iteration_log = []

        # test data
        if self.N_test > 0:
            if mode == "train":
                self.Xcoarset = tf.convert_to_tensor(
                    np.tile(self.Xcoarse, (self.N_test, 1)), dtype=DTYPE)
                self.Xfinet = tf.convert_to_tensor(
                    np.tile(self.Xfine, (self.N_test, 1)), dtype=DTYPE)
                self.loct = tf.convert_to_tensor(
                    np.tile(self.locFun, (self.N_test, 1)), dtype=DTYPE)
                if self.withData:
                    if self.elementwise:
                        testData_index = self.exactBlocks(
                            self.parameterMatrix(self.test_parameters))
                    else:
                        testData_index = [self.TestData[i][dofF, :]
                                          for i in range(self.N_test)]
                        testData_index = np.vstack(testData_index)[
                            :, [self.index]]
                if True:
                    fig = plt.figure(figsize=(20, 25))
                    self.plotCoeff(self.test_parameters[:16], np.floor(
                        np.sqrt(min(self.N_test, 16))), fig, "Test")
                if self.variableCoeffParam:
                    self.Ctest = self.test_parameters[:, dofF]
                    self.Ctestx = tf.convert_to_tensor(np.reshape(
                        self.Ctest, [self.N_test*N_v_pp_fine, 1], "C"), dtype=DTYPE)
                    self.Stestt = util.to_sparse_tensor(self.systemMatrixDOF(
                        self.test_parameters, self.index), DTYPE)
                    self.RHSt = tf.convert_to_tensor(
                        self.rhsBlocks(self.test_parameters), dtype=DTYPE)
                else:
                    Ctest_full = self.parameterMatrix(self.test_parameters)
                    self.Ctest = Ctest_full[:, self.par_idx]
                    self.Ctestx = tf.convert_to_tensor(
                        np.repeat(self.Ctest, N_v_pp_fine, 0), dtype=DTYPE)
                    self.Stestt = util.to_sparse_tensor(
                        self.systemMatrixDOF(Ctest_full, self.index), DTYPE)
                    self.RHSt = tf.convert_to_tensor(
                        self.rhsBlocks(Ctest_full), dtype=DTYPE)
                self.It = util.to_sparse_tensor(
                    spp.kron(spp.eye(self.N_test), I_patch, format="csr"), DTYPE)
                self.ShFt = tf.convert_to_tensor(
                    np.tile(shapeFunctions, (self.N_test, 1)), dtype=DTYPE)
                if self.withData:
                    self.TestData_tf = tf.convert_to_tensor(
                        testData_index, dtype=DTYPE)
            elif mode == "eval":
                self.Xcoarset = tf.convert_to_tensor(
                    np.tile(self.Xcoarse, (self.N_test, 1)), dtype=DTYPE)
                self.Xfinet = tf.convert_to_tensor(
                    np.tile(self.Xfine, (self.N_test, 1)), dtype=DTYPE)
                self.loct = tf.convert_to_tensor(
                    np.tile(self.locFun, (self.N_test, 1)), dtype=DTYPE)
                if True:
                    fig = plt.figure(figsize=(20, 25))
                    self.plotCoeff(self.test_parameters[:16], np.floor(
                        np.sqrt(min(self.N_test, 16))), fig, "Eval")
                if self.variableCoeffParam:
                    self.Ctest = self.test_parameters[:, dofF]
                    self.Ctestx = tf.convert_to_tensor(np.reshape(
                        self.Ctest, [self.N_test*N_v_pp_fine, 1], "C"), dtype=DTYPE)
                    self.Stestt = util.to_sparse_tensor(self.systemMatrixDOF(
                        self.test_parameters, self.index), DTYPE)
                    self.RHSt = tf.convert_to_tensor(
                        self.rhsBlocks(self.test_parameters), dtype=DTYPE)
                else:
                    Ctest_full = self.parameterMatrix(self.test_parameters)
                    self.Ctest = Ctest_full[:, self.par_idx]
                    self.Ctestx = tf.convert_to_tensor(
                        np.repeat(self.Ctest, N_v_pp_fine, 0), dtype=DTYPE)
                    self.Stestt = util.to_sparse_tensor(
                        self.systemMatrixDOF(Ctest_full, self.index), DTYPE)
                    self.RHSt = tf.convert_to_tensor(
                        self.rhsBlocks(Ctest_full), dtype=DTYPE)
                self.It = util.to_sparse_tensor(
                    spp.kron(spp.eye(self.N_test), I_patch, format="csr"), DTYPE)
                self.ShFt = tf.convert_to_tensor(
                    np.tile(shapeFunctions, (self.N_test, 1)), dtype=DTYPE)

        # log and config
        if resultPath:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = resultPath + 'logs/train/' + \
                current_time + '/BF_' + str(index)
            self.train_summary_writer = tf.summary.create_file_writer(
                train_log_dir)
            if self.N_test > 0:
                test_log_dir = resultPath + 'logs/test/' + \
                    current_time + '/BF_' + str(index)
                self.test_summary_writer = tf.summary.create_file_writer(
                    test_log_dir)

        self.model = model[0] if len(model) > 0 else self.initialize_model(self.layers)
        if len(model) > 0 and model[1] is not None:
            self.sa_weight_tf = tf.Variable(np.reshape(
                model[1], [N_constr*self.N_coeff, -1]), dtype=DTYPE, trainable=True)
        else:
            self.sa_weight_tf = tf.Variable(tf.random.uniform(
                [N_constr*self.N_coeff, N_out], dtype=DTYPE), trainable=True)
        if len(model) > 0 and model[2] is not None:
            self.sa_weightData_tf = tf.Variable(np.reshape(
                model[2], [N_v_pp_fine*self.N_coeff, -1]), dtype=DTYPE, trainable=True)
        else:
            self.sa_weightData_tf = tf.Variable(tf.random.uniform(
                [N_v_pp_fine*self.N_coeff, N_out], dtype=DTYPE), trainable=True)

        self.trainable_parameters = self.model.trainable_variables
        self.summary = self.model.summary

    def input_scaling(self):
        # affine map of the inputs (x, p) onto [-1, 1]: patch coordinates and training parameters
        Xa, Xl = self.Xfine[0, :], self.Xfine[-1, :]
        scale = list(2/(Xl-Xa))
        offset = list(-2*Xa/(Xl-Xa)-1)
        for pmin, pmax in zip(np.min(self.Cfinex, 0), np.max(self.Cfinex, 0)):
            # each parameter column separately
            if pmax > pmin:
                scale.append(2/(pmax-pmin))
                offset.append(-2*pmin/(pmax-pmin)-1)
            else:
                scale.append(1.)
                offset.append(-pmin)
        return [float(v) for v in scale], [float(v) for v in offset]

    def output_scale(self):
        # size (rms) of the exact discrete correction for the middle training coefficient
        if self.variableCoeffParam:
            return 1.
        A = self.PDEcoeff(
            self.Mids, [self.Cfull[self.N_coeff//2]]).flatten()
        rms = np.sqrt(np.mean(self.patchCorrector(A, self.index)**2))
        return float(rms) if rms > 0 else 1.

    def rhsBlocks(self, coeffs):
        # right-hand sides of the correction problem for the given coefficients, stacked as the inputs
        return np.vstack([self.patchRHS(self.PDEcoeff(self.Mids, coeff).flatten(), self.index)
                          for coeff in coeffs])

    def exactBlocks(self, coeffs):
        # exact discrete corrections for the given coefficients, stacked as the inputs
        return np.vstack([self.patchCorrector(self.PDEcoeff(self.Mids, coeff).flatten(), self.index)
                          for coeff in coeffs])

    def initialize_model(self, layers):
        input = tf.keras.layers.Input(shape=(layers[0],),
                                      name="input_layer_"+str(self.index),
                                      dtype=self.dtype)
        inp = input
        if self.normalize:
            scale, offset = self.input_scaling()
            inp = tf.keras.layers.Rescaling(scale, offset, name="input_scaling_"+str(self.index),
                                            dtype=self.dtype)(inp)
        for depth, width in enumerate(layers[1:-1]):
            inp = self.layerType(width,
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer(
                                     depth),
                                 bias_initializer=self.bias_initializer(
                                     depth),
                                 name="hidden_layer_" +
                                 str(depth+1)+"_"+str(self.index),
                                 dtype=self.dtype)(inp)
        out = tf.keras.layers.Dense(layers[-1],
                                    activation=None,  # self.activation,
                                    kernel_initializer=self.kernel_initializer(
                                        len(layers)-2),
                                    bias_initializer=self.bias_initializer(
                                        len(layers)-2),
                                    name="output_layer_"+str(self.index),
                                    dtype=self.dtype)(inp)
        if self.normalize:
            out = tf.keras.layers.Rescaling(self.output_scale(), name="output_scaling_"+str(self.index),
                                            dtype=self.dtype)(out)
        model = tf.keras.models.Model(inputs=[input], outputs=[
                                      out], name=self.model_name+str(self.index))
        model.compile(optimizer=self.optimizer)
        return model

    def exportModel(self):
        # model and self-adaptive weights in the format expected by the argument model of __init__
        return [self.model, self.sa_weight_tf.numpy(), self.sa_weightData_tf.numpy()]

    def forward_pass(self, X, COEFF, LOC):
        # LOC: precomputed localization function at the nodes X
        u = LOC * self.model(tf.concat([X, COEFF], 1))
        if self.exact_constraint:
            u = self.project(u)
        return u

    def project(self, u, W=None):
        # orthogonal projection of each coefficient block of u (patch nodes x outputs) onto the
        # corrections satisfying the interpolation constraint: v - W W'v
        W = self.W_tf if W is None else W
        n = tf.shape(W)[0]
        k = tf.shape(u)[1]
        v = tf.reshape(u, [-1, n, k])
        Wv = tf.einsum('nr,cnk->crk', W, v)
        v = v - tf.einsum('nr,crk->cnk', W, Wv)
        return tf.reshape(v, [-1, k])

    def u_model(self, X, COEFF, LOC):
        u_nn = self.forward_pass(X, COEFF, LOC)
        return u_nn

    def u_model_interp(self, u, I):
        u_nn = tf.sparse.sparse_dense_matmul(I, u)
        return u_nn

    def res_data(self, u, DATA):
        return tf.subtract(u, DATA)

    def correction_model(self, u, S, RHS):
        # discrete energy 0.5 u'Su - RHS'u, summed over the coefficients, for each output
        # separately (the outputs of element-wise networks are independent problems)
        Su = tf.sparse.sparse_dense_matmul(S, u)
        return 0.5*tf.reduce_sum(u*Su, axis=0) - tf.reduce_sum(RHS*u, axis=0)

    def loss(self, X, COEFF, LOC, I, S, RHS):
        u = self.u_model(X, COEFF, LOC)
        data = self.TrainData_tf if self.withData else None
        return self.loss_terms(u, I, S, RHS, self.sa_weight_tf, self.sa_weightData_tf, data)

    def loss_terms(self, u, I, S, RHS, sa_weight, sa_weightData, data):
        # loss of the network output u: energy (+ penalized interpolation constraint) (+ data)
        res_pde = self.correction_model(u, S, RHS)
        res_interp = self.u_model_interp(u, I)
        zero = tf.zeros([1], dtype=u.dtype)
        if data is not None:
            res_data = self.res_data(u, data)
            mse_data = tf.reduce_mean(
                tf.square(sa_weightData*res_data), axis=0)
        else:
            res_data = zero
            mse_data = zero
        mse_res = res_pde
        if self.exact_constraint:
            # constraint satisfied by construction, pure energy
            loss_value = mse_res + mse_data
        else:
            mse_interp = tf.reduce_mean(
                tf.square(sa_weight*res_interp), axis=0)
            loss_value = mse_res + mse_interp + mse_data
        return loss_value, mse_res, tf.reduce_mean(tf.square(res_interp), axis=0), tf.reduce_mean(tf.square(res_data), axis=0)

    @tf.function
    def evaluateTestLoss(self, Xtest, Ctest, LOCtest, Itest, Stest, RHStest):
        utest = self.u_model(Xtest, Ctest, LOCtest)
        res_pde = self.correction_model(utest, Stest, RHStest)
        res_interp = self.u_model_interp(utest, Itest)
        if self.withData:
            res_data = self.res_data(utest, self.TestData_tf)
            mse_data = tf.reduce_mean(tf.square(res_data), axis=0)
        else:
            res_data = tf.constant([0.], dtype=self.dtype)
            mse_data = tf.constant([0.], dtype=self.dtype)
        mse_res = res_pde
        mse_interp = tf.reduce_mean(tf.square(res_interp), axis=0)
        loss_value = mse_res + mse_interp + mse_data
        return loss_value, mse_res, mse_interp, mse_data

    def setup_batches(self, I_patch, N_out):
        """Mini-batches of coefficients for Adam: the energy u'S_c u is evaluated matrix-free as the
        sum over the fine elements e of the patch of a_c(e) u_e' K u_e (K: element stiffness matrix
        of the uniform fine mesh), so that random subsets of coefficients can be selected cheaply.
        (Fine elements outside the patch do not contribute since u vanishes on the patch boundary.)"""
        assert not self.variableCoeffParam and not self.withData, \
            "coefficient mini-batches: not implemented for variableCoeffParam or withData"
        dofF = self.dofF[self.index]
        elems = self.PatchF[self.index]
        nodes = self.nodesID2elem_fine[elems, :]
        el_nodes = np.searchsorted(dofF, nodes)
        assert np.all(dofF[el_nodes] == nodes)
        dt = self.dtype
        self.n_patch = np.size(dofF)
        self.el_nodes_tf = tf.constant(el_nodes, tf.int32)
        self.K_el_tf = tf.constant(util.localStiffness(self.N_fine, a=self.a), dt)
        self.A_el_tf = tf.constant(np.stack([self.PDEcoeff(self.Mids, c).flatten()[elems]
                                             for c in self.Cfull]), dt)
        self.RHSb_tf = tf.constant(self.train_parts["RHS"], dt)
        self.Cb_tf = tf.constant(self.Cfine, dt)
        self.Ib_tf = tf.constant(I_patch.toarray(), dt)
        self.Xb_tf = tf.constant(self.Xfine, dt)
        self.locb_tf = tf.constant(self.locFun, dt)

    def batch_loss(self, idx):
        # loss of the coefficients idx, scaled to the size of the full sum over all coefficients
        B = tf.shape(idx)[0]
        n = self.n_patch
        X = tf.tile(self.Xb_tf, [B, 1])
        C = tf.repeat(tf.gather(self.Cb_tf, idx), n, axis=0)
        loc = tf.tile(self.locb_tf, [B, 1])
        u = self.u_model(X, C, loc)
        U = tf.reshape(u, [B, n, -1])
        Ue = tf.gather(U, self.el_nodes_tf, axis=1)
        KUe = tf.einsum('ij,bejk->beik', self.K_el_tf, Ue)
        a = tf.gather(self.A_el_tf, idx)[:, :, None, None]
        scale = tf.cast(self.N_coeff, u.dtype)/tf.cast(B, u.dtype)
        mse_res = scale*(0.5*tf.reduce_sum(a*Ue*KUe, axis=[0, 1, 2])
                         - tf.reduce_sum(tf.gather(self.RHSb_tf, idx)*U, axis=[0, 1]))
        res_interp = tf.einsum('rn,bnk->brk', self.Ib_tf, U)
        mse_mid = tf.reduce_mean(tf.square(res_interp), axis=[0, 1])
        if self.exact_constraint:
            loss_value = mse_res
        else:
            r = tf.shape(self.Ib_tf)[0]
            sa = tf.gather(tf.reshape(self.sa_weight_tf, [self.N_coeff, r, -1]), idx)
            loss_value = mse_res + tf.reduce_mean(tf.square(sa*res_interp), axis=[0, 1])
        return loss_value, mse_res, mse_mid, tf.zeros([1], dtype=u.dtype)

    @tf.function
    def evaluateTrainLoss(self):
        return self.train_loss()

    def train_loss(self):
        return self.loss(self.Xfinec_tf, self.Cfinex_tf, self.locc_tf,
                         self.Ic_tf, self.Sfinec_tf, self.RHSc_tf)

    def test_loss(self):
        return self.evaluateTestLoss(self.Xfinet, self.Ctestx, self.loct,
                                     self.It, self.Stestt, self.RHSt)

    def sa_variables(self):
        # self-adaptive weights (updated by gradient ascent)
        sa = [] if self.exact_constraint else [self.sa_weight_tf]
        if self.withData:
            sa.append(self.sa_weightData_tf)
        return sa

    @tf.function
    def opt_min_max(self):
        sa_variables = self.sa_variables()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_parameters)
            tape.watch(sa_variables)
            if self.coeff_batch > 0:
                idx = tf.random.shuffle(tf.range(self.N_coeff))[:self.coeff_batch]
                loss_value, mse_res, mse_mid, mse_data = self.batch_loss(idx)
            else:
                loss_value, mse_res, mse_mid, mse_data = self.train_loss()
        gradients = tape.gradient(
            loss_value, [self.trainable_parameters, sa_variables])
        self.optimizer.apply_gradients(zip(gradients[0],
                                           self.trainable_parameters))
        if len(sa_variables) > 0:
            self.optimizer_weights.apply_gradients(zip([-g for g in gradients[1]],
                                                       sa_variables))
        return loss_value, mse_res, mse_mid, mse_data

    def train_model(self, epochs, freq_log, early_stopping="paper", es_patience=2000, es_rel_tol=1E-5,
                    es_min_epochs=0):
        """Adam (min-max for self-adaptive weights) for at most epochs steps. Early stopping:
        "plateau": stop when the monitored loss (energy plus the unweighted data residual) has not
            decreased by more than es_rel_tol (relative) during es_patience epochs, but not before
            es_min_epochs; with the penalty formulation, additionally the interpolation residual
            must be below 5E-6. Without self-adaptive weights (exact constraint, no data), the
            weights with the lowest loss among the logged steps are restored at the end.
        "paper": criterion of the original code (relative standard deviation of the energy over
            the last 10 logs below 1E-5, interpolation residual below 5E-6, data below 1E-4)
        "none": no early stopping
        The loss is checked every freq_log epochs."""
        assert early_stopping in ["plateau", "paper", "none"]
        restore = early_stopping == "plateau" and len(self.sa_variables()) == 0
        best, best_weights = np.inf, None
        ref, ref_epoch = np.inf, 0
        self.stopped_epoch = None
        for epoch in range(epochs):
            mse_total, mse_res, mse_mid, mse_data = self.opt_min_max()
            if epoch % freq_log == 0:
                if self.coeff_batch > 0:
                    # monitor the loss of all coefficients, not the noisy mini-batch estimate
                    mse_total, mse_res, mse_mid, mse_data = self.evaluateTrainLoss()
                self.mse_res_log.append(np.sum(mse_res))
                self.mse_mid_log.append(np.sum(mse_mid))
                self.mse_data_log.append(np.sum(mse_data))
                self.iteration_log.append(epoch)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(
                        "energy's loss", np.sum(mse_res), step=epoch)
                    tf.summary.scalar(
                        "data's loss", np.sum(mse_data), step=epoch)
                    tf.summary.scalar("interpolation condition",
                                      np.sum(mse_mid), step=epoch)
                    tf.summary.scalar("weights mean", tf.reduce_mean(
                        [tf.reduce_mean(weight) for weight in self.trainable_parameters]), step=epoch)
                    tf.summary.scalar(
                        "learning rate", self.optimizer.learning_rate, step=epoch)
                if self.N_test > 0:
                    _, mse_res_test, mse_mid_test, mse_data_test = self.test_loss()
                    with self.test_summary_writer.as_default():
                        tf.summary.scalar("test energy's loss",
                                          mse_res_test[0], step=epoch)
                        tf.summary.scalar("test data's loss",
                                          mse_data_test[0], step=epoch)
                        tf.summary.scalar(
                            "test interpolation condition", mse_mid_test[0], step=epoch)
                if early_stopping == "paper":
                    if epoch // freq_log > 10 and np.std(self.mse_res_log[-10:])/np.abs(np.mean(self.mse_res_log[-10:])) < 1E-5 \
                            and np.mean(self.mse_mid_log[-10:]) < 5E-6 and np.mean(self.mse_data_log[-10:]) < 1E-4:
                        self.stopped_epoch = epoch
                        break
                elif early_stopping == "plateau":
                    monitor = self.mse_res_log[-1] + self.mse_data_log[-1]
                    if monitor < best:
                        best = monitor
                        if restore:
                            best_weights = [w.numpy()
                                            for w in self.trainable_parameters]
                    # significant improvement over the reference (the first log sets it)
                    if not np.isfinite(ref) or monitor < ref - es_rel_tol*np.abs(ref):
                        ref, ref_epoch = monitor, epoch
                    constraint_ok = self.exact_constraint or self.mse_mid_log[-1] < 5E-6
                    if epoch >= es_min_epochs and epoch - ref_epoch >= es_patience and constraint_ok:
                        self.stopped_epoch = epoch
                        break
        if self.stopped_epoch is not None:
            print("-> Training for model ", self.index,
                  " stopped after convergence at epoch ", self.stopped_epoch)
        if best_weights is not None:
            loss = self.train_loss()
            if np.sum(loss[1]) + np.sum(loss[3]) > best:
                for w, value in zip(self.trainable_parameters, best_weights):
                    w.assign(value)

    # the L-BFGS fine tuning can evaluate the loss in double precision (see lbfgs)
    lbfgs_float64 = True

    def float64_loss(self):
        """Training loss in double precision on a float64 copy of the network weights.
        Returns (variables, loss function). The network is evaluated layer by layer (Rescaling and
        Dense layers), the data of the loss are converted from the stored numpy/scipy sources."""
        params = self.trainable_parameters
        variables = [tf.Variable(p.numpy().astype(np.float64)) for p in params]
        position = {id(p): i for i, p in enumerate(params)}
        layers = []
        for layer in self.model.layers[1:]:
            if isinstance(layer, tf.keras.layers.Rescaling):
                layers.append(("rescale", tf.constant(layer.scale, tf.float64),
                               tf.constant(layer.offset, tf.float64)))
            elif isinstance(layer, tf.keras.layers.Dense):
                layers.append(("dense", position[id(layer.kernel)], position[id(layer.bias)],
                               layer.activation))
            else:
                raise NotImplementedError(
                    "float64 fine tuning: layer type " + type(layer).__name__)
        parts = self.train_parts
        N = self.N_coeff
        sel = np.arange(N)
        if 0 < self.lbfgs_coeffs < N:
            # fixed random subset of the coefficients
            sel = np.sort(np.random.default_rng(self.index).choice(N, self.lbfgs_coeffs, replace=False))
        m = np.size(sel)
        n = np.size(self.Xfine, 0)
        f64 = tf.float64
        X = tf.constant(np.concatenate([np.tile(self.Xfine, (m, 1)),
                                        np.repeat(parts["C"][sel], n, 0)], axis=1), f64)
        loc = tf.constant(np.tile(self.locFun, (m, 1)), f64)
        I = util.to_sparse_tensor(spp.kron(spp.eye(m), parts["I"], format="csr"), f64)
        S = util.to_sparse_tensor(spp.block_diag([parts["S"][c] for c in sel], format="csr"), f64)
        k = np.shape(parts["RHS"])[2]
        RHS = tf.constant(np.reshape(parts["RHS"][sel], (m*n, k)), f64)
        data = None if parts["data"] is None else tf.constant(
            np.reshape(parts["data"][sel], (m*n, k)), f64)
        r = parts["I"].shape[0]
        sa = tf.constant(np.reshape(np.reshape(self.sa_weight_tf.numpy(), (N, r, -1))[sel], (m*r, -1)), f64)
        saData = tf.constant(np.reshape(np.reshape(self.sa_weightData_tf.numpy(), (N, n, -1))[sel],
                                        (m*n, -1)), f64)
        W = tf.constant(self.W, f64) if self.exact_constraint else None

        def loss():
            x = X
            for layer in layers:
                if layer[0] == "rescale":
                    x = x*layer[1] + layer[2]
                else:
                    x = layer[3](tf.matmul(x, variables[layer[1]]) + variables[layer[2]])
            u = loc*x
            if self.exact_constraint:
                u = self.project(u, W)
            return self.loss_terms(u, I, S, RHS, sa, saData, data)
        return variables, loss

    def lbfgs(self, maxiter, freq_log=100, maxcor=50, precision="float64"):
        """L-BFGS (scipy) for the network parameters, with fixed self-adaptive weights, used as fine
        tuning after Adam. precision="float64" evaluates the loss and gradient in double precision
        (on a copy of the weights, the result is written back to the float32 model): in single
        precision the line search typically fails after a few iterations once Adam has converged,
        since the remaining decrease of the loss is below the float32 resolution."""
        params = self.trainable_parameters
        if precision == "float64" and self.lbfgs_float64 and self.train_parts["S"] is not None:
            variables, loss_fun = self.float64_loss()
            dtype = tf.float64
        else:
            variables, loss_fun, dtype = params, self.train_loss, self.dtype
        shapes = [p.shape for p in variables]
        sizes = [int(np.prod(sh)) for sh in shapes]

        def assign(w, target=variables, dt=dtype):
            w = tf.convert_to_tensor(w, dtype=dt)
            for p, part, sh in zip(target, tf.split(w, sizes), shapes):
                p.assign(tf.reshape(part, sh))

        @tf.function
        def loss_and_grad():
            with tf.GradientTape() as tape:
                loss_value, mse_res, mse_mid, mse_data = loss_fun()
                loss = tf.reduce_sum(loss_value)
            grads = tape.gradient(loss, variables)
            return loss, tf.concat([tf.reshape(g, [-1]) for g in grads], 0), mse_res, mse_mid, mse_data

        state = {"it": 0, "logs": None}

        def fun(w):
            assign(w)
            loss, grad, *logs = loss_and_grad()
            state["logs"] = logs
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        def callback(w):
            state["it"] += 1
            if state["it"] % freq_log == 0 and state["logs"] is not None:
                step = (self.iteration_log[-1] if len(self.iteration_log) > 0 else 0) + 1
                mse_res, mse_mid, mse_data = state["logs"]
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("L-BFGS energy's loss", np.sum(mse_res), step=step+state["it"])
                    tf.summary.scalar("L-BFGS interpolation condition", np.sum(mse_mid), step=step+state["it"])

        w0 = np.concatenate([p.numpy().ravel() for p in variables]).astype(np.float64)
        loss0 = fun(w0)[0]
        res = scipy.optimize.minimize(fun, w0, jac=True, method="L-BFGS-B", callback=callback,
                                      options={"maxiter": maxiter, "maxfun": int(1.25*maxiter),
                                               "maxcor": maxcor, "ftol": 1E-14, "gtol": 1E-12})
        # keep the better of the start and the final point, written to the model
        assign(res.x if res.fun <= loss0 else w0, target=params, dt=self.dtype)
        print("  - L-BFGS (%s) for model %s: loss %.8g -> %.8g after %d iterations (%s)"
              % (dtype.name, self.index, loss0, min(res.fun, loss0), res.nit, res.message))
        return res




    def plotCoeff(self, eps, cols, fig, label="Train"):
        N_coeff = np.size(eps, 0)
        PatchF = self.PatchF[self.index]
        x = self.Mids[PatchF, :]
        rows = int(math.ceil(N_coeff / cols))
        gs = gridspec.GridSpec(rows, int(cols))
        if np.size(x, axis=1) == 2:
            Nx = np.size(np.unique(x[:, 0]))
            Ny = np.size(np.unique(x[:, 1]))
            X = np.reshape(x[:, 0], [Nx, Ny], "F")
            Y = np.reshape(x[:, 1], [Nx, Ny], "F")
            for i in range(N_coeff):
                ax = fig.add_subplot(gs[i])
                # evaluate on the whole mesh (as used in the system matrices) and restrict to the
                # patch, since the coefficient functions normalize over the evaluation points
                coeff = self.PDEcoeff(self.Mids, [eps[i]])[PatchF]
                Z = np.reshape(coeff, [Nx, Ny], "F")
                im = ax.pcolormesh(X, Y, Z, cmap="RdBu_r")
                fig.colorbar(im)
            fig.supxlabel("$x$")
            fig.supylabel("$y$")
            os.makedirs(self.resultPath+label+"Coeff", exist_ok=True)
            plt.savefig(self.resultPath+label +
                        "Coeff/coeffient_Index_"+str(self.index))
            plt.close()


    def saveModel(self):
        util.save_h5(self.model, self.resultPath+"correctedHatModel"+str(self.index)+".h5")
        np.savetxt(self.resultPath+"sa_weights_"+str(self.index) +
                   ".txt", self.sa_weight_tf.numpy())
        np.savetxt(self.resultPath+"sa_weightsData_" +
                   str(self.index)+".txt", self.sa_weightData_tf.numpy())
        print("Model saved as: ", self.resultPath +
              "correctedHatModel"+str(self.index)+".h5")

    def loadModel(self, loadPath):
        model = util.load_h5(loadPath+"correctedHatModel"+str(self.index)+".h5")
        sa_weights = np.loadtxt(loadPath+"sa_weights_"+str(self.index)+".txt")
        sa_weightsData = np.loadtxt(
            loadPath+"sa_weightsData_"+str(self.index)+".txt")
        return model, sa_weights, sa_weightsData
