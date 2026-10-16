# Author: Mehdi Elasmi
# Classical neural baseline without the LOD structure, for comparison (make_figures.py)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import timeit
import numpy as np
import scipy.optimize
import tensorflow as tf
import util


class DeepRitzStepper:
    """Classical Deep Ritz time stepping on the fine mesh: one network u_theta(x) on the whole
    domain (homogeneous Dirichlet condition by the localization function), which minimizes in each
    backward Euler step the discrete energy
        0.5 u'(M/dt + S)u - b'u,   b = M (u_prev/dt + f),
    i.e., the fine FEM time step (eq. (10b)) restricted to network functions. The network is
    warm started from the previous step (Adam, then L-BFGS, double precision). It learns the
    solution for one coefficient sequence (no parametrization), as the classical Deep Ritz method
    discussed in Section 2.4 of the paper."""

    def __init__(self, Mesh, width=128, depth=8, adam_first=3000, adam=500, lbfgs=500,
                 lr=1E-3, seed=73):
        self.Mesh = Mesh
        self.adam_first, self.adam, self.lbfgs_iter, self.lr = adam_first, adam, lbfgs, lr
        tf.random.set_seed(seed)
        X = Mesh.vertices_fine
        lo, hi = X.min(0), X.max(0)
        self.X = tf.constant(2*(X-lo)/(hi-lo)-1, tf.float64)
        self.loc = tf.constant(util.localization(X, lo, hi), tf.float64)
        init = lambda k: tf.keras.initializers.glorot_normal(seed=seed+k)
        layers = [tf.keras.layers.Input((X.shape[1],), dtype=tf.float64)]
        for k in range(depth):
            layers.append(tf.keras.layers.Dense(width, activation="tanh", kernel_initializer=init(k),
                                                dtype=tf.float64))
        layers.append(tf.keras.layers.Dense(1, kernel_initializer=init(depth), dtype=tf.float64))
        self.model = tf.keras.Sequential(layers)
        self.params = self.model.trainable_variables

    def _u(self, scale):
        return scale*self.loc*self.model(self.X)

    def step(self, K, b, scale):
        """One time step: minimize 0.5 u'Ku - b'u over the network functions (K: sparse
        tf.SparseTensor, b: tf column). Returns the nodal values."""
        def energy():
            u = self._u(scale)
            return 0.5*tf.reduce_sum(u*tf.sparse.sparse_dense_matmul(K, u)) - tf.reduce_sum(b*u)

        opt = tf.keras.optimizers.Adam(self.lr)

        @tf.function
        def adam_step():
            with tf.GradientTape() as tape:
                e = energy()
            opt.apply_gradients(zip(tape.gradient(e, self.params), self.params))
            return e

        n_adam = self.adam_first if not getattr(self, "_started", False) else self.adam
        self._started = True
        for _ in range(n_adam):
            adam_step()
        # L-BFGS
        shapes = [p.shape for p in self.params]
        sizes = [int(np.prod(s)) for s in shapes]

        def assign(w):
            for p, part, sh in zip(self.params, tf.split(tf.constant(w, tf.float64), sizes), shapes):
                p.assign(tf.reshape(part, sh))

        @tf.function
        def loss_grad():
            with tf.GradientTape() as tape:
                e = energy()
            g = tape.gradient(e, self.params)
            return e, tf.concat([tf.reshape(x, [-1]) for x in g], 0)

        def fun(w):
            assign(w)
            e, g = loss_grad()
            return float(e), g.numpy()

        w0 = np.concatenate([p.numpy().ravel() for p in self.params])
        e0 = fun(w0)[0]
        res = scipy.optimize.minimize(fun, w0, jac=True, method="L-BFGS-B",
                                      options={"maxiter": self.lbfgs_iter, "maxcor": 50,
                                               "ftol": 1E-15, "gtol": 1E-12})
        assign(res.x if res.fun <= e0 else w0)
        return self._u(scale).numpy()

    def solve(self, A_stiff, A_mass, f, u0, time, scale=None):
        """Backward Euler as solvers.solveFineFEMParabolic, each step by the network. scale: size
        of the solution (output scaling of the network; default from the coarse FEM solution).
        Returns the nodal values (fine mesh x times) and the time per step [s]."""
        import solvers
        Mesh = self.Mesh
        NF = Mesh.N_fine
        if scale is None:
            uc, _ = solvers.solveCoarseFEMParabolic(Mesh, A_stiff, A_mass, f, u0, time)
            scale = float(np.max(np.abs(uc))) or 1.
        u_t = np.zeros((np.prod(NF+1), np.size(time)))
        u_t[:, 0:1] = np.reshape(u0, (-1, 1))
        f_fine = f(Mesh.vertices_fine)
        S_loc = util.localStiffness(NF, a=Mesh.a)
        M_loc = util.localMass(NF, a=Mesh.a)
        t0 = timeit.default_timer()
        for m in range(1, np.size(time)):
            dt = time[m] - time[m-1]
            S = util.assembleGlobalMatrices(S_loc, NF, 'cg', weight=A_stiff[:, m].flatten())
            M = util.assembleGlobalMatrices(M_loc, NF, 'cg', weight=A_mass[:, m].flatten())
            K = util.to_sparse_tensor(M/dt + S, tf.float64)
            b = tf.constant(M @ (u_t[:, m-1:m]/dt + f_fine), tf.float64)
            u_t[:, m:m+1] = self.step(K, b, scale)
            print("   Deep Ritz step %d/%d done (%.0f s)" % (m, np.size(time)-1,
                                                          timeit.default_timer()-t0), flush=True)
        return u_t, (timeit.default_timer()-t0)/max(np.size(time)-1, 1)
