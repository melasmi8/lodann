# Author: Mehdi Elasmi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import battery_physics
from matplotlib import gridspec
import math


class PDEcoeff_blend:
    """Rough coefficient that depends smoothly on the parameter p in [0, 1]: geometric blend of
    n_fields independent random fields, piecewise constant on a uniform grid with n_eps cells per
    direction (scale eps = 1/n_eps) and log-uniform values in [mineps, maxeps]. For n_fields = 2,
    a(x, p) = a_0(x)^(1-p) a_1(x)^p; for more fields, p runs through a_0 -> a_1 -> ... piecewise."""

    def __init__(self, domain, n_eps=36, mineps=0.1, maxeps=1., n_fields=2, seed=73):
        self.domain = np.array(domain, dtype=float)
        self.n_eps = n_eps
        self.mineps = mineps
        self.maxeps = maxeps
        self.n_fields = n_fields
        self.seed = seed
        dim = np.size(self.domain, 0)
        # own generator: independent of (and not changing) the global numpy seed
        rng = np.random.default_rng(seed)
        self.log_fields = rng.uniform(np.log(mineps), np.log(maxeps),
                                      size=(n_fields, n_eps**dim))

    def cells(self, x):
        # index of the eps-cell containing the points x
        z = (x - self.domain[:, 0])/(self.domain[:, 1]-self.domain[:, 0])
        ind = np.clip(np.floor(z*self.n_eps).astype(int), 0, self.n_eps-1)
        return np.sum(ind * self.n_eps**np.arange(np.size(x, 1)), axis=1)

    def __call__(self, x, p):
        N = np.size(x, axis=0)
        c = self.cells(x)
        interp_coeff = np.zeros((N*np.size(p)))
        for i, pi in enumerate(np.ravel(p)):
            # position between the fields
            s = np.clip(pi, 0., 1.)*(self.n_fields-1)
            k = min(int(np.floor(s)), self.n_fields-2)
            t = s - k
            interp_coeff[i*N:(i+1)*N] = np.exp((1-t)*self.log_fields[k, c]
                                               + t*self.log_fields[k+1, c])
        return interp_coeff.flatten()[:, None]


class PDEcoeff_battery_blend:
    """Battery cell (layers in y): cathode collector (conductivity ccc),
    active material (AM), anode collector (acc), AM, ..., repeated rep times. The AM conductivity is
    a rough random field that depends smoothly on p in [0, 1] (PDEcoeff_blend: geometric blend of
    n_fields random fields, piecewise constant on a grid with n_eps cells per direction, values
    log-uniform in [am_min, am_max]). Only the AM depends on p."""

    def __init__(self, domain, ccc=236.3, acc=398.71, am_min=1., am_max=5., n_eps=60, n_fields=2,
                 seed=73, rep=4, repP=4, stack=None):
        # stack: layer stack of battery_physics.layer_stack (interfaces on the fine mesh); None: the
        # original pattern below
        self.domain = np.array(domain, dtype=float)
        self.ccc = ccc
        self.acc = acc
        self.rep = rep
        self.repP = repP
        self.stack = stack
        self.am = PDEcoeff_blend(domain, n_eps=n_eps, mineps=am_min, maxeps=am_max,
                                 n_fields=n_fields, seed=seed)
        # layer pattern in y: [CC, AM, AC, AM] * rep + CC, with thicknesses repP : 4 repP
        self.interpolation_points = np.linspace(
            self.domain[-1, 0], self.domain[-1, 1], 10*repP*rep+repP)
        self.am_layers = np.concatenate((np.tile(np.concatenate(
            ([0]*repP, [1]*4*repP, [0]*repP, [1]*4*repP), 0), (rep)), [0]*repP), 0)
        self.collectors = np.concatenate((np.tile(np.concatenate(([ccc]*repP, [0]*4*repP, [
            acc]*repP, [0]*4*repP), 0), (rep)), [ccc]*repP), 0)

    def in_am(self, x):
        # 1 in the active material, 0 in the collectors
        if self.stack is not None:
            return (battery_physics.layer_kind(self.domain, self.stack, x[:, -1])
                    == battery_physics.AM).astype(float)
        return interp1d(self.interpolation_points, self.am_layers, kind="previous")(x[:, -1])

    def collector_values(self, x):
        # conductivity of the collectors, 0 in the active material
        if self.stack is not None:
            vals = np.array([self.ccc, 0., self.acc])
            return vals[battery_physics.layer_kind(self.domain, self.stack, x[:, -1])]
        return interp1d(self.interpolation_points, self.collectors, kind="previous")(x[:, -1])

    def am_layer_index(self, x):
        # index of the active material layer (counted from y = 0) containing x, -1 in the collectors
        if self.stack is not None:
            _, kinds = battery_physics.layer_stack(self.domain, self.stack)
            am_count = np.cumsum(kinds == battery_physics.AM)-1
            idx = battery_physics.layer_index(self.domain, self.stack, x[:, -1])
            return np.where(kinds[idx] == battery_physics.AM, am_count[idx], -1)
        layers = -np.ones(np.size(self.am_layers))
        count = -1
        for i, v in enumerate(self.am_layers):
            if v == 1 and (i == 0 or self.am_layers[i-1] == 0):
                count += 1
            if v == 1:
                layers[i] = count
        return interp1d(self.interpolation_points, layers, kind="previous")(x[:, -1]).astype(int)

    def n_am_layers(self):
        if self.stack is not None:
            _, kinds = battery_physics.layer_stack(self.domain, self.stack)
            return int(np.sum(kinds == battery_physics.AM))
        return int(np.sum(np.diff(np.concatenate(([0], self.am_layers))) == 1))

    def __call__(self, x, p):
        N = np.size(x, axis=0)
        in_am = self.in_am(x)
        collectors = self.collector_values(x)
        interp_coeff = np.zeros((N*np.size(p)))
        for i, pi in enumerate(np.ravel(p)):
            interp_coeff[i*N:(i+1)*N] = in_am * \
                self.am(x, [pi]).ravel() + collectors
        return interp_coeff.flatten()[:, None]


class PDEcoeff_front:
    """Damage front: rough base conductivity (piecewise constant on the eps-grid, log-uniform in
    [a_min, a_max]) reduced by the factor kappa in a damaged region that grows with p in [0, 1]:
        a(x, p) = a_base(x) (1 - (1 - kappa) D(x, p)),  D = logistic((frac_max p - phi(x)) / delta),
    where phi is a smooth random field (correlation length ~ corr) with uniform marginals, i.e.,
    the damaged area fraction is about frac_max p. The damaged regions nucleate, grow and connect
    as p increases (strongly nonlinear dependence on p, smoothed over the width delta)."""

    def __init__(self, domain, n_eps=36, a_min=0.3, a_max=1., kappa=0.1, frac_max=0.7, delta=0.015,
                 corr=0.1, n_modes=60, seed=73):
        self.domain = np.array(domain, dtype=float)
        self.base = PDEcoeff_blend(domain, n_eps=n_eps, mineps=a_min, maxeps=a_max, n_fields=2,
                                   seed=seed)
        self.kappa = kappa
        self.frac_max = frac_max
        self.delta = delta
        rng = np.random.default_rng(seed+1)
        self.wave = rng.normal(0., 1./corr, size=(n_modes, np.size(self.domain, 0)))
        self.phase = rng.uniform(0., 2*np.pi, n_modes)
        # empirical distribution of the field on a reference grid (uniform marginals of phi)
        g = [np.linspace(lo, hi, 201) for lo, hi in self.domain]
        ref = np.stack(np.meshgrid(*g), -1).reshape(-1, np.size(self.domain, 0))
        self.cdf = np.sort(self._field(ref))

    def _field(self, x):
        z = (x - self.domain[:, 0])/(self.domain[:, 1]-self.domain[:, 0])
        return np.cos(z @ self.wave.T + self.phase).sum(1)

    def phi(self, x):
        return np.searchsorted(self.cdf, self._field(x))/np.size(self.cdf)

    def damage(self, x, p):
        return 1./(1.+np.exp(-(self.frac_max*p - self.phi(x))/self.delta))

    def __call__(self, x, p):
        N = np.size(x, axis=0)
        a_base = self.base(x, [0.]).ravel()
        phi = self.phi(x)
        interp_coeff = np.zeros((N*np.size(p)))
        for i, pi in enumerate(np.ravel(p)):
            D = 1./(1.+np.exp(-(self.frac_max*pi - phi)/self.delta))
            interp_coeff[i*N:(i+1)*N] = a_base*(1.-(1.-self.kappa)*D)
        return interp_coeff.flatten()[:, None]


class PDEcoeff_battery_multi:
    """Battery cell (layers as in PDEcoeff_battery_blend) with one parameter per region: the active
    material (AM) layers are split into n_seg segments in x, region r = (AM layer, segment), and the
    parameter vector p in [0, 1]^(n_layers n_seg) describes the local state of each region:
        mode "magnitude": lambda_AM = lambda_base(x) * factor^(2 p_r - 1)   (e.g. temperature or
                          state of charge dependent conductivity)
        mode "damage":    lambda_AM = lambda_base(x) * (1 - (1 - kappa) D_r(x)), damage front in each
                          region with level frac_max p_r (as PDEcoeff_front)
        mode "blend":     lambda_AM = lambda_0(x)^(1 - p_r) lambda_1(x)^p_r
    lambda_base = sqrt(lambda_0 lambda_1), lambda_0, lambda_1 rough random fields (as in
    PDEcoeff_battery_blend). The coefficient on a patch only depends on the parameters of the
    regions intersecting it (relevant_parameters)."""

    def __init__(self, domain, ccc=236.3, acc=398.71, am_min=10., am_max=50., n_eps=60, n_seg=6,
                 mode="magnitude", factor=5., kappa=0.3, frac_max=0.7, delta=0.015, corr=0.1,
                 seed=73, rep=4, repP=4, stack=None):
        self.domain = np.array(domain, dtype=float)
        self.cell = PDEcoeff_battery_blend(domain, ccc=ccc, acc=acc, am_min=am_min, am_max=am_max,
                                           n_eps=n_eps, n_fields=2, seed=seed, rep=rep, repP=repP,
                                           stack=stack)
        assert mode in ["magnitude", "damage", "blend"]
        self.mode = mode
        self.factor = factor
        self.front = PDEcoeff_front(domain, kappa=kappa, frac_max=frac_max, delta=delta, corr=corr,
                                    seed=seed)
        self.n_layers = self.cell.n_am_layers()
        self.n_seg = n_seg
        self.n_par = self.n_layers*n_seg

    def region(self, x):
        # region index of the points x (-1 in the collectors)
        layer = self.cell.am_layer_index(x)
        z = (x[:, 0]-self.domain[0, 0])/(self.domain[0, 1]-self.domain[0, 0])
        seg = np.clip(np.floor(z*self.n_seg).astype(int), 0, self.n_seg-1)
        return np.where(layer >= 0, layer*self.n_seg + seg, -1)

    def relevant_parameters(self, x):
        # parameters on which the coefficient at the points x depends
        return np.setdiff1d(np.unique(self.region(x)), [-1])

    def __call__(self, x, P):
        N = np.size(x, axis=0)
        P = np.atleast_2d(P)
        reg = self.region(x)
        in_am = reg >= 0
        log_fields = self.cell.am.log_fields
        c = self.cell.am.cells(x)
        collectors = self.cell(x, [0.]).ravel()*(~in_am)
        lam_base = np.exp(0.5*(log_fields[0, c] + log_fields[1, c]))
        phi = self.front.phi(x) if self.mode == "damage" else None
        interp_coeff = np.zeros((N*np.size(P, 0)))
        for i, p in enumerate(P):
            t = np.where(in_am, np.asarray(p)[np.maximum(reg, 0)], 0.)
            if self.mode == "magnitude":
                am = lam_base*self.factor**(2*t-1)
            elif self.mode == "damage":
                D = 1./(1.+np.exp(-(self.front.frac_max*t - phi)/self.front.delta))
                am = lam_base*(1.-(1.-self.front.kappa)*D)
            else:
                am = np.exp((1-t)*log_fields[0, c] + t*log_fields[1, c])
            interp_coeff[i*N:(i+1)*N] = np.where(in_am, am, collectors)
        return interp_coeff.flatten()[:, None]


def make_PDEcoeff(name, domain, *args, **kwargs):
    # creates the coefficient class PDEcoeff_<name>
    return globals()["PDEcoeff_"+name](domain, *args, **kwargs)


def plot_coeff(fun, x, eps, cols, fig):
    # x should be the points where the coefficient is used, i.e., the element midpoints Mesh.Mids
    N = np.size(eps, 0)
    rows = int(math.ceil(N / cols))
    gs = gridspec.GridSpec(rows, int(cols))
    if np.size(x, axis=1) == 1:
        ax = fig.add_subplot()
        coeff = fun(x, eps)
        ax.plot(x, coeff)
        ax.set_xlabel('x')
        ax.set_ylabel('A')
    elif np.size(x, axis=1) == 2:
        for i in range(N):
            ax = fig.add_subplot(gs[i])
            coeff = fun(x, [eps[i]])
            Nx = np.size(np.unique(x[:, 0]))
            Ny = np.size(np.unique(x[:, 1]))
            X = np.reshape(x[:, 0], [Ny, Nx])
            Y = np.reshape(x[:, 1], [Ny, Nx])
            Z = np.reshape(coeff, [Ny, Nx])
            im = ax.pcolormesh(X, Y, Z)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title("p "+str(i))
            fig.colorbar(im)
    # fig.tight_layout()


def plotandSaveCoeff(path, fun, x, eps, index, ax, fig, label):
    # x should be the points where the coefficient is used, i.e., the element midpoints Mesh.Mids
    import figures
    plt.close(fig)
    coeff = fun(x, eps).ravel()
    if np.size(x, axis=1) == 1:
        fig, ax = plt.subplots()
        ax.plot(x, coeff)
        ax.set_xlabel('x')
        ax.set_ylabel('a')
        plt.savefig(path+"coefficient_"+label+"_"+str(index))
        plt.close()
        return
    nx, ny = np.size(np.unique(x[:, 0])), np.size(np.unique(x[:, 1]))
    Z = np.reshape(coeff, (ny, nx))
    figures.save_png(path+"coefficient_"+label+"_"+str(index)+".png", [
        figures.field_panel(Z, "coefficient (%s, no. %s)" % (label, index), kind="elements",
                            log=Z.max() > 10*Z.min() > 0, cbar_label="a")])
