# Author: Mehdi Elasmi
"""Generic part of the PDE coefficients: the interface of a coefficient, a reusable rough random
field, plotting, and the loading of the coefficient class of an example.

The coefficient of an example is a subclass of Coefficient, defined in the example folder (e.g.
examples/damage_front/coefficient.py) and referenced in the configuration by its import path,
    coeff_fun="examples.damage_front.coefficient:DamageFront",
with the keyword arguments coeff_kwargs of its constructor."""
import importlib
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Coefficient:
    """Coefficient a(x, p) of the PDE, parametrized by p (a scalar or a vector with n_par entries).
    Subclasses implement __call__ and, for vector-valued parameters, set n_par and optionally
    override relevant_parameters."""
    n_par = 1

    def __init__(self, domain):
        self.domain = np.array(domain, dtype=float)

    def __call__(self, x, p):
        """Values at the points x (array (N, d), the fine element midpoints) for each parameter in
        the list p (scalars, or rows of length n_par), stacked as a column vector of length N len(p)."""
        raise NotImplementedError

    def relevant_parameters(self, x):
        # indices of the parameters on which the coefficient at the points x depends (all by
        # default); each network only receives these parameters as input
        return np.arange(self.n_par)


class RandomField:
    """Rough random fields, piecewise constant on a uniform grid with n_eps cells per direction
    (scale eps = 1/n_eps) and log-uniform values in [a_min, a_max] (n_fields independent fields)."""

    def __init__(self, domain, n_eps=36, a_min=0.1, a_max=1., n_fields=2, seed=73):
        self.domain = np.array(domain, dtype=float)
        self.n_eps = n_eps
        dim = np.size(self.domain, 0)
        # own generator: independent of (and not changing) the global numpy seed
        rng = np.random.default_rng(seed)
        self.log_fields = rng.uniform(np.log(a_min), np.log(a_max), size=(n_fields, n_eps**dim))

    def cells(self, x):
        # index of the eps-cell containing the points x
        z = (x - self.domain[:, 0])/(self.domain[:, 1]-self.domain[:, 0])
        ind = np.clip(np.floor(z*self.n_eps).astype(int), 0, self.n_eps-1)
        return np.sum(ind * self.n_eps**np.arange(np.size(x, 1)), axis=1)

    def field(self, x, k=0):
        # values of the field k at the points x
        return np.exp(self.log_fields[k, self.cells(x)])


def make_coefficient(cfg):
    """Coefficient of a configuration: the class cfg["coeff_fun"] ("package.module:Class") with
    the keyword arguments cfg["coeff_kwargs"]."""
    module, name = cfg["coeff_fun"].split(":")
    return getattr(importlib.import_module(module), name)(cfg["domain"], **cfg["coeff_kwargs"])


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
