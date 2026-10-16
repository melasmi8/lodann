# Author: Mehdi Elasmi
# Heat source, heat capacity and initial temperature of the battery cell examples
# (layers in y: cathode collector, active material, anode collector, active material, ...)
import numpy as np
from scipy.interpolate import interp1d

# layer kinds of a layer stack
CC, AM, AC = 0, 1, 2


def layer_stack(domain, stack):
    """Layer stack in y given by the dict stack with the entries
        n_rep:   number of repetitions of [CC, AM, AC, AM]
        t_coll:  thickness of the collector layers (CC and AC) in units
        t_am:    thickness of the active material layers (AM) in units
        closing: True: a CC layer closes the stack (default False)
    The domain is divided into n_rep (2 t_coll + 2 t_am) (+ t_coll) units of equal thickness, such
    that all layer interfaces lie on a fine mesh with a multiple of this number of cells in y.
    Returns the interfaces (y coordinates, increasing) and the kinds of the layers in between."""
    t_c, t_a = stack["t_coll"], stack["t_am"]
    kinds, thick = [], []
    for _ in range(stack["n_rep"]):
        kinds += [CC, AM, AC, AM]
        thick += [t_c, t_a, t_c, t_a]
    if stack.get("closing", False):
        kinds.append(CC)
        thick.append(t_c)
    edges = np.concatenate(([0], np.cumsum(thick)))/np.sum(thick)
    lo, hi = domain[-1]
    return lo + (hi-lo)*edges, np.array(kinds)


def layer_index(domain, stack, y):
    # index of the layer containing the points y (interfaces belong to the upper layer)
    edges, kinds = layer_stack(domain, stack)
    return np.clip(np.searchsorted(edges, y, side="right")-1, 0, len(kinds)-1)


def layer_kind(domain, stack, y):
    _, kinds = layer_stack(domain, stack)
    return kinds[layer_index(domain, stack, y)]


def _layers(domain, values, rep=4, stack=None):
    # piecewise constant function of y with the values = (CC, AM, AC) of the layers. Without stack:
    # the original pattern [CC, AM, AC, AM] * rep + CC (relative thicknesses 4 : 16 : 4 : 16)
    if stack is not None:
        vals = np.array(values, dtype=float)
        return lambda y: vals[layer_kind(np.array(domain, dtype=float), stack, np.ravel(y))]
    interpolation_points = np.linspace(domain[1, 0], domain[1, 1], 40*rep+4)
    cc, am, ac = values
    pattern = np.concatenate((np.tile(np.concatenate(
        ([cc]*4, [am]*16, [ac]*4, [am]*16), 0), (rep)), [cc]*4), 0)
    return interp1d(interpolation_points, pattern, kind="previous")


def heat_source(domain, rep=4, q=3., stack=None):
    # heat source q in the active material, 0 in the collectors
    layers = _layers(domain, (0., 1., 0.), rep, stack)
    return lambda x: q*np.squeeze(layers(x[:, 1])).flatten()[:, None]


def heat_capacity(domain, rep=4, c_am=1010.119, rho_am=2094.302, stack=None):
    # volumetric heat capacity rho c_p of the layers (Table 1 of the paper)
    c = _layers(domain, (897.8, c_am, 384.65), rep, stack)
    rho = _layers(domain, (2706.77, rho_am, 8710.2), rep, stack)
    return lambda x: (np.squeeze(c(x[:, 1]))*np.squeeze(rho(x[:, 1]))).flatten()[:, None]


def initial_temperature(u0_const=0., ep=0.0074639*4/184, dirichlet_data=0.):
    # constant u0_const, smoothly connected to the Dirichlet data within the distance ep of y = 0
    def u0(x):
        u = u0_const*np.ones_like(x[:, [1]])
        c = 3*(u-dirichlet_data)/ep**2
        d = -2*(u-dirichlet_data)/ep**3
        near = np.any(x[:, [1]] <= ep, axis=1).reshape([-1, 1])
        return near*(u + c*x[:, [1]]**2 + d*x[:, [1]]**3) + (~near)*u
    return u0
