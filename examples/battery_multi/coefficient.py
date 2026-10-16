# Author: Mehdi Elasmi
import numpy as np
from coefficients import Coefficient, RandomField

# layer kinds of the cell: cathode collector, active material, anode collector
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
    lo, hi = np.array(domain, dtype=float)[-1]
    return lo + (hi-lo)*edges, np.array(kinds)


def layer_index(domain, stack, y):
    # index of the layer containing the points y (interfaces belong to the upper layer)
    edges, kinds = layer_stack(domain, stack)
    return np.clip(np.searchsorted(edges, y, side="right")-1, 0, len(kinds)-1)


def layer_kind(domain, stack, y):
    _, kinds = layer_stack(domain, stack)
    return kinds[layer_index(domain, stack, y)]


class BatteryCell(Coefficient):
    """Heat conductivity of a battery cell window: layer stack (layer_stack) of cathode collectors
    (conductivity ccc), active material (AM) and anode collectors (acc). The AM layers are split into
    n_seg segments in x; each region r = (AM layer, segment) has its own state p_r in [0, 1] (e.g. the
    local state of charge), which scales the AM conductivity:
        lambda_AM(x, p) = lambda_base(x) factor^(2 p_r - 1),
    with lambda_base = sqrt(lambda_0 lambda_1) of two rough random fields lambda_0, lambda_1
    (log-uniform in [am_min, am_max] on a grid with n_eps cells per direction). The coefficient on a
    patch only depends on the parameters of the regions intersecting it (relevant_parameters)."""

    def __init__(self, domain, stack, ccc=236.3, acc=398.71, am_min=1., am_max=5., n_eps=60,
                 n_seg=6, factor=1.5, seed=73):
        super().__init__(domain)
        self.stack = stack
        self.ccc = ccc
        self.acc = acc
        self.factor = factor
        self.am = RandomField(domain, n_eps=n_eps, a_min=am_min, a_max=am_max, n_fields=2, seed=seed)
        _, kinds = layer_stack(self.domain, stack)
        self.am_count = np.cumsum(kinds == AM)-1  # index of the AM layer of each layer
        self.n_layers = int(np.sum(kinds == AM))
        self.n_seg = n_seg
        self.n_par = self.n_layers*n_seg

    def region(self, x):
        # region index of the points x (-1 in the collectors)
        _, kinds = layer_stack(self.domain, self.stack)
        idx = layer_index(self.domain, self.stack, x[:, -1])
        layer = np.where(kinds[idx] == AM, self.am_count[idx], -1)
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
        collectors = np.array([self.ccc, 0., self.acc])[layer_kind(self.domain, self.stack, x[:, -1])]
        c = self.am.cells(x)
        lam_base = np.exp(0.5*(self.am.log_fields[0, c] + self.am.log_fields[1, c]))
        interp_coeff = np.zeros((N*np.size(P, 0)))
        for i, p in enumerate(P):
            t = np.where(in_am, np.asarray(p)[np.maximum(reg, 0)], 0.)
            am = lam_base*self.factor**(2*t-1)
            interp_coeff[i*N:(i+1)*N] = np.where(in_am, am, collectors)
        return interp_coeff.flatten()[:, None]
