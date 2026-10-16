# Author: Mehdi Elasmi
import numpy as np
from coefficients import Coefficient, RandomField


class DamageFront(Coefficient):
    """Damage front: rough base conductivity (piecewise constant on the eps-grid, log-uniform in
    [a_min, a_max]) reduced by the factor kappa in a damaged region that grows with p in [0, 1]:
        a(x, p) = a_base(x) (1 - (1 - kappa) D(x, p)),  D = logistic((frac_max p - phi(x)) / delta),
    where phi is a smooth random field (correlation length ~ corr) with uniform marginals, i.e.,
    the damaged area fraction is about frac_max p. The damaged regions nucleate, grow and connect
    as p increases (strongly nonlinear dependence on p, smoothed over the width delta)."""

    def __init__(self, domain, n_eps=36, a_min=0.3, a_max=1., kappa=0.1, frac_max=0.7, delta=0.015,
                 corr=0.1, n_modes=60, seed=73):
        super().__init__(domain)
        self.base = RandomField(domain, n_eps=n_eps, a_min=a_min, a_max=a_max, n_fields=2, seed=seed)
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

    def __call__(self, x, p):
        N = np.size(x, axis=0)
        a_base = self.base.field(x, 0)
        phi = self.phi(x)
        interp_coeff = np.zeros((N*np.size(p)))
        for i, pi in enumerate(np.ravel(p)):
            D = 1./(1.+np.exp(-(self.frac_max*pi - phi)/self.delta))
            interp_coeff[i*N:(i+1)*N] = a_base*(1.-(1.-self.kappa)*D)
        return interp_coeff.flatten()[:, None]
