# Author: Mehdi Elasmi
# Accuracy of the classical LOD (no networks) against the fine FEM solution for the test case of an
# example, to choose the coarse mesh and the localization parameter before training, e.g.
#   python lod_accuracy.py damage_front
#   python lod_accuracy.py battery_multi N_coarse=[6,6] N_fine_per_element=[22,22] ell=1
# (same command line overrides as train_model.py). Prints the relative errors (in %) at the final
# time and averaged over time of coarse FEM, frozen LOD (basis of the first coefficient) and LOD.
import sys
import os
import timeit
from pathlib import Path
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CODE = str(Path(__file__).parent)
sys.path.insert(0, CODE)
import training
import LOD
import solvers
import util

example = sys.argv[1].rstrip("/").split("/")[-1]
sys.path.insert(0, os.path.join(CODE, "examples", example))
import train_model
import solve_parabolic
cfg = training.config_from_argv(train_model.config, sys.argv[2:])

P = solve_parabolic.problem(cfg)
case = [c for c in P["cases"] if c["label"] == "test"][0]
coeffs = np.atleast_2d(case["coeff"]).reshape(len(case["coeff"]), -1)
T = np.linspace(0, 1, len(coeffs)) if case["T"] is None else case["T"]
Mesh, PDEcoeff, FE = training.build_setting(cfg, coeffs[:2], coeffs[:2])
N = len(coeffs)
A_stiff = np.zeros((np.size(FE.Mids, 0), N))
A_mass = np.ones_like(A_stiff)
G = []
t0 = timeit.default_timer()
for i, c in enumerate(coeffs):
    A = PDEcoeff(FE.Mids, [c] if np.size(c) > 1 else [float(np.ravel(c)[0])]).flatten()
    A_stiff[:, i] = A
    if P["mass_fun"] is not None:
        A_mass[:, i] = P["mass_fun"](FE.Mids).flatten()
    G.append(LOD.lodCorrector(FE, A)[0])
t_lod = (timeit.default_timer()-t0)/N
u0 = P["u0"](FE.vertices_fine) if callable(P["u0"]) else P["u0"]
args = (A_stiff, A_mass, P["rhs"], u0, T)
u_fem, _ = solvers.solveFineFEMParabolic(FE, *args)
u_lod, _ = solvers.solveHomParabolic(FE, util.Bases(G), *args)
u_coarse, _ = solvers.solveCoarseFEMParabolic(FE, *args)
u_frozen, _ = solvers.solveHomParabolic(FE, util.Bases([G[0]]*N), *args)


def rel(u, r):
    e = np.linalg.norm(u[:, 1:]-r[:, 1:], axis=0)/np.linalg.norm(r[:, 1:], axis=0)*100
    return e[-1], e.mean()


print("%s H=1/%d h=1/%d ell=%d | contrast %.3g | LOD basis %.2f s/coefficient | error final/mean [%%]: "
      "coarse FEM %.2f/%.2f, frozen LOD %.2f/%.2f, LOD %.2f/%.2f | max |u_h| %.4g"
      % (example, cfg["N_coarse"][0], cfg["N_coarse"][0]*cfg["N_fine_per_element"][0], cfg["ell"],
         A_stiff.max()/A_stiff.min(), t_lod, *rel(u_coarse, u_fem), *rel(u_frozen, u_fem),
         *rel(u_lod, u_fem), np.abs(u_fem).max()))
