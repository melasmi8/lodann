# Author: Mehdi Elasmi
from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import parabolic
import battery_physics
import training
import PDEcoeff_functions
from train_model import config


def state_trajectories(n_par, T, lo=0., hi=1., seed=37):
    # smooth random evolution of the region states: p_r(t) = mid + amp sin(2 pi (f_r t + phi_r)),
    # with frequencies f_r in [0.5, 1.5] and random phases (one row per time point)
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.5, 1.5, n_par)
    phi = rng.uniform(0., 1., n_par)
    mid, amp = (lo+hi)/2, 0.45*(hi-lo)
    return mid + amp*np.sin(2*np.pi*(np.outer(T, f) + phi))


def charging_trajectories(n_par, T, seed=37):
    # local states of charge during a charging process (T: relative times in [0, 1]): each region
    # is charged from s0_r in [0, 0.2] to s1_r in [0.8, 1] at its own pace,
    # p_r(t) = s0_r + (s1_r - s0_r) t^gamma_r with gamma_r in [0.6, 1.6] (one row per time point)
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(0., 0.2, n_par)
    s1 = rng.uniform(0.8, 1., n_par)
    gamma = rng.uniform(0.6, 1.6, n_par)
    return s0 + (s1-s0)*np.asarray(T, dtype=float)[:, None]**gamma


def problem(config):
    """Time-dependent problem of the example (also used by make_figures.py): heat source, initial
    temperature, heat capacity rho c_p (battery_physics.py) and the evaluation cases.
    problem_kwargs (config): physical units, i.e., the computational domain (0, 1)^2 is a window of
    the cell of size length [m]; then rho c_p d_t u - div(lambda grad u) = q in the window becomes
    length^2 rho c_p d_t u - div(lambda grad u) = length^2 q in (0, 1)^2 (time in s, temperature in K,
    q: heat source in the active material [W/m^3]), and scenario "charge" (local states of charge
    during charging) or "oscillating" (sinusoidal states, default)."""
    domain = np.array(config["domain"])
    pk = config.get("problem_kwargs", {})
    L = pk.get("length", 1.)
    q = pk.get("heat_source", 3.)
    stack = config["coeff_kwargs"].get("stack")
    n_par = PDEcoeff_functions.make_PDEcoeff(
        config["coeff_fun"], domain, **config["coeff_kwargs"]).n_par
    N_eval = 25
    T = np.linspace(0, config["T_final"], N_eval)
    rhs0 = battery_physics.heat_source(domain, q=q, stack=stack)
    mass0 = battery_physics.heat_capacity(domain, stack=stack)
    cases = []
    # training parameters: the first N_eval training vectors as a sequence in time (the states
    # jump between time steps; tests the reproduction of the training coefficients)
    trainFile = training.results_path(config)+"trainParameter_Iter0.txt"
    if os.path.exists(trainFile):
        train = np.reshape(np.loadtxt(trainFile), (-1, n_par))[:N_eval]
        cases.append(dict(label="train", coeff=train, T=T[:np.shape(train)[0]], first=0))
    # test: evolution of all region states in time (not seen in training), in the relative time
    if pk.get("scenario", "oscillating") == "charge":
        test = charging_trajectories(n_par, T/config["T_final"])
    else:
        test = state_trajectories(n_par, T/config["T_final"], config["epsilon_min"],
                                  config["epsilon_max"])
    cases.append(dict(label="test", coeff=test, T=T, first=1))
    return dict(rhs=lambda x: L**2*rhs0(x),
                u0=battery_physics.initial_temperature(),
                mass_fun=lambda x: L**2*mass0(x),
                cases=cases)


if __name__ == '__main__':
    # same command line overrides as for train_model.py
    config = training.config_from_argv(config)
    loadPath = training.results_path(config)
    print("+ Path to config file: ", loadPath)
    assert os.path.exists(
        loadPath), "This case has not been trained yet! Start train_model first!"
    P = problem(config)
    for case in P["cases"]:
        res = parabolic.run_case(loadPath, case["coeff"], case["T"], P["rhs"], P["u0"],
                                 mass_fun=P["mass_fun"], label=case["label"])
        parabolic.plot_solutions(loadPath, res, case["label"], first=case["first"])
