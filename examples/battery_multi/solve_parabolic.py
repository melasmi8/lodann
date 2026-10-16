# Author: Mehdi Elasmi
from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import parabolic
import training
import coefficients
from examples.battery_multi.coefficient import layer_kind
from train_model import config

# volumetric heat capacity rho c_p of the layers (cathode collector, active material, anode collector)
RHO = np.array([2706.77, 2094.302, 8710.2])  # density [kg/m^3]
C_P = np.array([897.8, 1010.119, 384.65])  # specific heat capacity [J/(kg K)]


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
    temperature, heat capacity rho c_p and the evaluation cases, in physical units: the
    computational domain (0, 1)^2 is a window of the cell of size L = problem_kwargs["length"] [m];
    then rho c_p d_t u - div(lambda grad u) = q in the window becomes
    L^2 rho c_p d_t u - div(lambda grad u) = L^2 q in (0, 1)^2 (time in s, temperature in K,
    q = problem_kwargs["heat_source"] in the active material [W/m^3], 0 in the collectors)."""
    domain = np.array(config["domain"])
    stack = config["coeff_kwargs"]["stack"]
    L = config["problem_kwargs"]["length"]
    q = config["problem_kwargs"]["heat_source"]
    n_par = coefficients.make_coefficient(config).n_par
    N_eval = 25
    T = np.linspace(0, config["T_final"], N_eval)
    cases = []
    # training parameters: the first N_eval training vectors as a sequence in time (the states
    # jump between time steps; tests the reproduction of the training coefficients)
    trainFile = training.results_path(config)+"trainParameter_Iter0.txt"
    if os.path.exists(trainFile):
        train = np.reshape(np.loadtxt(trainFile), (-1, n_par))[:N_eval]
        cases.append(dict(label="train", coeff=train, T=T[:np.shape(train)[0]], first=0))
    # test: local states of charge during charging (not seen in training), in the relative time
    test = charging_trajectories(n_par, T/config["T_final"])
    cases.append(dict(label="test", coeff=test, T=T, first=1))

    def rhs(x):
        return L**2*(q*np.array([0., 1., 0.])[layer_kind(domain, stack, x[:, 1])])[:, None]

    def mass_fun(x):
        kind = layer_kind(domain, stack, x[:, 1])
        return L**2*(C_P[kind]*RHO[kind])[:, None]

    return dict(rhs=rhs, u0=lambda x: np.zeros((np.size(x, 0), 1)), mass_fun=mass_fun, cases=cases)


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
