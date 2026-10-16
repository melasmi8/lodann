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
from train_model import config


def problem(config):
    """Time-dependent problem of the example (also used by make_figures.py): right-hand side,
    initial value, heat capacity (None: 1) and the evaluation cases (label, coefficient
    parameters, time points (None: equidistant in [0, 1]), first time step to plot)."""
    # test: random parameters (sorted in time), different from the training ones
    test = np.sort(np.random.default_rng(37).uniform(
        config["epsilon_min"], config["epsilon_max"], 25))
    return dict(rhs=lambda x: np.ones((np.size(x, 0), 1)),
                u0=lambda x: np.zeros_like(x[:, [1]]),
                mass_fun=None,
                cases=[dict(label="train", coeff="train_coeff", T=None, first=0),
                       dict(label="test", coeff=test, T=None, first=1)])


if __name__ == '__main__':
    # same command line overrides as for train_model.py, e.g. ell=2 elementwise=True
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
