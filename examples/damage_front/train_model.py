# Author: Mehdi Elasmi
# train_model: configuration of the training, see training.DEFAULTS for all options.
# Every entry can be overridden on the command line, e.g. python train_model.py epochs=1001
# Example 1 of the paper: damage front. A rough base conductivity in [0.3, 1] is reduced by the
# factor 10 in damaged regions that nucleate, grow and connect as p increases (PDEcoeff_front:
# threshold of a correlated random field, smoothed over the width delta). Strongly nonlinear in p:
# a basis computed for one p is ~14 % off for others.
from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
import training

config = training.make_config(
    coeff_fun="front",
    coeff_kwargs={"n_eps": 36, "a_min": 0.3, "a_max": 1.0, "kappa": 0.1, "frac_max": 0.7,
                  "delta": 0.015, "corr": 0.1, "seed": 73},
    domain=[[0, 1], [0, 1]],
    # H = 1/12, h = 1/36, ell = 2: classical LOD ~1 % error against fine FEM at the final time
    # (H = 1/6, ell = 1: ~5 %; with H = 1/6 a larger ell does not help, the error is dominated by H)
    N_coarse=[12, 12],  # nbr of elements
    N_fine_per_element=[3, 3],
    ell=2,
    epsilon_min=0.0,  # parameter range of p
    epsilon_max=1.0,
    N_coeff=40,
    N_test=25,
    epochs=30001,
    freq_log=100,
    width_net=128,
    depth_net=8,
    initial_lr=1E-3,
    decay_rate=0.9,
    decay_steps=1000,
    lr_sa=1E-2,
    decay_rate_sa=0.9,
    decay_steps_sa=1000,
    elementwise=True,  # one network per coarse element, correctors as in eq. (7) of the paper
    exact_constraint=True,  # interpolation constraint enforced by the projection of the output
    normalize=True,  # input normalization and output scaling
    # training stages: 0: Adam only, 1: Adam + L-BFGS fine tuning, 2: fine tuning of a trained case
    mode=1,
    finetune_optimizer="lbfgs",  # "lbfgs" or "adam"
    finetune_epochs=2000,  # L-BFGS iterations (or Adam steps)
    parallelize=False,
    result_root=str(Path(__file__).parent / "results")+"/",
)

if __name__ == '__main__':
    training.run_training(training.config_from_argv(config))
