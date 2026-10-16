# Author: Mehdi Elasmi
# train_model: configuration of the training, see training.DEFAULTS for all options.
# Every entry can be overridden on the command line, e.g. python train_model.py epochs=1001
# Example 2: thermal model of a battery cell window during fast charging, with physical data.
# Geometry: stack of 6 x [cathode collector (CC, Al), active material (AM), anode collector (AC, Cu),
# AM], collector : AM thickness = 1 : 10 (16.5 um : 165 um), window of 2.178 mm x 2.178 mm, all
# layer interfaces on the fine mesh (h = 1/132, one fine cell per collector layer).
# AM conductivity: rough random field, log-uniform in [1, 5] W/(m K) (lambda_base), scaled by the
# local state of charge p_r of its region (12 AM layers x 4 segments = 48 regions/parameters):
# lambda = lambda_base 1.5^(2 p_r - 1), i.e., by a factor in [1/1.5, 1.5] (the thermal conductivity
# of LiCoO2 changes by a factor ~1.5 with lithiation). Collectors: 236.3 (Al), 398.71 (Cu) W/(m K).
# Heat capacities in battery_physics.py. Each element network only gets the parameters of the
# regions intersecting its patch.
# Evaluation (solve_parabolic.py): heat source 2e5 W/m^3 in the AM, homogeneous Dirichlet data,
# 20 min of charging (T_final = 1200 s, 24 time steps), local states of charge rising from
# [0, 0.2] to [0.8, 1] at region-dependent rates.
# Mesh: H = 1/12, ell = 2 (check the LOD accuracy first: python lod_accuracy.py battery_multi).
from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
import training

config = training.make_config(
    coeff_fun="battery_multi",
    coeff_kwargs={"ccc": 236.3, "acc": 398.71, "am_min": 1., "am_max": 5., "n_eps": 132,
                  "n_seg": 4, "mode": "magnitude", "factor": 1.5, "seed": 73,
                  "stack": {"n_rep": 6, "t_coll": 1, "t_am": 10, "closing": False}},
    domain=[[0, 1], [0, 1]],  # computational domain, the window of size problem_kwargs["length"]
    N_coarse=[12, 12],  # H = 1/12
    N_fine_per_element=[11, 11],  # h = 1/132
    ell=2,
    elementwise=True,  # one network per coarse element, correctors as in eq. (7) of the paper
    epsilon_min=0.0,  # range of each region parameter (local state of charge)
    epsilon_max=1.0,
    N_coeff=256,  # random parameter vectors (Latin hypercube)
    N_test=16,
    coeff_batch=16,  # coefficients per Adam step
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
    exact_constraint=True,  # interpolation constraint enforced by the projection of the output
    normalize=True,  # input normalization and output scaling
    # training stages: 0: Adam only, 1: Adam + L-BFGS fine tuning, 2: fine tuning of a trained case
    mode=1,
    finetune_optimizer="lbfgs",  # "lbfgs" or "adam"
    finetune_epochs=1000,  # L-BFGS iterations (or Adam steps)
    lbfgs_coeffs=32,  # fixed subset of the coefficients for L-BFGS
    assemble_GModel=False,  # the networks have different inputs
    parallelize=False,
    # evaluation: physical units (window size [m], heat source in the AM [W/m^3]), charging scenario
    T_final=1200.,
    problem_kwargs={"length": 2.178e-3, "heat_source": 2e5, "scenario": "charge"},
    result_root=str(Path(__file__).parent / "results")+"/",
)

if __name__ == '__main__':
    training.run_training(training.config_from_argv(config))
