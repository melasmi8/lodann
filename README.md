# LOD-ANN: neural numerical homogenization based on Deep Ritz corrections

Code for the numerical examples of the paper

> M. Elasmi, F. Krumbiegel, R. Maier: *Neural numerical homogenization based on Deep Ritz corrections*.

LOD-ANN approximates the corrections of the localized orthogonal decomposition (LOD) by neural
networks that take the spatial coordinates and the parameters of the coefficient as input. One
network is trained per coarse element by minimizing the energy of the local correction problems
for a set of parameter samples (Deep Ritz approach, no precomputed corrections needed). The
interpolation constraint of the corrections is enforced exactly by a fixed projection of the network
output. Once trained, the networks provide the LOD basis for any parameter by a forward pass, which
is used to solve time-dependent problems with changing coefficients.

Author of the code: Mehdi Elasmi (`util.py` and `LOD.py`: Mehdi Elasmi and Felix Krumbiegel).

## Installation

Python 3.11 in a virtual environment:
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
The training runs on a GPU if TensorFlow finds one (for TensorFlow 2.14 with CUDA on Linux, e.g.,
`pip install "tensorflow[and-cuda]==2.14.0"`). The figures are written as LaTeX files (pgfplots)
and compiled with `pdflatex` if it is installed.

## Reproducing the results of the paper

From the repository root, with the virtual environment activated:

**Training** of all networks (both examples and the network size study of Example 1):
```
bash run_paper_examples.sh train
```

**Evaluation**: solves the time-dependent problems with fine FEM (reference), coarse FEM, LOD with a
frozen basis, LOD, and LOD-ANN (Example 2 also with a classical Deep Ritz network), and writes all
figures and tables:
```
bash run_paper_examples.sh evaluate
```

`N_WORKERS=4 bash run_paper_examples.sh train` trains four networks in parallel processes, which
share the GPU. Two further steps are available: `bash run_paper_examples.sh check` prints the accuracy
of the classical LOD against fine FEM for the chosen meshes (no training), and
`bash run_paper_examples.sh smoke` trains two networks per example for a few epochs (quick test of
the installation, results in `/tmp/lodann_smoke`).

The results are written to `examples/<example>/results/<case>/`: the trained networks
(`correctedHatModel<K>.h5`, one per coarse element K), the training logs (`logs/`, TensorBoard), the
configuration (`config.json`) and, in `figures/`, the figures of the paper as PDFs (with their
LaTeX sources and data), `summary.json` and `summary_table.tex` (errors and costs) and an overview
`figures.pdf`. The network size study is written to `examples/damage_front/results_size/`.

The two examples of the paper:

| Example | folder | coefficient | mesh |
|---|---|---|---|
| 1: damage front | `examples/damage_front` | rough conductivity with damaged regions growing with one parameter p | H = 1/12, h = 1/36, ℓ = 2 |
| 2: battery cell with multiple parameters | `examples/battery_multi` | layered battery cell window (physical data), conductivity of the active material depending on 48 local states of charge | H = 1/12, h = 1/132, ℓ = 2 |

## Running a single example

Training (here Example 1):
```
python examples/damage_front/train_model.py
```
Evaluation with all figures and tables:
```
python make_figures.py examples/damage_front
```
Options of `make_figures.py`: `--baseline` (also the classical Deep Ritz network on the whole domain,
trained in every time step), `--no-compile` (only write the LaTeX files), `--vector-fields` (fields as
vector graphics). A quick evaluation without figures (errors of LOD-ANN against LOD in the terminal):
```
python examples/damage_front/solve_parabolic.py
```

Every entry of the configuration (`train_model.py` of the example, all options with their defaults in
`training.DEFAULTS`) can be overridden on the command line with `key=value`. The same overrides have
to be passed to `make_figures.py` and `solve_parabolic.py`, since they determine the results folder.
For example, a test with smaller networks on a coarser mesh:
```
python examples/damage_front/train_model.py width_net=32 depth_net=4 N_coarse=[6,6] N_fine_per_element=[6,6] ell=1
python make_figures.py examples/damage_front width_net=32 depth_net=4 N_coarse=[6,6] N_fine_per_element=[6,6] ell=1
```
Frequently used options:

| option | meaning |
|---|---|
| `N_coarse`, `N_fine_per_element`, `ell` | coarse mesh (elements per direction), fine elements per coarse element, localization parameter |
| `width_net`, `depth_net` | network size |
| `N_coeff`, `coeff_batch` | number of training parameters, coefficients per Adam step (0: all) |
| `epochs`, `initial_lr`, `decay_rate`, `decay_steps` | Adam (with early stopping when the loss stagnates) |
| `mode`, `finetune_epochs` | 0: Adam only, 1: Adam followed by L-BFGS with `finetune_epochs` iterations (default), 2: L-BFGS fine tuning of a trained case |
| `parallelize`, `n_workers` | train several networks in parallel processes |
| `indices` | train only the networks of the given coarse elements, e.g. `indices=[0,52]` |
| `result_root` | folder of the results |

Before training, `python lod_accuracy.py <example> [key=value ...]` shows the accuracy of the
classical LOD against fine FEM for a mesh and localization parameter (no networks needed), e.g.
`python lod_accuracy.py damage_front N_coarse=[6,6] N_fine_per_element=[6,6] ell=1`.

The training progress can be followed with TensorBoard: `tensorboard --logdir examples/damage_front/results/<case>`.

## Creating a new example

1. **Coefficient.** Add a class `PDEcoeff_<name>` to `PDEcoeff_functions.py` (see `PDEcoeff_front` for
   one parameter and `PDEcoeff_battery_multi` for several parameters):
   - `__init__(self, domain, **kwargs)`: the keyword arguments are set by `coeff_kwargs` in the configuration;
   - `__call__(self, x, p)`: values of the coefficient at the points `x` (array of shape `(N, d)`, the
     fine element midpoints) for each parameter in the list `p`, stacked as a column vector of length
     `N len(p)`;
   - for a vector of parameters: the attribute `n_par` (number of parameters) and, optionally, the
     method `relevant_parameters(x)` that returns the indices of the parameters on which the
     coefficient at the points `x` depends. Each network then only receives these parameters.
2. **Example folder.** Create `examples/<my_example>/` with an empty `__init__.py`, a `results/` folder and
   - `train_model.py`: the configuration, `config = training.make_config(coeff_fun="<name>", coeff_kwargs={...}, ...)`,
     with `result_root` set to the `results/` folder of the example and
     `training.run_training(training.config_from_argv(config))` as main program (copy
     `examples/damage_front/train_model.py`);
   - `solve_parabolic.py`: a function `problem(config)` that returns a dict with the right-hand side
     `rhs` and the initial value `u0` (functions of the points, returning column vectors), the
     heat capacity `mass_fun` (function of the points, or `None` for 1), and a list `cases` of
     evaluation cases `dict(label=..., coeff=..., T=..., first=...)`: the parameters of the
     coefficient in each time step (`"train_coeff"` for the training parameters), the time points
     (`None`: equidistant in [0, 1], one per parameter), and the first time step to plot (copy
     `examples/damage_front/solve_parabolic.py`).
3. **Run** `python examples/<my_example>/train_model.py` and `python make_figures.py examples/<my_example>`.

## Files

| file | content |
|---|---|
| `training.py` | configuration (`DEFAULTS`) and training driver for all examples |
| `LODANN.py` | network of one coarse element: localization, normalization, exact interpolation constraint, energy loss, Adam and L-BFGS |
| `DiscreteSetting.py`, `Geo.py` | fine and coarse meshes, patches, local system matrices and interpolation |
| `LOD.py`, `util.py` | classical LOD correctors, finite element assembly and utilities |
| `PDEcoeff_functions.py`, `battery_physics.py` | coefficients of the examples, heat source and heat capacity of the battery cell |
| `solvers.py`, `parabolic.py`, `evaluate_model.py` | backward Euler solvers (fine FEM, coarse FEM, LOD), evaluation of the trained networks |
| `make_figures.py`, `figures.py`, `baselines.py` | figures and tables of the paper, classical Deep Ritz baseline |
| `lod_accuracy.py` | accuracy of the classical LOD for a mesh (before training) |
| `run_paper_examples.sh` | training and evaluation of the examples of the paper |
| `examples/damage_front/network_size.py` | network size study of Example 1 |
