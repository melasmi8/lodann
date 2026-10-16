#!/usr/bin/env bash
# Author: Mehdi Elasmi
# Training and evaluation of the two examples of the paper.
#   Example 1 (examples/damage_front):  damage front, H = 1/12, h = 1/36, ell = 2
#   Example 2 (examples/battery_multi): battery cell window during fast charging, 48 local states of
#                                       charge, H = 1/12, h = 1/132, ell = 2
# Usage (from the repository root, with the virtual environment activated):
#   bash run_paper_examples.sh train      # train all networks of both examples (and of the
#                                         # network size study of Example 1)
#   bash run_paper_examples.sh evaluate   # solve the time-dependent problems, write all figures
#                                         # and tables (results/<case>/figures/ of each example)
# Further steps:
#   bash run_paper_examples.sh check      # accuracy of the classical LOD against fine FEM for the
#                                         # chosen meshes (no training, CPU)
#   bash run_paper_examples.sh smoke      # short training of two networks per example in
#                                         # /tmp/lodann_smoke (quick test of the installation)
# Environment: N_WORKERS=<n> trains n networks in parallel processes (they share the GPU), e.g.
#   N_WORKERS=4 bash run_paper_examples.sh train
set -e
cd "$(dirname "$0")"
STEP=${1:?"usage: bash run_paper_examples.sh train|evaluate|check|smoke"}
N_WORKERS=${N_WORKERS:-1}
PAR=""
if [ "$N_WORKERS" -gt 1 ]; then PAR="parallelize=True n_workers=$N_WORKERS"; fi

case "$STEP" in
check)
    python lod_accuracy.py damage_front
    python lod_accuracy.py battery_multi
    ;;
smoke)
    for ex in damage_front battery_multi; do
        echo "=== smoke test $ex"
        python examples/$ex/train_model.py epochs=201 finetune_epochs=10 indices=[0,52] \
            assemble_GModel=False result_root=/tmp/lodann_smoke/$ex/
    done
    ;;
train)
    echo "=== Example 1: training"
    python examples/damage_front/train_model.py $PAR
    echo "=== Example 2: training"
    python examples/battery_multi/train_model.py $PAR
    echo "=== Example 1: network size study (training)"
    python examples/damage_front/network_size.py train
    ;;
evaluate)
    echo "=== Example 1: evaluation and figures"
    python make_figures.py examples/damage_front
    echo "=== Example 2: evaluation and figures (with the Deep Ritz baseline)"
    python make_figures.py examples/battery_multi --baseline
    echo "=== Example 1: network size study (evaluation)"
    python examples/damage_front/network_size.py
    ;;
*)
    echo "unknown step: $STEP (train, evaluate, check, smoke)"; exit 1
    ;;
esac
