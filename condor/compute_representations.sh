#!/bin/bash
cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
unrolled_args=($@)
echo unrolled_args: ${unrolled_args[@]}
pip install scikit-learn
python3 scripts/compute_model_reprs.py ${unrolled_args[@]}
