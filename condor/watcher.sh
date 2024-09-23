#!/bin/bash

set -e
cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
pip install -e .
python scripts/watch_and_rate.py target.mol2 $HOME/ligands predictions.csv
