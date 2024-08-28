cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
wandb disabled
python3 scripts/generate_dataset.py
