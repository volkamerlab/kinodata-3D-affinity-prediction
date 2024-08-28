cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/crocodoc_ig.py --rmsd $1 --split_type $2 --fold $3
