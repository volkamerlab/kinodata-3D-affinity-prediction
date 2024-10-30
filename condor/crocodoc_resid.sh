cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/crocodoc_residues.py --model_type $1 --filter_max_rmsd_value $2 --split_type $3
