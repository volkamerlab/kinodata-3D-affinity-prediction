cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
wandb enabled
python3 scripts/$1.py --split_type $2 --split_index $4 --config $5 --filter_rmsd_max_value $3
