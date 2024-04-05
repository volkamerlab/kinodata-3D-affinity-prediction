cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/$1.py --split_type $2 --split_index $5 --config $4 --filter_rmsd_max_value $3
