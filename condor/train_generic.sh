cd ${HOME}/kinodata-3D-affinity-prediction
pip install scikit-learn
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/$1.py --split_type $2 --filter_rmsd_max_value $3 --split_index $4 --config $5
