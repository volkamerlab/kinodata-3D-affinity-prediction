cd ${HOME}/kinodata-3D-affinity-prediction
pip install scikit-learn
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/train_dimenet.py --split_type $1 --filter_rmsd_max_value $2 --split_index $3