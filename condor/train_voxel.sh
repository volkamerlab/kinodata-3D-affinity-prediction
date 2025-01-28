cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/train_voxel_model.py --split_type $1 --fold $2 --hidden_channels $3
