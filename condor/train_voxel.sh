cd ${HOME}/kinodata-3D-affinity-prediction
pip install lightning==2.5.0
export WANDB_API_KEY=$(cat wandb_api_key)
cd docktgrid && pip install .
cd ..
python3 scripts/train_voxel_model.py --split_type $1 --fold $2 --hidden_channels $3
