cd ${HOME}/kinodata-3D-affinity-prediction
pip install resmo
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/doc_the_voxel.py --fold $1 --model $2
