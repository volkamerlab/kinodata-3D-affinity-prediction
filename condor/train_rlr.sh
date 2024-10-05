cd $HOME/raquel/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
pip install -e .
python3 kinodata/training/train.py
