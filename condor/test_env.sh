cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
echo "pip freeze"
pip freeze
echo "pip3 freeze"
pip3 freeze
wandb disabled
python3 scripts/test_env.py
