cd ${HOME}/kinodata-3D-affinity-prediction
pip install scikit-learn
export WANDB_API_KEY=$(cat wandb_api_key)
SCRIPT="$1"
shift  # Remove the first argument
python3 scripts/${SCRIPT}.py "$@"
