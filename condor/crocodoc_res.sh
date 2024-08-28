cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
wandb disabled
if pip show kinodata > /dev/null 2>&1; then
    echo "kinodata is installed."
else
    echo "kinodata is not installed."
    pip install -e .
fi
echo "python3 scripts/crocodoc_residues.py --model_path $1"
python3 scripts/crocodoc_residues.py --model_path $1
