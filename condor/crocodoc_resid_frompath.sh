cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/crocodoc_residues.py --model_path data/cgnn_new/$1 --outfile $2
