cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
pip install scikit-learn
pip install -e .
python3 scripts/crocodoc_mask_residues_cgnnx.py --training_run_id $1 --outfile $2
