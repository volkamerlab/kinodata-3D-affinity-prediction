cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/train_voxel_model.py --split_type $1 --fold $2 --hidden_channels $3 --batch_size $4 --acc_grad_batches $5 --kernel_sizes $6 --lr $7 --lr_decay $8 --pool_every $9
