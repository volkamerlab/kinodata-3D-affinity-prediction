cd ${HOME}/kinodata-3D-affinity-prediction
export WANDB_API_KEY=$(cat wandb_api_key)
python3 scripts/train_voxel_model.py --split_type $1 --fold $2 --hidden_channels $2 --batch_size $3 --acc_grad_batches $4 --kernel_sizes $5 --lr $6 --lr_decay $7 --random_rotation_augmentations $8