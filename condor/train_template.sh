
cd /home/jgross/kinodata-docked-rescore
export WANDB_API_KEY=$(cat wandb_api_key)
for ID in 0 1 2 3 4
do	
	python3 scripts/$1.py --split_type $2 --split_index $ID --config $4.yaml --filter_rmsd_max_value $3
done
