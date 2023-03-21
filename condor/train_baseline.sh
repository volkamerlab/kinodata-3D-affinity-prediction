cd /home/jgross/kinodata-docked-rescore
export WANDB_API_KEY=$(cat wandb_api_key)
for ID in 7 11 13 17 19
do	
	python3 train_ligand_gnn_baseline.py --data_split "data/splits/scaffold/scaffold_split_$ID.csv"
done
