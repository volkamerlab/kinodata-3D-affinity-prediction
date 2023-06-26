
cd /home/jgross/kinodata-docked-rescore
export WANDB_API_KEY=$(cat wandb_api_key)
for ID in 0 1 2 3 4
do	
	python3 scripts/$1.py --data_split "data/splits/$2/seed_$ID.csv"
done
