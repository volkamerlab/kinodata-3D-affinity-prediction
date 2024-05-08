#!/bin/bash

user=michael

set -e

ident=$1
echo ${ident}

out_path=data/interaction_analysis
out_path_id=${out_path}/${ident}
mkdir -p $out_path
if [ -e ${out_path}/${ident}.csv ]; then
    exit 0
fi

sudo docker run --rm -v ${PWD}:/results -w /results -u $(id -u ${USER}):$(id -g ${USER}) pharmai/plip:latest -xyq -o $out_path_id -f data/pdbs/${ident}.pdb
sudo chown -R $user data/interaction_analysis/$ident
python scripts/report2csv.py $ident
mv $out_path_id/report.csv $out_path/$ident.csv
rm -rf $out_path_id
