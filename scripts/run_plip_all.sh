#!/bin/bash
set -e

ls -1 data/pdbs|sed 's/.pdb//g' > idents
parallel --jobs 6 sh scripts/plip_pose.sh :::: idents
rm idents
