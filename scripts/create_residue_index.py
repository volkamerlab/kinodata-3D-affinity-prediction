from tqdm import tqdm
import json
from collections import defaultdict

import pandas as pd
import numpy as np
from pathlib import Path

from kinodata.data.io.read_klifs_mol2 import read_klifs_mol2
from kinodata.data.utils.remap_residue_index import remap_residue_index


print("Reading base data...")
df = pd.read_csv('data/processed/kinodata3d.csv')


pbar = tqdm(df.iterrows(), total=len(df))
for _, row in pbar:
    ident = int(row["ident"])
    target_fname = Path("data") / "processed" / "residue_atom_index" / f"ident_{ident}.json"
    if target_fname.exists():
        continue
    pocket_file = Path(row["pocket_mol2_file"])
    pocket_df = read_klifs_mol2(pocket_file, with_bonds=False)
    pocket_df = pocket_df[pocket_df["atom.type"].str.upper() != "H"].reset_index()
    residue_idx = pocket_df["residue.subst_id"].values
    residue_idx = remap_residue_index(row["structure.pocket_sequence"], residue_idx)
    lookup_table = defaultdict(list)
    atom_idx = pocket_df.index.values
    for r, a in zip(residue_idx, atom_idx):
        lookup_table[int(r)].append(int(a))
    with open(target_fname, "w") as f:
        json.dump(lookup_table, f)
    del lookup_table
    del target_fname
    del pocket_df
    del residue_idx
    del atom_idx