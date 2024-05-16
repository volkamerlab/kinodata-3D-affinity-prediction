import kinodata
from kinodata.data import KinodataDocked, Filtered
from kinodata.data.data_module import create_dataset
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.transform import TransformToComplexGraph, FilterDockingRMSD
from kinodata.types import *
from kinodata.data.utils.dataset_key import KinodataChemblKey


import json
from pathlib import Path
from typing import Any
import requests as req
import functools

import torch

import kinodata.configuration as cfg
from kinodata.model import ComplexTransformer, DTIModel, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph
from kinodata.data.io.read_klifs_mol2 import read_klifs_mol2

import pandas as pd
import numpy as np
import tqdm
import wandb

pocketfile = (
    lambda ident: f'data/raw/mol2/pocket/{df.loc[ident]["similar.klifs_structure_id"]}_pocket.mol2'
)

bitvec = lambda mask: np.array([c == "1" for c in mask])


@functools.cache
def klifs_encoding():
    resp = req.get("https://klifs.net/api_v2/interactions_get_types")

    encodings = dict()
    for entry in resp.json():
        key = ["0"] * 7
        key[int(entry["position"]) - 1] = "1"
        encodings["".join(key)] = entry["name"]

    return encodings


@functools.cache
def klifs_ifp(klifs_id):
    resp = req.get(
        "https://klifs.net/api_v2/interactions_get_IFP", {"structure_ID": klifs_id}
    )
    resp.raise_for_status()
    ifp = resp.json()[0]["IFP"]
    interactions = []
    residues = []
    for i in range(85):
        res_interactions = bitvec(ifp[i * 7 : i * 7 + 7])
        for mask, interaction in klifs_encoding().items():
            if np.any(bitvec(mask) & res_interactions):
                residues.append(i)
                interactions.append(interaction)
    return pd.DataFrame({"residue": residues, "klifs_interaction": interactions})


def ligand_atom_histogram(
    dataset: KinodataDocked,
    meta: pd.DataFrame,
    key: KinodataChemblKey,
    identaids: pd.DataFrame,
):
    all_deltas = list()
    for _, row in tqdm.tqdm(identaids.iterrows(), total=len(identaids)):
        try:
            ident = row["activities.activity_id"]
            demo_data = dataset[key[ident]]
            n_pocket = len(demo_data["pocket"]["z"])
            n_ligand = len(demo_data["ligand"]["z"])

            ident_ = identaids.set_index("activities.activity_id").loc[ident]["ident"]
            deltas = pd.read_csv(f"data/crocodoc_out/delta_{ident_}.csv")
            deltas["ligand_node"] = [
                max(row["source_node"], row["target_node"]) - n_pocket
                for _, row in deltas.iterrows()
            ]
            ligand_df = pd.DataFrame(
                {
                    "atom_id": np.arange(n_ligand),
                    "ligand_atom_type": demo_data["ligand"]["z"],
                }
            )
            deltas = deltas.merge(ligand_df, left_on="ligand_node", right_on="atom_id")

            all_deltas.append(deltas)
        except:
            pass
    pd.concat(all_deltas).to_csv("data/ligand_atom_deltas.csv")


def pl_deltas(
    dataset: KinodataDocked,
    meta: pd.DataFrame,
    key: KinodataChemblKey,
    identaids: pd.DataFrame,
):
    all_deltas = list()
    for _, row in tqdm.tqdm(identaids.iterrows(), total=len(identaids)):
        try:
            ident = row["activities.activity_id"]
            demo_data = dataset[key[ident]]
            n_pocket = len(demo_data["pocket"]["z"])
            n_ligand = len(demo_data["ligand"]["z"])

            ident_ = identaids.set_index("activities.activity_id").loc[ident]["ident"]
            deltas = pd.read_csv(f"data/crocodoc_out/delta_{ident_}.csv")
            deltas["ligand_node"] = [
                max(row["source_node"], row["target_node"]) - n_pocket
                for _, row in deltas.iterrows()
            ]
            deltas["pocket_node"] = [
                min(row["source_node"], row["target_node"])
                for _, row in deltas.iterrows()
            ]
            ligand_df = pd.DataFrame(
                {
                    "atom_id": np.arange(n_ligand),
                    "ligand_atom_type": demo_data["ligand"]["z"],
                    "ligand_atom_pos": list(
                        demo_data["ligand"]["pos"].detach().numpy()
                    ),
                }
            )
            pocket_df = pd.DataFrame(
                {
                    "atom_id": np.arange(n_pocket),
                    "pocket_atom_type": demo_data["pocket"]["z"],
                    "pocket_atom_pos": list(
                        demo_data["pocket"]["pos"].detach().numpy()
                    ),
                }
            )
            deltas = deltas.merge(ligand_df, left_on="ligand_node", right_on="atom_id")
            deltas = deltas.merge(pocket_df, left_on="pocket_node", right_on="atom_id")
            deltas["edge_length"] = np.sqrt(
                np.stack(
                    (deltas["ligand_atom_pos"] - deltas["pocket_atom_pos"]) ** 2
                ).sum(1)
            )

            all_deltas.append(deltas)
        except:
            pass
    pd.concat(all_deltas).to_csv("data/pl_deltas.csv")


def main():
    wandb.init(mode="disabled")
    dataset = KinodataDocked()
    print('reading meta data')
    df = pd.read_csv(
        "data/raw/kinodata3d_meta.csv", index_col="activities.activity_id"
    )
    key = KinodataChemblKey(dataset)
    identaids = pd.read_csv("data/processed/ident_aids.csv")
    crocodoc_done = pd.read_csv(
        "data/crocodoc_done.csv", header=None, names=["ident"]
    )
    identaids = identaids.merge(crocodoc_done, how="inner")
    print('start analysis')
    ligand_atom_histogram(dataset, df, key, identaids)
    pl_deltas(dataset, df, key, identaids)


if __name__ == "__main__":
    main()
