import wandb
wandb.init(mode="disabled")

from kinodata.data import KinodataDocked, Filtered
from kinodata.data.data_module import create_dataset
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.transform import TransformToComplexGraph, FilterDockingRMSD
from kinodata.types import *


import json
from pathlib import Path
from typing import Any

import torch

import kinodata.configuration as cfg
from kinodata.model import RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.transform import TransformToComplexGraph

import pandas as pd
import tqdm

cfg.register(
    "crocodoc",
    split_type="scaffold-k-fold",
    remove_hydrogen=True,
    filter_rmsd_max_value=2.0,
)
config = cfg.get("crocodoc")
config = config.update_from_args()

# transform concatenates pocket and ligand nodes (in that order)
transform = TransformToComplexGraph() # pocket - ligand

def prepare_data(split_fold, relevant_klifs_ids):
    rmsd_filter = FilterDockingRMSD(config["filter_rmsd_max_value"])
    data_cls = Filtered(
        KinodataDocked(),
        rmsd_filter
    )
    test_split = KinodataKFoldSplit(config["split_type"], 5)
    split = test_split.split(data_cls())[split_fold]
    test_data = create_dataset(
        data_cls,
        dict(),
        split.test_split,
        None,
    )
    def is_relevant(data):
        try:
            ident = data["ident"].item()
            relevant_klifs_ids.loc[ident]
            return True
        except KeyError:
            return False
    
    data_list = [data for data in test_data if is_relevant(data)]
    for data in data_list:
        ident = data["ident"].item()
        data["similar.klifs_structure_id"] = relevant_klifs_ids.loc[ident]["similar.klifs_structure_id"]
    return data_list
    

model_dir = Path("models")
assert model_dir.exists(), (model_dir, Path().absolute())

def path_to_model(rmsd_threshold: int, split_type: str, split_fold: int, model_type: str) -> Path:
    p = model_dir / f"rmsd_cutoff_{rmsd_threshold}" / split_type / str(split_fold) / model_type
    if not p.exists():
        p.mkdir(parents=True)
    return p

model_cls = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer
}

def load_wandb_config(
    config_file: Path
) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
    return config

def load_from_checkpoint(
    rmsd_threshold: int,
    split_type: str,
    fold: int,
    model_type: str
) -> RegressionModel:
    cls = model_cls[model_type]
    p = path_to_model(rmsd_threshold, split_type, fold, model_type)
    model_ckpt = list(p.glob("**/*.ckpt"))[0]
    model_config = p / "config.json"
    ckp = torch.load(model_ckpt, map_location="cpu")
    config = cfg.Config(load_wandb_config(model_config))
    model = cls(config)
    assert isinstance(model, RegressionModel)
    model.load_state_dict(ckp["state_dict"])
    return model

@torch.no_grad()
def inference(
    data,
    model,
) -> pd.DataFrame:
    structural_interactions = model.interaction_module.interactions[1]
    def get_pl_edges(data):
        edges, _ = structural_interactions(data)
        n_pocket = data[NodeType.Pocket].z.shape[0]
        edge_in_pocket = edges < n_pocket
        pl_edges = edge_in_pocket[0] ^ edge_in_pocket[1]
        pl_edge_idcs = torch.where(pl_edges)[0]
        return edges, pl_edges, pl_edge_idcs
    
    deltas = []
    source_nodes = []
    target_nodes = []

    graf = transform(data)
    graf[NodeType.Complex].batch = torch.zeros(graf[NodeType.Complex].x.shape[0]).type(torch.int64)
    structural_interactions.hacky_mask = None
    edges, _ = structural_interactions(graf)
    structural_interactions.hacky_mask = torch.ones(len(edges.T)).type(torch.bool)
    edge_index, pl_edges, pl_edge_idcs = get_pl_edges(graf)
    
    reference_prediction = model(graf)
    for candidate_edge in pl_edge_idcs:
        # Compute change against reference prediction
        structural_interactions.hacky_mask[candidate_edge] = False
        deltas.append((reference_prediction - model(graf)[0]).item())
        structural_interactions.hacky_mask[candidate_edge] = True
        source_node, target_node = edge_index.T[candidate_edge]
        source_nodes.append(source_node.item())
        target_nodes.append(target_node.item())
            
    deltas = pd.DataFrame({
        'delta': deltas, 
        "source_node": source_nodes,
        "target_node": target_nodes,
    })
    deltas["reference"] = reference_prediction.flatten().item()
    deltas["ident"] = data.ident.item()
    deltas["n_pocket"] = data[NodeType.Pocket].z.shape[0]
    deltas["n_ligand"] = data[NodeType.Ligand].z.shape[0]
    return deltas

# TODO ligand node idx, pocket node idx, reference prediction, ident, delta

def main():
    split = config.get("split_type")
    relevant_klifs_ids = pd.read_csv("data/crocodoc_relevant_klifs_ids.csv").set_index("ident")
    for fold in [0,1,2,3,4]:
        print(f"Crocodoc for split/5-fold-cv {split}/{fold}")
        print(" > loading data...")
        test_data = prepare_data(fold, relevant_klifs_ids)
        print(" > loading model...")
        cgnn_3d = load_from_checkpoint(2, split, fold, "CGNN-3D")
        cgnn_3d.train(False)
        pbar = tqdm.tqdm(test_data)
        for data in pbar:
            ident = data["ident"].item()
            pbar.set_description(f" > inference")
            deltas = inference(data, cgnn_3d)
            deltas["fold"] = fold
            deltas["klifs_structure_id"] = data["similar.klifs_structure_id"]
            deltas.to_csv(f"data/crocodoc_out/delta_{ident}.csv", index=False)
            

if __name__ == "__main__":
    main()