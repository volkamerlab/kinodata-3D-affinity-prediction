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
from kinodata.model import ComplexTransformer, DTIModel, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph

import matplotlib.pyplot as plt
import seaborn as sns

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

def prepare_data(split_fold):
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
    return test_data

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

    with torch.no_grad():
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
    all_deltas = None
    for fold in range(5):
        print(f"Crocodoc for split/5-fold-cv {split}/{fold}")
        print(" > loading data...")
        test_data = prepare_data(fold)
        print(" > loading model...")
        cgnn_3d = load_from_checkpoint(2, split, fold, "CGNN-3D")
        cgnn_3d.train(False)
        for data in tqdm.tqdm(test_data, desc=" > inference..."):
            deltas = inference(data, cgnn_3d)
            deltas["fold"] = fold
            if all_deltas is None:
                all_deltas = deltas
            else:
                all_deltas = pd.concat((all_deltas, deltas), axis="index")
    all_deltas["split_type"] = split
    all_deltas

if __name__ == "__main__":
    main()