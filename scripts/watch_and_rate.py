from pathlib import Path
from typing import Any, Union
import json
from functools import cache
import sys
import os
from os import listdir
from os.path import isfile, join
import time

import numpy as np
import torch
from torch_geometric.data import HeteroData, collate
from rdkit import Chem
from rdkit.Chem import AddHs, Kekulize, MolFromMol2File, PandasTools, rdchem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Geometry import Point3D
import wandb
import pandas as pd
import tqdm

from kinodata import configuration as cfg
from kinodata.model import ComplexTransformer, DTIModel, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph
from kinodata.data.utils.scaffolds import mol_to_scaffold
from kinodata.data.featurization.rdkit import (
    set_atoms,
    set_bonds,
)
from kinodata.types import NodeType

MODEL_CLS = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer,
}


def path_to_model(
    model_dir: Path,
    rmsd_threshold: int,
    split_type: str,
    split_fold: int,
    model_type: str,
) -> Path:
    p = (
        model_dir
        / f"rmsd_cutoff_{rmsd_threshold}"
        / split_type
        / str(split_fold)
        / model_type
    )
    if not p.exists():
        p.mkdir(parents=True)
    return p


def load_wandb_config(config_file: Path) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
    return config


def load_from_checkpoint(
    model_dir: Path, rmsd_threshold: int, split_type: str, fold: int, model_type: str
) -> RegressionModel:
    cls = MODEL_CLS[model_type]
    p = path_to_model(model_dir, rmsd_threshold, split_type, fold, model_type)
    model_ckpt = list(p.glob("**/*.ckpt"))[0]
    model_config = p / "config.json"
    ckp = torch.load(model_ckpt, map_location="cpu")
    config = cfg.Config(load_wandb_config(model_config))
    model = cls(config)
    assert isinstance(model, RegressionModel)
    model.load_state_dict(ckp["state_dict"])
    return model


def make_pyg(pocket: Mol, ligand: Mol, batch: int = 0) -> HeteroData:
    data = HeteroData()

    data = set_atoms(ligand, data, NodeType.Ligand)
    data = set_bonds(ligand, data, NodeType.Ligand)

    data = set_atoms(pocket, data, NodeType.Pocket)
    data = set_bonds(pocket, data, NodeType.Pocket)

    return data


def read_molecule(path: Union[str, Path], remove_hydrogen: bool = True) -> Any:
    path = Path(path)
    if path.suffix == ".mol":
        molecule = MolFromMolFile(str(path))
    if path.suffix == ".mol2":
        molecule = MolFromMol2File(str(path), sanitize=False)
    elif path.suffix == ".sdf":
        suppl = SDMolSupplier(str(path))
        molecule = next(iter(suppl))
    else:
        raise NotImplementedError(f"Unable to handle {path.suffix}-files.")

    if molecule is None:
        raise ValueError(f"Invalid file {path}")

    if not remove_hydrogen:
        AddHs(molecule)
    Kekulize(molecule)

    return molecule


def read_models(
    model_dir: Path = Path(".") / "models",
    threshold: int = 2,
    split: str = "scaffold-k-fold",
):
    model_dir = Path(".") / "models"
    assert model_dir.exists()
    print("load models")
    models = [
        load_from_checkpoint(model_dir, threshold, split, fold, "CGNN-3D")
        for fold in tqdm.tqdm(range(5))
    ]
    for model in models:
        model.eval()
    return models


def fname2ident(filename: Union[str, Path]) -> str:
    filename = Path(filename)
    return filename.stem


def main():
    wandb.init(mode="disabled")
    pocket_file = sys.argv[1]
    ligand_dir = Path(sys.argv[2])
    assert Path(ligand_dir).exists(), f"invalid ligand path: {ligand_dir}"
    predictions_file = Path(sys.argv[3])
    if predictions_file.exists():
        preds = pd.read_csv(predictions_file, names=["ident", "prediction"])
        rated_ligands = list(preds["ident"].values)
    else:
        rated_ligands = list()

    models = read_models()
    transform = TransformToComplexGraph(remove_heterogeneous_representation=True)
    pocket = read_molecule(pocket_file)

    print("watch for new ligands")
    while True:
        ligand_files = [
            Path(ligand_dir) / f
            for f in listdir(str(ligand_dir))
            if isfile(join(ligand_dir, f)) and fname2ident(f) not in rated_ligands
        ]

        if len(ligand_files) == 0:
            time.sleep(1)
            continue

        data_list = list()
        for ligand_file in ligand_files:
            ligand = read_molecule(ligand_file)
            data = make_pyg(pocket, ligand)
            data = transform(data)
            data_list.append(data)
        data, _, _ = collate.collate(HeteroData, data_list, add_batch=True)
        print("apply models to", list(map(fname2ident, ligand_files)))
        with torch.no_grad():
            results = torch.stack([model(data) for model in models])

        with open(str(predictions_file), "a") as file:
            for i, filename in enumerate(ligand_files):
                ident = fname2ident(filename)
                pred_vec = results[:, i, :].squeeze()
                file.write(
                    f"{ident}," + ",".join([str(p.item()) for p in pred_vec]) + "\n"
                )
                rated_ligands.append(ident)


if __name__ == "__main__":
    main()
