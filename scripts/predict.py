from functools import partial
import json
from pathlib import Path
from typing import Any, Optional
from argparse import ArgumentParser
import sys
import logging
import hashlib


import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from rdkit.Chem import PandasTools  # type: ignore
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from kinodata.data.featurization.biopandas import add_pocket_information
from kinodata.data.featurization.rdkit import set_atoms, set_bonds
from kinodata.types import NodeType
import kinodata.configuration as configuration
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.data.dataset import (
    apply_transform_instance_permament,
    ComplexInformation,
    _DATA,
)
from kinodata.transform.to_complex_graph import TransformToComplexGraph
from kinodata.configuration import Config

logging.basicConfig(level=logging.INFO)
PROCESSED = _DATA / "processed"
RAW = _DATA / "raw"
if not PROCESSED.exists():
    PROCESSED.mkdir()
if not RAW.exists():
    RAW.mkdir()


def read_config(config_file: Path) -> dict[str, Any]:
    assert config_file.exists()
    raw_dict = json.loads(config_file.read_text())
    return {key: value["value"] for key, value in raw_dict.items()}


def patch_config(config: Config) -> Config:
    return config


def load_data(multi_sdf_file: Path, remove_hydrogen: bool = True) -> list[HeteroData]:
    logging.info(f"Loading data for {multi_sdf_file}")
    md5 = hashlib.md5(multi_sdf_file.read_bytes()).hexdigest()
    if (cached_data_list := PROCESSED / f"md5.pt").exists():
        logging.info(f"Using cached processed data for md5 {md5}")
        ...  # TODO load cached data
    logging.info(f"Creating torch geometric HeteroData objects")
    df_ligands = PandasTools.LoadSDF(str(multi_sdf_file))
    data_list = []
    for index, row in tqdm(df_ligands.iterrows(), total=df_ligands.shape[0]):
        klifs_id = row["klifs.structure_id"]
        complex_info = ComplexInformation(
            molecule=row["ROMol"],
            kinodata_ident=row["activities.activity_id"],
            compound_smiles=row["smiles"],
            activity_value=0,
            activity_type="",
            pocket_mol2_file=RAW / "pocket" / f"{klifs_id}_pocket.mol2",
            docking_score=float(row["Chemgauss4"]),
            posit_probability=float(row["POSIT::Probability"]),
            klifs_structure_id=int(klifs_id),
            pocket_sequence="",
            predicted_rmsd=0,
            remove_hydrogen=True,
        )
        data = process_pyg(complex_info)
        if data is not None:
            data_list.append(data)
    logging.info(f"Done! Caching result in {cached_data_list}")
    torch.save(data_list, cached_data_list)
    return data_list


def process_pyg(
    complex: ComplexInformation,
) -> Optional[HeteroData]:
    data = HeteroData()
    try:
        data = set_atoms(complex.ligand, data, NodeType.Ligand)
        data = set_bonds(complex.ligand, data, NodeType.Ligand)

        data = set_atoms(complex.pocket, data, NodeType.Pocket)
        data = set_bonds(complex.pocket, data, NodeType.Pocket)

        data = add_pocket_information(data, complex.pocket_mol2_file)
    except Exception as e:
        logging.warning(f"Exception: {e} when processing {complex}")
        return None

    data.y = torch.tensor(complex.activity_value).view(1)
    data.docking_score = torch.tensor(complex.docking_score).view(1)
    data.posit_prob = torch.tensor(complex.docking_score).view(1)
    data.predicted_rmsd = torch.tensor(complex.predicted_rmsd).view(1)
    data.pocket_sequence = complex.pocket_sequence
    data.activity_type = complex.activity_type
    data.ident = complex.kinodata_ident
    data.smiles = complex.compound_smiles
    return data


def load_model(
    path: Path,
    map_location: Any = None,
):
    logging.info(f"Loading model from {args.model_path}")
    assert path.exists()
    ckpt_file = list(path.rglob("*.ckpt"))[0]
    ckpt = torch.load(ckpt_file, map_location=map_location)
    config = Config(read_config(path / "config.json"))
    config = patch_config(config)
    model = make_model(config)
    model.load_state_dict(ckpt["state_dict"])
    return model, config


parser = ArgumentParser()
parser.add_argument("model_path", type=Path)
parser.add_argument("sdf_path", type=Path)
parser.add_argument("--device", type=str, default="cpu")

# usage example
# predict.py models/scaffold
if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.model_path, map_location=device)
    data_list = load_data(
        args.sdf_path, remove_hydrogen=config.get("remove_hydrogen", True)
    )
    pass
