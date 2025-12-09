from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Optional
from argparse import ArgumentParser
import sys
import logging
import hashlib


import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from rdkit.Chem import PandasTools  # type: ignore
from tqdm import tqdm
import pandas as pd

sys.path.append(".")
sys.path.append("..")

from kinodata.data.featurization.biopandas import add_pocket_information
from kinodata.data.featurization.rdkit import set_atoms, set_bonds
from kinodata.types import NodeType
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.data.dataset import (
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
    if (cached_data_list := PROCESSED / f"{md5}.pt").exists():
        logging.info(f"Using cached data list for md5 {md5}")
        return torch.load(cached_data_list)
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
) -> tuple[ComplexTransformer, Config]:
    logging.info(f"Loading model from {args.model_path}")
    assert path.exists()
    ckpt_file = list(path.rglob("*.ckpt"))[0]
    ckpt = torch.load(ckpt_file, map_location=map_location)
    config = Config(read_config(path / "config.json"))
    config = patch_config(config)
    model = make_model(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config


parser = ArgumentParser()
parser.add_argument("model_path", type=Path)
parser.add_argument("sdf_path", type=Path)
parser.add_argument("output_path", type=Path)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--batch_size", type=int, default=32)

# usage example
# predict.py models/scaffold-k-fold/0/CGNN-3D data/raw/greg/combined_with_klifs.sdf data/processed/output.csv
if __name__ == "__main__":
    args = parser.parse_args()
    assert Path(args.output_path).suffix == ".csv", "Can only write to csv file"
    device = torch.device(args.device)
    model, config = load_model(args.model_path, map_location=device)
    data_list = load_data(
        args.sdf_path, remove_hydrogen=config.get("remove_hydrogen", True)
    )
    transform = TransformToComplexGraph(True)
    logging.info("Applying ToComplex transform to data list")
    data_list = [transform(data) for data in data_list]
    predict_output = defaultdict(list)
    pbar = tqdm(total=len(data_list), desc="Obtaining model predictions")
    for batch in DataLoader(data_list, batch_size=args.batch_size):
        batch = batch.to(device)
        with torch.inference_mode():
            out: torch.Tensor = model(batch)
        predict_output["prediction"].extend(out.flatten().tolist())
        predict_output["activities.activity_id"].extend(batch.ident)
        pbar.update(batch.num_graphs)

    df_predictions = pd.DataFrame(predict_output)
    df_predictions["model"] = args.model_path.parts[-1]
    df_predictions["split_type"] = config.get("split_type", "unknown")
    df_predictions["split_fold"] = config.get("split_index", -1)
    df_predictions["rmsd_cutoff"] = config.get("filter_rmsd_max_value", float("nan"))
    df_predictions.to_csv(
        args.output_path,
        index=False,
        mode="a" if Path(args.output_path.exists()) else "w",
    )
