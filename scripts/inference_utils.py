import copy

import wandb

wandb.init(mode="disabled")

import json
import multiprocessing as mp
from pathlib import Path

import kinodata.configuration as cfg
import pandas as pd
import torch
from kinodata.data import Filtered, KinodataDocked
from kinodata.data.data_module import create_dataset
from kinodata.data.dataset import _DATA
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.model import RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.model.regression import cat_many
from kinodata.transform import FilterDockingRMSD, TransformToComplexGraph
from kinodata.transform.mask_residues import MaskResidues
from kinodata.data.featurization.atoms import AtomFeatures
from kinodata.data.featurization.bonds import NUM_BOND_TYPES
from kinodata.types import *
from kinodata.util import wandb_interface, ModelInfo
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

DEFAULT_MODEL_DIR = Path("models")
assert DEFAULT_MODEL_DIR.exists(), (DEFAULT_MODEL_DIR, Path().absolute())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path_to_model(
    rmsd_threshold: int, split_type: str, split_fold: int, model_type: str
) -> Path:
    p = (
        DEFAULT_MODEL_DIR
        / f"rmsd_cutoff_{rmsd_threshold}"
        / split_type
        / str(split_fold)
        / model_type
    )
    if not p.exists():
        p.mkdir(parents=True)
    return p


model_cls = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer,
}


def load_wandb_config(config_file: Path) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
    config = {
        key: None if str(value).lower() == "none" else value
        for key, value in config.items()
    }
    return config


def load_model_from_checkpoint(
    rmsd_threshold: int | None = None,
    split_type: str | None = None,
    split_fold: int | None = None,
    model_type: str | None = None,
    model_path: Path | None = None,
    model_info: ModelInfo | None = None,
) -> tuple[RegressionModel, cfg.Config]:
    if model_info is not None:
        model_ckpt = model_info.fp_model_ckpt
        config = model_info.config
        model_type = model_info.model_type
    else:
        if model_path is None:
            assert rmsd_threshold is not None
            assert split_type is not None
            assert split_fold is not None
            assert model_type is not None
            model_path = path_to_model(
                rmsd_threshold, split_type, split_fold, model_type
            )
        model_ckpt = list(model_path.glob("**/*.ckpt"))[0]
        model_config = model_path / "config.json"
        config = load_wandb_config(model_config)
    if "ablate_binding_features" not in config:
        config["atom_attr_size"] = AtomFeatures.size - 6
        config["edge_size"] = NUM_BOND_TYPES - 1
        config["ablate_binding_features"] = -1
    elif config["ablate_binding_features"] == 0:
        config["atom_attr_size"] = AtomFeatures.size
    elif config["ablate_binding_features"] == 1:
        config["atom_attr_size"] = AtomFeatures.size - 3
    cls = model_cls[model_type]
    config = cfg.Config(config)
    ckp = torch.load(model_ckpt, map_location=DEVICE)
    model = cls(config)
    assert isinstance(model, RegressionModel)
    model.load_state_dict(ckp["state_dict"])
    return model, config


def prepare_data(config, need_cplx_representation=True):
    transform = lambda x: x
    if need_cplx_representation:
        to_cplx = TransformToComplexGraph(remove_heterogeneous_representation=True)
        transform = to_cplx
    if config.get("ablate_binding_features", None) == 1:

        def mask_features(data):
            data["complex"].x = data["complex"].x[:, :-3]
            return data

        transform = Compose([transform, mask_features])
    rmsd_filter = FilterDockingRMSD(config["filter_rmsd_max_value"])
    data_cls = Filtered(KinodataDocked(), rmsd_filter)
    test_split = KinodataKFoldSplit(config["split_type"], 5)
    split = test_split.split(data_cls())[config["split_index"]]
    test_data = create_dataset(
        data_cls,
        dict(),
        split.test_split,
        None,
    )
    # ?
    forbidden_seq = set(
        [
            "KALGKGLFSMVIRITLKVVGLRILNLPHLILEYCKAKDIIRFLQQKNFLLLINWGIR",
            "LIGKGDSARLDYLVVGRLLQLVREP",
            "LIIGKGDFGKVELSALKVVDIIRLILDYLVVGRLLQLVRE",
            "NKMGEGGFGVVYKVAVKKLQFDQEIKVMAKCQENLVELLGFCLVYVYMPNGSLLDRLSCFLHENHHIHRDIKSANILLISDFGLA",
            "_ALNVLDMSQKLYLLSSLDPYLLEMYSYLILEAPEGEIFNLLRQYLHSAMIIYRDLKPHNVLFIAA",
        ]
    )
    data_list = [
        transform(data)
        for data in test_data
        if data.pocket_sequence not in forbidden_seq
    ]

    source_df = test_data.df
    source_df = source_df.set_index("ident")

    def add_metdata(data):
        row = source_df.loc[data["ident"].item()]
        data["chembl_activity_id"] = torch.tensor([int(row["activities.activity_id"])])
        data["klifs_structure_id"] = torch.tensor(
            [int(row["similar.klifs_structure_id"])]
        )
        return data

    data_list = [add_metdata(data) for data in data_list]
    return data_list
