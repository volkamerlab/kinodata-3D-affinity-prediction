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
from kinodata.types import *
from kinodata.util import wandb_interface, ModelInfo
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

RESIDUE_ATOM_INDEX = _DATA / "processed" / "residue_atom_index"
CPU_COUNT = 12
HAS_GPU = torch.cuda.is_available()
FORCE_CPU = False
DEVICE = "cpu" if FORCE_CPU or not HAS_GPU else "cuda:0"


def make_config():
    cfg.register(
        "crocodoc",
        split_type="scaffold-k-fold",
        remove_hydrogen=True,
        filter_rmsd_max_value=2.0,
        split_index=0,
        edges_only=False,
        model_type="CGNN",
        outfile=None,
    )
    config = cfg.get("crocodoc")
    config["model_path"] = None
    config = config.update_from_args("model_path", "outfile")
    return config


model_dir = Path("models")
assert model_dir.exists(), (model_dir, Path().absolute())


def path_to_model(
    rmsd_threshold: int, split_type: str, split_fold: int, model_type: str
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


model_cls = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer,
}


def load_wandb_config(config_file: Path) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
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
    if config["ablate_binding_features"] == 0:
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


def load_single_index(file: Path):
    with open(file, "r") as f:
        try:
            dictionary = json.load(f)
        except json.decoder.JSONDecodeError:
            dictionary = None
    ident = int(file.stem.split("_")[-1])
    return (ident, dictionary)


def get_ident(file: Path):
    ident = int(file.stem.split("_")[-1])
    return ident


def load_residue_atom_index(idents, parallelize=True):
    files = list(RESIDUE_ATOM_INDEX.iterdir())
    files = [file for file in files if get_ident(file) in idents]
    assert len(files) == len(idents)
    progressing_iterable = tqdm(files, desc="Loading residue atom index...")
    if parallelize:
        with mp.Pool(CPU_COUNT) as pool:
            tuples = pool.map(load_single_index, progressing_iterable)
    else:
        tuples = [load_single_index(f) for f in progressing_iterable]
    return dict(tuples)


if __name__ == "__main__":
    predict_reference = True
    config = make_config()
    if "model_path" in config and config["model_path"] is not None:
        config["model_path"] = Path(config["model_path"])
    if config.get("model", None):
        print("Loading model from", config["model_path"])
    if config.get("outfile", None):
        print("Will write output to", config["outfile"])

    model_info = None
    if (model_path := config.get("model_path", None)) is not None:
        model_info = ModelInfo.from_dir(model_path)
    model, model_config = load_model_from_checkpoint(
        rmsd_threshold=2,
        split_type=config["split_type"],
        split_fold=config["split_index"],
        model_type=config["model_type"],
        model_info=model_info,
    )

    print("Creating data list...")
    data_list = prepare_data(model_config)

    print("Checking data...")
    for data in tqdm(data_list):
        edge_index = data[
            NodeType.Complex, RelationType.Covalent, NodeType.Complex
        ].edge_index
        assert data[NodeType.Complex].x.shape[0] > edge_index.max()

    idents = set([int(data["ident"].item()) for data in data_list])

    index = load_residue_atom_index(idents, parallelize=False)
    del_list = []

    for k, v in index.items():
        if v is None:
            print(f"Removing ident {k} due to missing index")
            del_list.append(k)

    for k in del_list:
        del index[k]

    print("Preparing residue masking transform...")
    transform = MaskResidues(index, edges_only=config["edges_only"])

    trainer = Trainer(
        logger=None,
        auto_select_gpus=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    fold = int(model_config["split_index"])
    split_type = model_config["split_type"].split("-")[0]
    model_type = model_config["model_type"]
    model_type_repr = model_type.replace("-", "").lower()
    if predict_reference:
        predictions = trainer.predict(model, DataLoader(data_list, batch_size=32))
        predictions = cat_many(predictions)
        meta = cat_many(
            [
                {
                    "ident": data["ident"],
                    "chembl_activity_id": data["chembl_activity_id"],
                    "klifs_structure_id": data["klifs_structure_id"],
                }
                for data in data_list
            ]
        )
        df = pd.DataFrame(
            {
                "ident": meta["ident"].cpu().numpy(),
                "chembl_activity_id": meta["chembl_activity_id"].cpu().numpy(),
                "klifs_structure_id": meta["klifs_structure_id"].cpu().numpy(),
                "reference_pred": predictions["pred"].cpu().numpy(),
                "target": predictions["target"].cpu().numpy(),
            }
        )
        df.to_csv(
            _DATA
            / "crocodoc_out"
            / "residue_binding_feat_study"
            / f"reference_{split_type}_{fold}_{model_type_repr}.csv",
            index=False,
        )

    part = 0
    while True:
        print(len(transform))
        pre_filter = [data for data in data_list if transform.filter(data)]
        transformed_data_list = [transform(copy.copy(data)) for data in pre_filter]
        transformed_data_list = transformed_data_list
        predictions = trainer.predict(
            model, DataLoader(transformed_data_list, batch_size=32, shuffle=False)
        )
        predictions = cat_many(predictions)
        meta = cat_many(
            [
                {
                    "ident": data["ident"],
                    "chembl_activity_id": data["chembl_activity_id"],
                    "klifs_structure_id": data["klifs_structure_id"],
                    "masked_residue": data.masked_residue,
                }
                for data in transformed_data_list
            ]
        )
        masked_resname = [data.masked_resname for data in transformed_data_list]
        masked_res_letter = [data.masked_res_letter for data in transformed_data_list]
        df = pd.DataFrame(
            {
                "ident": meta["ident"].cpu().numpy(),
                "chembl_activity_id": meta["chembl_activity_id"].cpu().numpy(),
                "klifs_structure_id": meta["klifs_structure_id"].cpu().numpy(),
                "masked_residue": meta["masked_residue"].cpu().numpy(),
                "masked_pred": predictions["pred"].cpu().numpy(),
                "masked_resname": masked_resname,
                "masked_res_letter": masked_res_letter,
            }
        )
        if model_path is not None:
            file_name = f"residue_delta_{str(model_path).replace('/', '_')}"
        else:
            file_name = f"residue_delta_{split_type}_{fold}_{model_type_repr}"
        file_name = config.get("outfile", file_name)
        file_name = f"{file_name}_part_{part}"
        if config["edges_only"]:
            file_name += "_edges_only"
        df.to_csv(
            _DATA / "crocodoc_out" / "residue_binding_feat_study" / f"{file_name}.csv",
            index=False,
        )
        part += 1
        if len(transform) == 0:
            break
