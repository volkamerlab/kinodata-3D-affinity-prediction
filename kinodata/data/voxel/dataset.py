from pathlib import Path

import torch
from ..dataset import ComplexInformation
from docktgrid import VoxelDataset as DocktgridVoxelDataset
from docktgrid import MolecularParser, MolecularData
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .klifs_parser import KlifsSymbolParser, KlifsPocketParser

from docktgrid.voxel_dataset import VoxelDataset
from docktgrid import VoxelGrid
from docktgrid.view import BasicView

_default_voxel = VoxelGrid([BasicView()], 1, (24, 24, 24))


class MolecularDataRegister:

    def __init__(self, parser):
        self.parser = parser
        self._molecular_data = {}

    def __getitem__(self, file_path: Path | str):
        ext = file_path.suffix
        file_path = str(file_path)
        if file_path in self._molecular_data:
            return self._molecular_data[file_path]
        mol_data = self.parser.parse_file(file_path, ext)
        self._molecular_data[file_path] = mol_data
        return mol_data


ligand_register = MolecularDataRegister(KlifsSymbolParser())
pocket_register = MolecularDataRegister(KlifsPocketParser())


def _prohibited_compelx(lig: MolecularData, ptn: MolecularData) -> bool:
    symbols = set.union(set(list(lig.element_symbols)), set(list(ptn.element_symbols)))
    if "A" in symbols:
        return True
    return False


class VoxelDataset(DocktgridVoxelDataset):

    def __init__(
        self,
        protein_files,
        ligand_files,
        labels,
        voxel,
        molparser=...,
        transform=None,
        root_dir="",
        metadata=None,
    ):
        super().__init__(
            protein_files, ligand_files, labels, voxel, molparser, transform, root_dir
        )
        assert len(metadata) == len(labels)
        self.metadata = torch.tensor(metadata)

        mask = []
        masked_lig_files = []
        masked_ptn_files = []
        for lig, ptn in zip(ligand_files, protein_files):
            assert isinstance(lig, MolecularData)
            assert isinstance(ptn, MolecularData)
            if _prohibited_compelx(lig, ptn):
                mask.append(True)
                continue
            masked_lig_files.append(lig)
            masked_ptn_files.append(ptn)
            mask.append(False)
        self.lig_files = masked_lig_files
        self.ptn_files = masked_ptn_files
        self.labels = self.labels[mask]
        self.metadata = self.metadata[mask]

    def __getitem__(self, idx):
        voxs, label = super().__getitem__(idx)
        return (
            voxs,
            label,
            torch.tensor(idx, dtype=torch.long),
            self.metadata[idx],
        )


def _ensure_opt_int_array(arr) -> np.ndarray | None:
    if arr is None:
        return arr
    return np.array(arr, dtype=int)


def make_dataset(
    metadata: pd.DataFrame,
    voxel: VoxelGrid = _default_voxel,
) -> VoxelDataset:
    print(f"Creating dataset with {len(metadata)} samples...")
    activity_ids = []
    pocket_data = []
    ligand_data = []
    labels = []
    for activity_id, row in metadata.iterrows():
        activity_ids.append(activity_id)
        pocket_data.append(pocket_register[row["pocket_file"]])
        ligand_data.append(ligand_register[row["ligand_file"]])
        labels.append(row["activities.standard_value"])
    return VoxelDataset(
        pocket_data,
        ligand_data,
        labels,
        voxel,
        MolecularParser(),
        metadata=activity_ids,
    )


def make_voxel_dataset_split(
    data_dir: Path,
    train_split: np.ndarray,
    val_split: np.ndarray | None,
    test_split: np.ndarray | None,
    voxel: VoxelGrid = _default_voxel,
    kinodata3d_df: pd.DataFrame = None,
):
    train_split = _ensure_opt_int_array(train_split)
    val_split = _ensure_opt_int_array(val_split)
    test_split = _ensure_opt_int_array(test_split)

    docktgrid_dir = data_dir / "docktgrid"
    pocket_dir = data_dir / "raw" / "mol2" / "pocket"

    if kinodata3d_df is None:
        kinodata3d_df = pd.read_csv(docktgrid_dir / "kinodata3d.csv")
    activity_id = kinodata3d_df["activities.activity_id"].astype(int)
    kinodata3d_df["activities.activity_id"] = activity_id
    if "pocket_file" not in kinodata3d_df.columns:
        kinodata3d_df["pocket_file"] = kinodata3d_df[
            "similar.klifs_structure_id"
        ].apply(lambda kid: pocket_dir / f"{kid}_pocket.mol2")
    kinodata3d_df["ligand_file"] = kinodata3d_df["activities.activity_id"].apply(
        lambda aid: docktgrid_dir / f"ligand_{aid}.mol2"
    )
    kinodata3d_df.set_index("activities.activity_id", inplace=True)

    train_dataset = make_dataset(kinodata3d_df.loc[train_split], voxel)
    val_dataset = (
        make_dataset(kinodata3d_df.loc[val_split], voxel)
        if val_split is not None
        else None
    )
    test_datset = (
        make_dataset(kinodata3d_df.loc[test_split], voxel)
        if test_split is not None
        else None
    )

    if val_dataset is None and test_datset is None:
        return train_dataset
    if val_dataset is not None and test_datset is None:
        return train_dataset, val_dataset
    if val_dataset is None and test_datset is not None:
        return train_dataset, test_datset
    if val_dataset is not None and test_datset is not None:
        return train_dataset, val_dataset, test_datset
