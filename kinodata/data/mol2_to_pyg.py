import itertools
import logging
from pathlib import Path
from typing import Iterable, TypeVar

import rdkit.Chem as Chem
import torch
from biopandas.mol2 import PandasMol2
from torch.utils.data import IterableDataset
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm

from kinodata.data.featurization.rd_features import (
    set_atoms,
    set_bonds,
)
from kinodata.data.io.read_sdf import read_sdf_molecules
from kinodata.data.utils.scaffolds import mol_to_scaffold
from kinodata.types import NodeType, RelationType

logger = logging.getLogger(__name__)

T = TypeVar("T")
Q = TypeVar("Q")

KLIFS_MOL2_COLUMNS = {
    0: ("atom_id", "int32"),
    1: ("atom_name", "string"),
    2: ("x", "float32"),
    3: ("y", "float32"),
    4: ("z", "float32"),
    5: ("atom_type", "string"),
    6: ("subst_id", "int32"),
    7: ("subst_name", "string"),
    8: ("charge", "float32"),
    9: ("atom.status_bit", "string"),
}


def _ensure_views_aligned(
    rdkit_mol,
    bp_data_frame,
):
    if rdkit_mol.GetNumAtoms() != len(bp_data_frame):
        raise ValueError(
            f"Number of atoms in RDKit molecule ({rdkit_mol.GetNumAtoms()}) "
            f"does not match number of atoms in Biopandas DataFrame ({len(bp_data_frame)})"
        )
    for atom, row in zip(
        rdkit_mol.GetAtoms(),
        bp_data_frame.itertuples(),
    ):
        rdkit_atom_name = atom.GetProp("_TriposAtomName")
        bp_atom_name = row.atom_name
        if rdkit_atom_name != bp_atom_name:
            raise ValueError(
                f"Atom names do not match: RDKit atom name '{rdkit_atom_name}' "
                f"does not match Biopandas atom name '{bp_atom_name}'"
            )
        rdkit_atom_type = atom.GetProp("_TriposAtomType")
        bp_atom_type = row.atom_type
        if rdkit_atom_type != bp_atom_type:
            raise ValueError(
                f"Atom types do not match: RDKit atom type '{rdkit_atom_type}' "
                f"does not match Biopandas atom type '{bp_atom_type}'"
            )
    return rdkit_mol, bp_data_frame


def _remove_hydrogens(rdkit_mol, bp_data_frame):
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    bp_data_frame = bp_data_frame[
        ~bp_data_frame.atom_type.str.startswith("H")
    ].reset_index()
    bp_data_frame.atom_id = bp_data_frame.index + 1
    rdkit_mol, bp_data_frame = _ensure_views_aligned(rdkit_mol, bp_data_frame)
    return rdkit_mol, bp_data_frame


def mol2_pocket_to_pyg(
    path: str | Path,
    columns: list[str] | None = KLIFS_MOL2_COLUMNS,
    remove_hydrogens: bool = False,
    sanity_check_alignment: bool = True,
) -> HeteroData:
    rdkit_mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    mol2_reader = PandasMol2()
    mol2_reader.read_mol2(path, columns)
    atom_df = mol2_reader.df
    if remove_hydrogens:
        rdkit_mol, atom_df = _remove_hydrogens(rdkit_mol, atom_df)
    if sanity_check_alignment:
        rdkit_mol, atom_df = _ensure_views_aligned(rdkit_mol, atom_df)

    data = HeteroData()
    set_atoms(rdkit_mol, data, NodeType.Pocket)
    set_bonds(rdkit_mol, data, NodeType.Pocket)
    data[NodeType.Pocket].atom_status_bit = list(atom_df["atom.status_bit"].values)

    data[
        NodeType.Pocket, RelationType.IsPartOf, NodeType.PocketResidue
    ].edge_index = torch.tensor(atom_df[["atom_id", "subst_id"]].values.T - 1)
    data[NodeType.PocketResidue].residue_types = list(
        atom_df[["subst_id", "subst_name"]].drop_duplicates().subst_name.str[:3].values
    )
    data[NodeType.PocketResidue].num_nodes = len(
        data[NodeType.PocketResidue].residue_types
    )
    return data


def _add_value(data, key, value, overwrite: bool = False) -> HeteroData:
    if not overwrite and key in data:
        raise KeyError(
            f"Key '{key}' already exists in data. Use `overwrite=True` to enable replacement."
        )
    if isinstance(value, float):
        value = torch.tensor([value], dtype=torch.float32)
    if isinstance(value, int):
        value = torch.tensor([value], dtype=torch.long)
    if isinstance(value, bool):
        value = torch.tensor([value], dtype=torch.bool)
    data[key] = value
    return data


class LazyComplexDataset(IterableDataset):
    def __init__(
        self,
        mol2_files: Iterable[str | Path],
        ligand_rdkit_molecules: Iterable[Chem.Mol],
        metadata: Iterable[dict] | None = None,
        required_ligand_properties: dict[str, str | float | int | bool] | None = None,
        mol2_pocket_to_pyg_options: dict | None = None,
    ):
        """
        A dataset that lazily loads protein pocket data from mol2 files and merges them
        with corresponding ligand molecules into torch_geometric HeteroData objects representing
        protein-ligand complexes.

        Args:
            mol2_files (Iterable[str  |  Path]): Paths to the mol2 files.
            ligand_rdkit_molecules (Iterable[Chem.Mol]): RDKit molecule objects for the ligands.
            metadata (Iterable[dict] | None, optional): Metadata for each complex. Defaults to None.
        """
        super().__init__()
        self._required_ligand_properties = required_ligand_properties or {}
        self._mol2_pocket_to_pyg_options = mol2_pocket_to_pyg_options or {}
        # make iterable that always return an empty dict
        if metadata is None:
            metadata = itertools.repeat(dict())
        self._mol2_files = mol2_files
        self._ligand_mols = ligand_rdkit_molecules
        self._metadata = metadata

    def __iter__(self):
        for pocket_file, ligand_mol, metadata in zip(
            self._mol2_files, self._ligand_mols, self._metadata
        ):
            complex_data = mol2_pocket_to_pyg(
                pocket_file, **self._mol2_pocket_to_pyg_options
            )
            set_atoms(ligand_mol, complex_data, NodeType.Ligand)
            set_bonds(ligand_mol, complex_data, NodeType.Ligand)
            complex_data.scaffolds = mol_to_scaffold(ligand_mol)
            for key, default in self._required_ligand_properties.items():
                try:
                    value = ligand_mol.GetProp(key)
                except KeyError:
                    logger.warning(
                        f"Key '{key}' not found in ligand molecule properties."
                        f"Using default value: {default}"
                    )
                    value = default
                complex_data = _add_value(complex_data, key, value)
            for key, value in metadata.items():
                complex_data = _add_value(complex_data, key, value)
            yield complex_data


def _check_homogenized_file_structure(
    root: str | Path,
    files: list[str | Path],
):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(
            f"Root directory '{root}' does not exist or is not a directory."
        )
    for file in map(Path, files):
        if file.parent != root:
            raise ValueError(
                f"File '{file}' is not in the root directory '{root}'. "
                "All files must be in the same directory."
            )


class Mol2SDFComplexDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        mol2_files: Iterable[str | Path],
        sdf_files: Iterable[str | Path] | None = None,
        multi_sdf_file: str | Path | None = None,
        processed_file_name: str | None = None,
        required_ligand_properties: dict[str, str | float | int | bool] | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Create a dataset that combines mol2 files for protein pockets with
        SDF files for ligands. The dataset will process the mol2 files and rdkit ligands into
        torch_geometric HeteroData objects representing protein-ligand complexes.

        Args:
            root (str | Path): a directory where the dataset will be stored.
            mol2_files (Iterable[str  |  Path]): mol2 files containing protein pocket data.
            sdf_files (Iterable[str  |  Path] | None, optional): SDF files containing ligand data. Defaults to None.
            multi_sdf_file (str | Path | None, optional): A single multi-SDF file containing ligand data. Defaults to None.
            processed_file_name (str | None, optional): The name of the processed file. Defaults to None.
            required_ligand_properties (dict[str, str  |  float  |  int  |  bool] | None, optional): Properties required for the ligands.
            dictionary keys are property names to Get from rdkit molecules, keys are default values. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Raises:
            ValueError: _description_
        """
        if multi_sdf_file is not None:
            self._sdf_files = [multi_sdf_file]
        elif sdf_files is not None:
            self._sdf_files = list(sdf_files)
        if sdf_files is None and multi_sdf_file is None:
            raise ValueError("Either `sdf_files` or `multi_sdf_file` must be provided.")
        self._mol2_files = list(mol2_files)
        _check_homogenized_file_structure(root / "raw", self._mol2_files)
        _check_homogenized_file_structure(root / "raw", self._sdf_files)
        self._processed_file_name = processed_file_name or "data.pt"
        self._required_ligand_properties = required_ligand_properties or {}
        self._verbose = verbose
        super().__init__(root, **kwargs)
        self.load(self.processed_paths[0])

    def raw_file_names(self) -> list[str]:
        return self._mol2_files + self._sdf_files

    def processed_file_names(self) -> list[str]:
        return [self._processed_file_name]

    def process(self):
        mol2_files = tqdm(self._mol2_files) if self._verbose else self._mol2_files
        lazy_dataset = LazyComplexDataset(
            mol2_files=mol2_files,
            ligand_rdkit_molecules=read_sdf_molecules(self._sdf_files),
            mol2_pocket_to_pyg_options=self._mol2_pocket_to_pyg_options,
            required_ligand_properties=self._required_ligand_properties,
        )
        data_list = list(lazy_dataset)
        self.save(
            data_list,
            self.processed_paths[0],
        )
