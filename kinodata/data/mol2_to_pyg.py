import itertools
from pathlib import Path
from typing import Callable, Iterable, TypeVar
from biopandas.mol2 import PandasMol2
import rdkit.Chem as Chem
import torch
from kinodata.data.featurization.rd_features import (
    set_atoms,
    set_bonds,
)
from torch_geometric.data import HeteroData
from torch.utils.data import IterableDataset

from kinodata.types import NodeType, RelationType
from kinodata.data.utils.scaffolds import mol_to_scaffold

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
):
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


class LazyComplexDataset(IterableDataset):
    def __init__(
        self,
        pocket_file_itr: Iterable[str | Path],
        ligand_mols_itr: Iterable[Chem.Mol],
        metadata_itr: Iterable[dict] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        # make iterable that always return an empty dict
        if metadata_itr is None:
            metadata_itr = itertools.repeat(dict())
        self.pocket_file_itr = pocket_file_itr
        self.ligand_mols_itr = ligand_mols_itr
        self.metadata_itr = metadata_itr

    def __iter__(self):
        for pocket_file, ligand_mol, metadata in zip(
            self.pocket_file_itr, self.ligand_mols_itr, self.metadata_itr
        ):
            complex_data = mol2_pocket_to_pyg(pocket_file, self.kwargs)
            set_atoms(ligand_mol, complex_data, NodeType.Ligand)
            set_bonds(ligand_mol, complex_data, NodeType.Ligand)
            complex_data.scaffolds = mol_to_scaffold(ligand_mol)
            for key, value in metadata.items():
                if isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float32)
                complex_data[key] = value
            yield complex_data
