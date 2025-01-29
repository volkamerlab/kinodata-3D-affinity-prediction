import json
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Literal, Optional, Set

import pandas as pd
import torch
from torch import full
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph
from tqdm import tqdm

from kinodata.data.io.read_klifs_mol2 import read_klifs_mol2
from kinodata.data.dataset import _DATA
from kinodata.data.utils.remap_residue_index import remap_residue_index

from ..types import MASK_RESIDUE_KEY, NodeType, RelationType

protein_letters_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

protein_letters_1to3 = {v: k for k, v in protein_letters_3to1.items()}


def mask_residue(
    data: HeteroData,
    residue_idx: int,
    residue_to_atom: Dict[int, List[int]],
    mask_bonds_only: bool = False,
    mask_pl_edges: bool = True,
    mask_pocket_atoms: bool = False,
) -> Optional[HeteroData]:
    if mask_pl_edges and mask_pocket_atoms:
        raise ValueError(
            "Only one of 'mask_pl_edges' and 'mask_pocket_atoms' can be True!"
        )

    res_letter = data.pocket_sequence[int(residue_idx) - 1]
    res_name = (
        protein_letters_1to3[res_letter]
        if res_letter in protein_letters_1to3
        else "???"
    )
    residue_atoms = residue_to_atom[residue_idx]

    node_store = data[NodeType.Complex]
    edge_store = data[NodeType.Complex, RelationType.Covalent, NodeType.Complex]
    x, z, pos = node_store.x, node_store.z, node_store.pos
    edge_index, edge_attr = edge_store.edge_index, edge_store.edge_attr

    num_nodes = x.size(0)

    if mask_bonds_only:
        residue_atoms = torch.tensor(residue_atoms)
        row, col = edge_index
        is_residue_pl_edge = torch.logical_or(
            torch.isin(row, residue_atoms), torch.isin(col, residue_atoms)
        )
        edge_index = torch.stack((row[~is_residue_pl_edge], col[~is_residue_pl_edge]))
        edge_attr = edge_attr[~is_residue_pl_edge]
    elif mask_pocket_atoms:
        mask = full((num_nodes,), 1, dtype=torch.bool)
        mask[residue_atoms] = False
        x = x[mask]
        z = z[mask]
        pos = pos[mask]
        edge_index, edge_attr = subgraph(
            mask, edge_index, edge_attr, relabel_nodes=True
        )
    elif mask_pl_edges:
        is_part_of_masked_residue = full((num_nodes,), 0, dtype=torch.bool)
        is_part_of_masked_residue[residue_atoms] = True
        data[NodeType.Complex][MASK_RESIDUE_KEY] = is_part_of_masked_residue

    data.masked_residue = torch.tensor([int(residue_idx)])
    data.masked_resname = res_name
    data.masked_res_letter = res_letter
    data[NodeType.Complex].x = x
    data[NodeType.Complex].z = z
    data[NodeType.Complex].pos = pos
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_index = (
        edge_index
    )
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_attr = (
        edge_attr
    )
    return data


def _load_single_index(file: Path):
    with open(file, "r") as f:
        try:
            dictionary = json.load(f)
        except json.decoder.JSONDecodeError:
            dictionary = None
    ident = int(file.stem.split("_")[-1])
    return (ident, dictionary)


def _get_ident(file: Path):
    ident = int(file.stem.split("_")[-1])
    return ident


def _load_residue_atom_index(idents, residue_index_dir: Path, parallelize=True):
    files = list(residue_index_dir.iterdir())
    files = [file for file in files if _get_ident(file) in idents]
    assert len(files) == len(idents), "Some residue index files are missing!"
    progressing_iterable = tqdm(files, desc="Loading residue atom index...")
    if parallelize:
        with mp.Pool() as pool:
            tuples = pool.map(_load_single_index, progressing_iterable)
    else:
        tuples = [_load_single_index(f) for f in progressing_iterable]
    return dict(tuples)


class MaskResidues(BaseTransform):
    open_list: Dict[int, Set[int]]
    residue_to_atom: Dict[int, Dict[int, List[int]]]
    RESIDUE_INDEX_DIR = _DATA / "processed" / "residue_atom_index"
    """
    Exhaustively masks residues one residue at a time.
    The order of masking is arbitrary.
    """

    @classmethod
    def precompute_residue_index(
        cls,
        kinodata_3d_file: Path = _DATA / "processed" / "kinodata3d.csv",
    ):
        if not kinodata_3d_file.exists():
            raise FileNotFoundError(
                f"Base kinodata3d file {str(kinodata_3d_file)} not found!"
            )
        print("Reading base data...")
        df = pd.read_csv(kinodata_3d_file)

        pbar = tqdm(df.iterrows(), total=len(df))
        for _, row in pbar:
            ident = int(row["ident"])
            target_fname = cls.RESIDUE_INDEX_DIR / f"ident_{ident}.json"
            if target_fname.exists():
                continue
            pocket_file = Path(row["pocket_mol2_file"])
            pocket_df = read_klifs_mol2(pocket_file, with_bonds=False)
            pocket_df = pocket_df[
                pocket_df["atom.type"].str.upper() != "H"
            ].reset_index()
            residue_idx = pocket_df["residue.subst_id"].values
            residue_idx = remap_residue_index(
                row["structure.pocket_sequence"], residue_idx
            )
            lookup_table = defaultdict(list)
            atom_idx = pocket_df.index.values
            for r, a in zip(residue_idx, atom_idx):
                lookup_table[int(r)].append(int(a))
            with open(target_fname, "w") as f:
                json.dump(lookup_table, f)
            del lookup_table
            del target_fname
            del pocket_df
            del residue_idx
            del atom_idx

    @classmethod
    def load_residue_index(cls, idents: list[int]) -> Dict[int, List[int]]:
        return _load_residue_atom_index(
            idents,
            residue_index_dir=cls.RESIDUE_INDEX_DIR,
            parallelize=False,
        )

    def __init__(
        self,
        residue_to_atom: Dict[int, Dict[int, List[int]]],
        mask_type: Literal["atom_objects", "pl_interactions", "bonds"] = "atom_objects",
    ) -> None:
        super().__init__()
        assert mask_type in ["atom_objects", "pl_interactions", "bonds"]
        self.mask_type = mask_type
        self.residue_to_atom = residue_to_atom
        self.open_list = {
            ident: set(index.keys()) for ident, index in residue_to_atom.items()
        }

    def __len__(self) -> int:
        return sum(len(val) for val in self.open_list.values())

    def filter(self, data: HeteroData) -> bool:
        ident = int(data["ident"].item())
        if ident not in self.open_list:
            return False
        return len(self.open_list[ident]) > 0

    def __call__(self, data: HeteroData) -> HeteroData:
        ident = int(data["ident"].item())
        open_residues = self.open_list[ident]
        if len(open_residues) == 0:
            data["masked_residue"] = torch.tensor([-1])
            return data
        residue_idx = open_residues.pop()
        return mask_residue(
            data,
            residue_idx,
            self.residue_to_atom[ident],
            mask_bonds_only=self.mask_type == "bonds",
            mask_pocket_atoms=self.mask_type == "atom_objects",
            mask_pl_edges=self.mask_type == "pl_interactions",
        )
