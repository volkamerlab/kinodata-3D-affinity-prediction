import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch_geometric.data import HeteroData

from kinodata.data.io import read_klifs_mol2
from .residue import sitealign_feature_lookup, amino_acid_to_int
from rdkit.Chem import GetPeriodicTable

pt = GetPeriodicTable()


def add_pocket_information(
    data: HeteroData,
    pocket_mol2_file: Union[str, Path],
    residue_feature_subset: Optional[List[str]] = [
        "hbd",
        "hba",
        "aromatic",
        "aliphatic",
        "charge_pos",
        "charge_neg",
    ],
) -> Optional[HeteroData]:
    """TODO
    This is kind of a mess and needs to be refactored.

    Parameters
    ----------
    data : HeteroData
        heterogeneous pyg data object
    pocket_mol2_file : Union[str, Path]
        path to KLIFS pocket mol2 file
    residue_feature_subset : Optional[List[str]], optional
        which kissim features to use for featurizing residues,
        by default [ "hbd", "hba", "aromatic", "aliphatic", "charge_pos", "charge_neg", ]

    Returns
    -------
    Optional[HeteroData]
        The modified data object with pocket information
        or None if a known exception is caught.
    """
    if residue_feature_subset is None:
        residue_feature_subset = sitealign_feature_lookup.columns

    # prepare pocket data frame
    # each row corresponds to one atom
    df = read_klifs_mol2(pocket_mol2_file)
    # TODO: are all residue names exactly 3 characters long??
    df["residue.type"] = df["residue.subst_name"].apply(lambda s: s[:3])
    df["atom.id"] = df["atom.id"] - df["atom.id"] - df["atom.id"].min()
    df["residue.subst_id"] = df["residue.subst_id"] - df["residue.subst_id"].min()

    # select unique residues and featurize them
    # by looking up sitealign features based on residue type
    # lookup table taken from KiSSim
    df_residue = (
        df[["residue.subst_id", "residue.type"]]
        .drop_duplicates()
        .set_index("residue.subst_id")
    )
    try:
        residues_features = (
            sitealign_feature_lookup[residue_feature_subset]
            .loc[df_residue["residue.type"]]
            .set_index(df_residue.index)
        )
    except KeyError as e:
        logging.warning(f"Unable to map residue to sitealign features: {e}")
        return None
    # add categorical integer feature for residue type
    df_residue["residue.type_idx"] = df_residue["residue.type"].apply(
        amino_acid_to_int.get
    )

    # compute 3D positions of residues as
    # geometric center of member atoms.
    residue_centers = df.groupby("residue.subst_id")[
        ["atom.x", "atom.y", "atom.z"]
    ].mean()

    # set categorical integer feature for atom type / element
    # based on atomic numbers
    df["atom.atomic_number"] = df["atom.type"].apply(
        lambda t: pt.GetAtomicNumber(t.split(".")[0])
    )

    # convert all features & positions to tensors and store them in the provided
    # pyg data object
    data["pocket_residue"].z = torch.tensor(
        df_residue["residue.type_idx"].values, dtype=torch.long
    )
    data["pocket_residue"].x = torch.tensor(residues_features.values, dtype=torch.float)
    data["pocket_residue"].pos = torch.tensor(
        residue_centers[["atom.x", "atom.y", "atom.z"]].values,
        dtype=torch.float,
    )
    data["pocket"].z = torch.tensor(df["atom.atomic_number"].values, dtype=torch.long)
    data["pocket"].x = torch.tensor(df["atom.charge"].values, dtype=torch.float).view(
        -1, 1
    )
    data["pocket"].pos = torch.tensor(
        df[["atom.x", "atom.y", "atom.z"]].values,
        dtype=torch.float,
    )

    # store atom/residue member relation as edges
    # (a, r) if atom a belongs to residue r
    data["pocket_residue", "contains", "pocket"].edge_index = torch.tensor(
        df[["residue.subst_id", "atom.id"]].values.T, dtype=torch.long
    )

    return data
