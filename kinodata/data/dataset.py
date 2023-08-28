import multiprocessing as mp
import os
import os.path as osp
import re
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Type

import pandas as pd
import requests  # type : ignore
import torch
from rdkit.Chem import AddHs, Kekulize, MolFromMol2File, PandasTools
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm
from rdkit import RDLogger

from kinodata.data.featurization.biopandas import add_pocket_information
from kinodata.data.featurization.rdkit import (
    append_atoms_and_bonds,
    set_atoms,
    set_bonds,
)
from kinodata.data.featurization.residue import (
    PHYSICOCHEMICAL,
    STRUCTURAL,
    add_kissim_fp,
    add_onehot_residues,
    load_kissim,
)
from kinodata.data.utils.pocket_sequence_klifs import add_pocket_sequence
from kinodata.types import NodeType
from kinodata.data.utils.scaffolds import mol_to_scaffold
from kinodata.transform.to_complex_graph import TransformToComplexGraph
from kinodata.transform.filter_activity import FilterActivityType, ActivityTypes

_DATA = Path(__file__).parents[2] / "data"


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", obj.__repr__())


class KinodataDocked(InMemoryDataset):
    def __init__(
        self,
        root: Optional[str] = str(_DATA),
        prefix: Optional[str] = None,
        remove_hydrogen: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = FilterActivityType([ActivityTypes.pic50]),
        post_filter: Optional[Callable] = None,
        residue_representation: Literal["sequence", "structural", None] = "sequence",
        require_kissim_residues: bool = True,
        use_multiprocessing: bool = True,
        num_processes: Optional[int] = None,
    ):
        self.remove_hydrogen = remove_hydrogen
        self._prefix = prefix
        self.residue_representation = residue_representation
        self.require_kissim_residues = require_kissim_residues
        self.use_multiprocessing = use_multiprocessing
        if self.use_multiprocessing:
            num_processes = num_processes if num_processes else os.cpu_count()
        self.num_processes = num_processes
        self.post_filter = post_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def pocket_sequence_file(self) -> Path:
        return Path(self.raw_dir) / "pocket_sequences.csv"

    @property
    def raw_file_names(self) -> List[str]:
        return ["kinodata_docked_with_rmsd.sdf.gz"]

    @property
    def processed_file_names(self) -> List[str]:
        # TODO add preprocessed kssim fingerprints?
        return [
            f"kinodata_docked.pt",
        ]

    @property
    def processed_dir(self) -> str:
        pdir = super().processed_dir
        if self.prefix is not None:
            pdir = osp.join(pdir, self.prefix)
        return pdir

    @property
    def prefix(self) -> Optional[str]:
        return self._prefix

    @property
    def pocket_dir(self) -> Path:
        return Path(self.raw_dir) / "mol2" / "pocket"

    def download(self):
        # TODO at some point set up public download?
        pass

    @cached_property
    def df(self) -> pd.DataFrame:
        print("Reading data frame..")
        df = PandasTools.LoadSDF(
            self.raw_paths[0],
            smilesName="compound_structures.canonical_smiles",
            molColName="molecule",
            embedProps=True,
            removeHs=self.remove_hydrogen,
        )
        df["activities.standard_value"] = df["activities.standard_value"].astype(float)
        df["docking.predicted_rmsd"] = df["docking.predicted_rmsd"].astype(float)

        print(f"Deduping data frame (current size: {df.shape[0]})...")
        group_key = [
            "compound_structures.canonical_smiles",
            "UniprotID",
            "activities.standard_type",
        ]
        mean_activity = (
            df.groupby(group_key)
            .agg({"activities.standard_value": "mean"})
            .reset_index()
        )
        best_structure = (
            df.sort_values(by="docking.predicted_rmsd", ascending=True)
            .groupby(group_key)[group_key + ["docking.predicted_rmsd", "molecule"]]
            .head(1)
        )
        deduped = pd.merge(mean_activity, best_structure, how="outer", on=group_key)
        df = pd.merge(
            df.drop_duplicates(group_key),
            deduped,
            how="left",
            on=group_key,
            suffixes=(".orig", None),
        )
        for col in ("activities.standard_value", "docking.predicted_rmsd", "molecule"):
            del df[f"{col}.orig"]
        df.set_index("ID", inplace=True)
        print(f"{df.shape[0]} complexes remain after deduplication.")

        print("Checking for missing pocket mol2 files...")
        df["similar.klifs_structure_id"] = (
            df["similar.klifs_structure_id"].astype(float).astype(int)
        )
        # get pocket mol2 files
        if not self.pocket_dir.exists():
            self.pocket_dir.mkdir(parents=True)

        struc_ids = df["similar.klifs_structure_id"].unique()
        pbar = tqdm(struc_ids, total=len(struc_ids))
        for structure_id in pbar:
            fp = self.pocket_dir / f"{structure_id}_pocket.mol2"
            if fp.exists():
                continue
            resp = requests.get(
                "https://klifs.net/api/structure_get_pocket",
                params={"structure_ID": structure_id},
            )
            resp.raise_for_status()
            fp.write_bytes(resp.content)

        pocket_mol2_files = {
            int(fp.stem.split("_")[0]): fp for fp in (self.pocket_dir).iterdir()
        }
        df["pocket_mol2_file"] = [
            pocket_mol2_files[row["similar.klifs_structure_id"]]
            for _, row in df.iterrows()
        ]

        # backwards compatability
        df["ident"] = df.index

        print("Adding pocket sequences", end=" ")
        if self.pocket_sequence_file.exists():
            print(f"from cached file {self.pocket_sequence_file}.")
            pocket_sequences = pd.read_csv(self.pocket_sequence_file)
            pocket_sequences["ident"] = pocket_sequences["ident"].astype(str)
            df = pd.merge(df, pocket_sequences, left_on="ident", right_on="ident")
        else:
            print("from KLIFS.")
            df = add_pocket_sequence(
                df, pocket_sequence_key="structure.pocket_sequence"
            )
            df[["ident", "structure.pocket_sequence"]].to_csv(
                self.pocket_sequence_file, index=False
            )

        return df

    def _process(self):
        f = osp.join(self.processed_dir, "post_filter.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.post_filter):
            warnings.warn(
                f"The `post_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )
        super()._process()
        path = osp.join(self.processed_dir, "post_filter.pt")
        torch.save(_repr(self.post_filter), path)

    def make_data_list(self) -> List[HeteroData]:
        RDLogger.DisableLog("rdApp.*")
        data_list = []
        tasks = [
            (
                row["ident"],
                row["compound_structures.canonical_smiles"],
                row["molecule"],
                float(row["activities.standard_value"]),
                row["activities.standard_type"],
                row["pocket_mol2_file"],
                float(row["docking.chemgauss_score"]),
                float(row["docking.posit_probability"]),
                int(row["similar.klifs_structure_id"]),
                row["structure.pocket_sequence"],
                float(row["docking.predicted_rmsd"]),
                self.remove_hydrogen,
                self.residue_representation,
                self.require_kissim_residues,
            )
            for _, row in tqdm(
                self.df.iterrows(),
                desc="Creating PyG object tasks..",
                total=len(self.df),
            )
        ]

        if self.use_multiprocessing:
            with mp.Pool(os.cpu_count()) as pool:
                data_list = pool.map(process_idx, tqdm(tasks))
        else:
            data_list = list(map(process_idx, tqdm(tasks)))

        data_list = [d for d in data_list if d is not None]
        return data_list

    def filter_transform(self, data_list: List[HeteroData]) -> List[HeteroData]:
        if self.pre_filter is not None:
            print("Applying pre filter..")
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            print("Applying pre transform..")
            data_list = [self.pre_transform(data) for data in data_list]
        if self.post_filter is not None:
            print("Applying post filter..")
            data_list = [data for data in data_list if self.post_filter(data)]
        return data_list

    def persist(self, data_list: List[HeteroData]):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process(self):
        data_list = self.make_data_list()
        data_list = self.filter_transform(data_list)
        self.persist(data_list)

    def ident_index_map(self) -> Dict[Any, int]:
        # this may be very slow if self.transform is computationally expensive
        mapping = [(int(data.ident), index) for index, data in enumerate(self)]
        return dict(mapping)


def _repr_filter(filter: Callable):
    return (
        repr(filter)
        .lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("(", "_")
        .replace(")", "")
    )


def Filtered(dataset: KinodataDocked, filter: Callable) -> Type[KinodataDocked]:
    """
    Creates a new, filtered dataset class with its own processed directory
    that caches the filtered data.

    Parameters
    ----------
    dataset : KinodataDocked
        original dataset
    filter : Callable
        filter, True means the item is not filtered.

    Returns
    -------
    Type[KinodataDocked]
    """

    class FilteredDataset(KinodataDocked):
        def __init__(self, **kwargs):
            for attr in ("transform", "pre_transform", "pre_filter", "post_filter"):
                if attr not in kwargs:
                    kwargs[attr] = getattr(dataset, attr, None)

            super().__init__(
                root=dataset.root,
                prefix=_repr_filter(filter),
                remove_hydrogen=dataset.remove_hydrogen,
                residue_representation=dataset.residue_representation,  # type: ignore
                require_kissim_residues=dataset.require_kissim_residues,
                use_multiprocessing=dataset.use_multiprocessing,
                **kwargs,
            )

        def make_data_list(self) -> List[HeteroData]:
            return [data for data in dataset]  # type: ignore

        def filter_transform(self, data_list: List[HeteroData]) -> List[HeteroData]:
            filtered = [data for data in data_list if filter(data)]
            log_file = Path(self.processed_dir) / "filter.log"
            log_file.write_text(f"Source: {Path(dataset.processed_dir).absolute()}")
            return filtered

    return FilteredDataset


def apply_transform_instance_permament(
    dataset: KinodataDocked, transform: Callable[[HeteroData], Optional[HeteroData]]
):
    desc = f"Applying permamnent transform {transform} to {dataset}..."
    transformed_data_list = []
    for index in tqdm(dataset.indices()):
        data = dataset.get(index)
        transformed_data = transform(data)
        if transformed_data is None:
            continue
        transformed_data_list.append(transformed_data)
    print("Done! Collating transformed data list...")
    data, slices = dataset.collate(transformed_data_list)
    dataset.data = data
    dataset.slices = slices
    dataset._data_list = None
    return dataset


def process_idx(args):
    (
        ident,
        smiles,
        ligand,
        activity,
        activity_type,
        pocket_file,
        docking_score,
        posit_prob,
        structure_id,
        pocket_sequence,
        predicted_rmsd,
        remove_hydrogen,
        residue_representation,
        require_kissim_residues,
    ) = args
    data = HeteroData()

    try:
        ligand_scaffold = mol_to_scaffold(ligand)
        if not remove_hydrogen:
            AddHs(ligand)
        Kekulize(ligand)
        data = set_atoms(ligand, data, NodeType.Ligand)
        data = set_bonds(ligand, data, NodeType.Ligand)

        pocket = MolFromMol2File(str(pocket_file))
        if not remove_hydrogen:
            AddHs(pocket)
        Kekulize(pocket)
        data = set_atoms(pocket, data, NodeType.Pocket)
        data = set_bonds(pocket, data, NodeType.Pocket)

        if residue_representation == "sequence":
            data = add_onehot_residues(data, pocket_sequence)
        elif residue_representation == "structural":
            data = add_pocket_information(data, pocket_file)
        elif residue_representation is not None:
            raise ValueError(residue_representation)

    except Exception as e:
        print(e)
        return None

    if require_kissim_residues:
        kissim_fp = load_kissim(structure_id)
        if kissim_fp is None:
            return None
        data = add_kissim_fp(data, kissim_fp, subset=PHYSICOCHEMICAL + STRUCTURAL)

    data.y = torch.tensor(activity).view(1)
    data.docking_score = torch.tensor(docking_score).view(1)
    data.posit_prob = torch.tensor(posit_prob).view(1)
    data.predicted_rmsd = torch.tensor(predicted_rmsd)
    data.pocket_sequence = pocket_sequence
    data.scaffold = ligand_scaffold
    data.activity_type = activity_type
    data.ident = ident
    data.smiles = smiles

    return data


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    dataset = KinodataDocked(remove_hydrogen=True, use_multiprocessing=True)
    print(dataset[0])
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=True)))
    print(batch)
