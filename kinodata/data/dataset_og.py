import logging
import multiprocessing as mp
import os
import os.path as osp
import re
from time import sleep
import warnings
from functools import cached_property, partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
)
from dataclasses import dataclass

import pandas as pd
import requests  # type : ignore
import torch
from rdkit.Chem import AddHs, Kekulize, MolFromMol2File, PandasTools  # type: ignore
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm
from rdkit import RDLogger

from kinodata.data.featurization.biopandas import add_pocket_information
from kinodata.data.featurization.rdkit import (
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
from kinodata.data.utils.pocket_sequence_klifs import (
    add_pocket_sequence,
    get_pocket_sequence,
    CachedSequences,
)
from kinodata.types import NodeType
from kinodata.data.utils.scaffolds import mol_to_scaffold
from kinodata.transform.filter_activity import FilterActivityType, ActivityTypes

#_DATA = Path(__file__).parents[2] / "data"
_DATA = Path(f"{Path(__file__).parents[2]}/data")


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", obj.__repr__())


def process_raw_data(
   raw_dir: Path,
    file_name: str = "kinodata_docked_v2.sdf.gz",
    remove_hydrogen: bool = True,
    pocket_dir: Optional[Path] = None,
    pocket_sequence_file: Optional[Path] = None,
    activity_type_subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
    if pocket_dir is None:
        pocket_dir = f"{raw_dir}/mol2/pocket"
    if pocket_sequence_file is None:
        pocket_sequence_file = f"{raw_dir}/pocket_sequences.csv"

    #raw_dir='/home/raquellrdc/Desktop/postdoc/fast_ml_final/new_data_try'

    raw_fp = f"{raw_dir}/{file_name}"
    file_path=raw_fp

    #print(file_path)

    # Check if the file exists
    if os.path.exists(file_path):
        print("sdf file exists.")
    else:
         print("sdf file does not exist.")
 
  
    print(f"Reading data frame from {raw_fp}...")
    df = PandasTools.LoadSDF(
         raw_fp,
         smilesName="compound_structures.canonical_smiles",
         molColName="molecule",
         embedProps=True,
         removeHs=remove_hydrogen,
     )
     
    if activity_type_subset is not None:
         
         #df = df.query("activities.standard_type in @activity_type_subset")
         df = df.query("`activities.standard_type` == @activity_type_subset")
    df["activities.standard_value"] = df["activities.standard_value"].astype(float)
    df["docking.predicted_rmsd"] = df["docking.predicted_rmsd"].astype(float)

    print(f"Deduping data frame (current size: {df.shape[0]})...")
    group_key = [
         "compound_structures.canonical_smiles",
         "UniprotID",
         "activities.standard_type",
      ]
    mean_activity = (
         df.groupby(group_key).agg({"activities.standard_value": "mean"}).reset_index()
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
     # df.set_index("ID", inplace=True)
    print(f"{df.shape[0]} complexes remain after deduplication.")

    print("Checking for missing pocket mol2 files...")
    df["similar.klifs_structure_id"] = (
         df["similar.klifs_structure_id"].astype(float).astype(int)
     )
    # get pocket mol2 files
    if not pocket_dir.exists():
         pocket_dir.mkdir(parents=True)

    struc_ids = df["similar.klifs_structure_id"].unique()
    pbar = tqdm(struc_ids, total=len(struc_ids))
    for structure_id in pbar:
        fp = pocket_dir / f"{structure_id}_pocket.mol2"
        if fp.exists():
             continue
        resp = requests.get(
             "https://klifs.net/api/structure_get_pocket",
             params={"structure_ID": structure_id},
         )
        resp.raise_for_status()
        fp.write_bytes(resp.content)

    pocket_mol2_files = {
         int(fp.stem.split("_")[0]): fp for fp in (pocket_dir).iterdir()
     }
    df["pocket_mol2_file"] = [
        pocket_mol2_files[row["similar.klifs_structure_id"]] for _, row in df.iterrows()
     ]

     # backwards compatability
    df["ident"] = df.index

    print("Adding pocket sequences...")
     # KLIFS API now sometimes decides to timeout
    print(df.shape)
    while True:
         try:
             with CachedSequences(pocket_sequence_file) as sequence_cache:
                 klifs_ids = df["similar.klifs_structure_id"].tolist()
                 for klifs_id in tqdm(klifs_ids):
                     if klifs_id in sequence_cache:
                         continue
                     sequence = get_pocket_sequence([klifs_id])
                     sequence_cache[klifs_id] = sequence
                 df_sequences = sequence_cache.to_data_frame()
                 df = pd.merge(df, df_sequences, on="similar.klifs_structure_id")
             break
         except Exception as e:
             print(f"Querying KLIFS for sequence from structure id raised {e}")
             print("Retrying..")
             sleep(10)
    print(df.shape)
 
     # if pocket_sequence_file.exists():
     #     print(f"from cached file {pocket_sequence_file}.")
 #    #     pocket_sequences = pd.read_csv(pocket_sequence_file)
  #   #     pocket_sequences["ident"] = pocket_sequences["ident"].astype(str)
   #  #     df = pd.merge(df, pocket_sequences, left_on="ident", right_on="ident")
   #  # else:
   #  #     print("from KLIFS.")
   #  #     df = add_pocket_sequence(df, pocket_sequence_key="structure.pocket_sequence")
   #  #     df[["ident", "structure.pocket_sequence", "similar.klifs_structure_id"]].to_csv(
   #  #         pocket_sequence_file, index=False
   #  #     )

    return df


@dataclass
class ComplexInformation:
    kinodata_ident: str
    compound_smiles: str
    molecule: Any
    activity_value: float
    activity_type: str
    pocket_mol2_file: Path
    docking_score: float
    posit_probability: float
    klifs_structure_id: int
    pocket_sequence: str
    predicted_rmsd: float
    remove_hydrogen: bool

    @classmethod
    def from_raw(cls, raw_data: pd.DataFrame, **kwargs) -> List["ComplexInformation"]:
        return [
            cls(
                row["ident"],
                row["compound_structures.canonical_smiles"],
                row["molecule"],
                float(row["activities.standard_value"]),
                row["activities.standard_type"],
                Path(row["pocket_mol2_file"]),
                float(row["docking.chemgauss_score"]),
                float(row["docking.posit_probability"]),
                int(row["similar.klifs_structure_id"]),
                row["structure.pocket_sequence"],
                float(row["docking.predicted_rmsd"]),
                **kwargs,
            )
            for _, row in raw_data.iterrows()
        ]

    @cached_property
    def ligand(self) -> Any:
        ligand = self.molecule
        if not self.remove_hydrogen:
            AddHs(ligand)
        Kekulize(ligand)
        return ligand

    @cached_property
    def pocket(self) -> Any:
        pocket = MolFromMol2File(str(self.pocket_mol2_file))
        if not self.remove_hydrogen:
            AddHs(pocket)
        Kekulize(pocket)
        return pocket


class KinodataDockedAgnostic:
    def __init__(
        self,
        raw_dir: str = str(_DATA / "raw"),
        remove_hydrogen: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.remove_hydrogen = remove_hydrogen
        print(f"Loading raw data from {self.raw_dir}...")
        self._df = process_raw_data(self.raw_dir, remove_hydrogen=remove_hydrogen)
        print("Converting to data list...")
        self.data_list = ComplexInformation.from_raw(
            self._df, remove_hydrogen=self.remove_hydrogen
        )
        print("Done!")

    @property
    def data_frame(self) -> pd.DataFrame:
        return self._df

    def __iter__(self) -> Iterator[ComplexInformation]:
        return iter(self.data_list)

    def __getitem__(self, idx: int) -> ComplexInformation:
        return self.data_list[idx]


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
        require_kissim_residues: bool = False,
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
        self.make_pyg_data = partial(
            process_pyg,
            residue_representation=self.residue_representation,
            require_kissim_residues=self.require_kissim_residues,
        )
        self.num_processes = num_processes
        self.post_filter = post_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def pocket_sequence_file(self) -> Path:
        return Path(self.raw_dir) / "pocket_sequences.csv"

    @property
    def raw_file_names(self) -> List[str]:
        return ["kinodata_docked_v2.sdf.gz"]
        #return ["kinodata_docked_full.sdf.gz"]

    @property
    def processed_file_names(self) -> List[str]:
        # TODO add preprocessed kssim fingerprints?
        return [
            f"kinodata_docked_v2.pt",
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
        return process_raw_data(
            Path(self.raw_dir),
            self.raw_file_names[0],
            self.remove_hydrogen,
            self.pocket_dir,
            self.pocket_sequence_file,
            activity_type_subset=["pIC50"],
        )

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
        print("Entering make_data_list")
        RDLogger.DisableLog("rdApp.*")
        data_list = []
        complex_info = ComplexInformation.from_raw(
            self.df, remove_hydrogen=self.remove_hydrogen
        )
        #import numpy as np
        #np.save('/home/raquellrdc/Desktop/postdoc/fast_ml_final/new_data_try/complex_info.npy', complex_info)
        #if self.use_multiprocessing:
        #    print('I am doing option 1')
        #    print(len(complex_info))
        #    print(complex_info[0])
        #    import numpy as np
        #    np.save(complex_info)
        #    print('saved')
        #    tasks = [
        #        (_complex, self.residue_representation, self.require_kissim_residues)
        #        for _complex in complex_info
        #    ]
        #    print(tasks)
        #    
        #    with mp.Pool(os.cpu_count()) as pool:
        #        data_list = pool.map(_process_pyg, tqdm(tasks))
        #else:
        print('I am doing option 2')



        ####
        from joblib import Parallel, delayed

        # Define the function to process a single item
        def process_single_item(item, residue_representation, require_kissim_residues):
            return process_pyg(item, residue_representation, require_kissim_residues)

        # Define the number of parallel processes to use
        n_jobs = 14  # Use all available CPU cores

        # Parallelize the processing of items
        data_list = Parallel(n_jobs=n_jobs)(
        delayed(process_single_item)(item, self.residue_representation, self.require_kissim_residues)
        for item in tqdm(complex_info))
        ###
        #process = partial(
        #        process_pyg,
        #        residue_representation=self.residue_representation,
        #        require_kissim_residues=self.require_kissim_residues,
        #    )
        #data_list = list(map(process, tqdm(complex_info)))

        data_list = [d for d in data_list if d is not None]
        print("Exiting make_data_list")
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


def repr_filter(filter: Callable):
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
                prefix=repr_filter(filter),
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
    print(f"Applying permamnent transform {transform} to {dataset}...")
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


# work-around hack for multiprocessing.Pool using pickle.
# pickling only works at top-level, impeding
# more elegant solution with functools.partial
def _process_pyg(args) -> Optional[HeteroData]:
    return process_pyg(*args)


def process_pyg(
    complex: Optional[ComplexInformation] = None,
    residue_representation: Literal["sequence", "structural", None] = "sequence",
    require_kissim_residues: bool = False,
) -> Optional[HeteroData]:
    if complex is None:
        logging.warning(f"process_pyg received None as complex input")
        return None
    data = HeteroData()
    try:
        ligand = complex.ligand
        data = set_atoms(ligand, data, NodeType.Ligand)
        data = set_bonds(ligand, data, NodeType.Ligand)
        ligand_scaffold = mol_to_scaffold(ligand)

        pocket = complex.pocket
        data = set_atoms(pocket, data, NodeType.Pocket)
        data = set_bonds(pocket, data, NodeType.Pocket)

        if residue_representation == "sequence":
            data = add_onehot_residues(data, complex.pocket_sequence)
        elif residue_representation == "structural":
            data = add_pocket_information(data, complex.pocket_mol2_file)
        elif residue_representation is not None:
            raise ValueError(residue_representation)

    except Exception as e:
        logging.warning(f"Exception: {e} when processing {complex}")
        return None

    if require_kissim_residues:
        kissim_fp = load_kissim(complex.klifs_structure_id)
        if kissim_fp is None:
            return None
        data = add_kissim_fp(data, kissim_fp, subset=PHYSICOCHEMICAL + STRUCTURAL)

    data.y = torch.tensor(complex.activity_value).view(1)
    data.docking_score = torch.tensor(complex.docking_score).view(1)
    data.posit_prob = torch.tensor(complex.docking_score).view(1)
    data.predicted_rmsd = torch.tensor(complex.predicted_rmsd).view(1)
    data.pocket_sequence = complex.pocket_sequence
    data.scaffold = ligand_scaffold
    data.activity_type = complex.activity_type
    data.ident = complex.kinodata_ident
    data.smiles = complex.compound_smiles
    return data
