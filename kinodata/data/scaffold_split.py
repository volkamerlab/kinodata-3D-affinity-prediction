from typing import Dict, Iterator, Protocol, Set, Generic, Tuple, TypeVar, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from kinodata.data.data_split import Split

ScaffoldSmiles = str
IdType = TypeVar("IdType")
PathLike = Union[Path, str]


class IdentifiableSmiles(Protocol, Generic[IdType]):
    ident: IdType
    smiles: str


class SmilesDataset(Protocol, Generic[IdType]):
    def __getitem__(self, index) -> IdentifiableSmiles[IdType]:
        ...

    def __iter__(self) -> Iterator[IdentifiableSmiles]:
        ...


class ScaffoldSplitting:
    @staticmethod
    def generate_scaffold_classes(
        dataset: SmilesDataset[IdType],
    ) -> Dict[ScaffoldSmiles, Set[IdType]]:
        scaffold_split: Dict[ScaffoldSmiles, Set[IdType]] = defaultdict(set)

        for index, item in enumerate(dataset):
            # https://www.blopig.com/blog/2021/06/out-of-distribution-generalisation-and-scaffold-splitting-in-molecular-property-prediction/
            mol = Chem.MolFromSmiles(item.smiles)
            mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            mol_scaffold_generic = MurckoScaffold.MakeScaffoldGeneric(mol_scaffold)
            smiles_scaffold_generic: ScaffoldSmiles = Chem.CanonSmiles(
                Chem.MolToSmiles(mol_scaffold_generic)
            )

            scaffold_split[smiles_scaffold_generic].add(item.ident)

        return scaffold_split

    @staticmethod
    def find_assignment(
        scaffolds: Dict[ScaffoldSmiles, Set[IdType]],
        target_train_size: float,
        skip_first_k: int = 3,
        seed: int = 0,
    ) -> Tuple[Dict[ScaffoldSmiles, Set[IdType]], Dict[ScaffoldSmiles, Set[IdType]]]:
        num_scaffolds = len(scaffolds)
        total_num_samples = sum(len(val) for val in scaffolds.values())
        target_test_size = 1 - target_train_size
        chunk_size = int(1 / (target_test_size))

        sc_smiles, members = zip(
            *sorted(scaffolds.items(), key=lambda scf: len(scf[1]), reverse=True)
        )
        rng = np.random.default_rng(seed)

        test_idxs = np.arange(skip_first_k, num_scaffolds, chunk_size)
        random_offsets = rng.integers(0, chunk_size, size=(test_idxs.shape[0],))
        test_idxs += random_offsets
        test_idxs[-1] = min(test_idxs[-1], num_scaffolds - 1)
        train = np.full(num_scaffolds, True)
        train[test_idxs] = False

        test_size = 0
        train_set = dict()
        test_set = dict()

        for i, (smiles, member_set) in enumerate(zip(sc_smiles, members)):
            if train[i]:
                train_set[smiles] = member_set
            elif test_size / total_num_samples < target_test_size:
                test_set[smiles] = member_set
                test_size += len(member_set)

        return train_set, test_set

    @staticmethod
    def split(
        dataset,
        train_size: float,
        seed: int,
    ) -> pd.DataFrame:
        print("Grouping by scaffolds..")
        scaffolds = ScaffoldSplitting.generate_scaffold_classes(dataset)
        print("Assigning scaffold groups to splits..")
        train, test = ScaffoldSplitting.find_assignment(
            scaffolds, train_size, seed=seed
        )
        split_data = {
            "ident": [],
            "split": [],
            "scaffold": [],
        }
        for data, label in ((train, "train"), (test, "test")):
            for scaffold, members in data.items():
                split_data["ident"].extend(list(members))
                split_data["split"].extend([label] * len(members))
                split_data["scaffold"].extend([scaffold] * len(members))

        df = pd.DataFrame(split_data)
        df["ident"] = df["ident"].astype(int)
        return df

    CACHE_FN = "scaffold_split_{seed}.csv"

    def __init__(
        self,
        train_size: float,
        val_size: float,
        test_size: float,
        cache_dir: PathLike,
    ) -> None:
        cache_dir = Path(cache_dir)
        assert np.isclose(train_size + val_size + test_size, 1.0)
        assert not cache_dir.is_file()
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.cache_dir = cache_dir

    def __call__(
        self,
        dataset,
        seed: int,
        return_df: bool = False,
        use_cache: bool = True,
    ) -> Union[Split, Tuple[Split, pd.DataFrame]]:
        cache = self.cache_dir / ScaffoldSplitting.CACHE_FN.format(seed=seed)
        if not cache.exists() or not use_cache:
            print(f"Creating scaffold split at {cache}..")
            df = ScaffoldSplitting.split(tqdm(dataset), self.train_size, seed)
            df.to_csv(cache, index=False)
        else:
            print(f"Loading scaffold split from {cache}..")
            df = pd.read_csv(cache)

        train_ident = df[df.split == "train"].ident.values
        test_ident = df[df.split == "test"].ident.values

        rng = np.random.default_rng(seed + 1)
        rng.shuffle(test_ident)
        pivot = int(
            test_ident.shape[0] * (self.val_size) / (self.val_size + self.test_size)
        )
        val_ident = test_ident[:pivot]
        test_ident = test_ident[pivot:]

        split = Split(train_ident, val_ident, test_ident)
        if return_df:
            return split, df
        return split


if __name__ == "__main__":
    from kinodata.data.dataset import KinodataDocked

    dataset = KinodataDocked()
    dataset = dataset
    splitter = ScaffoldSplitting(0.8, 0.1, 0.1, "data/splits/scaffold")
    for seed in [7, 11, 13, 17, 19]:
        split, df = splitter(dataset, seed, return_df=True)
        df_split = split.to_data_frame()
        df_joined = pd.merge(df, df_split, on="ident", how="inner")
        df_joined["split"] = df_joined["split_y"]
        df_joined.drop(columns=["split_x", "split_y"]).to_csv(
            f"scaffold_split_{seed}.csv", index=False
        )
