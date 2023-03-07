from collections import defaultdict
from pathlib import Path
from typing import Callable, List
from warnings import warn

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import requests
import torch
import torch.nn.functional as F
import multiprocessing as mp
from rdkit import RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdmolops import AddHs
from torch import Tensor
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from kinodata.transform.add_distances import AddDistancesAndInteractions
from kinodata.transform.filter_activity import FilterActivityType

BOND_TYPE_TO_IDX = defaultdict(int)  # other bonds will map to 0
BOND_TYPE_TO_IDX[BT.SINGLE] = 1
BOND_TYPE_TO_IDX[BT.DOUBLE] = 2
BOND_TYPE_TO_IDX[BT.TRIPLE] = 3
NUM_BOND_TYPES = len(BOND_TYPE_TO_IDX) + 1


_DATA = Path(__file__).parents[2] / "data"


class KinodataDocked(InMemoryDataset):
    def __init__(
        self,
        root: str = str(_DATA),
        remove_hydrogen: bool = True,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = FilterActivityType(["pIC50"]),
    ):
        self.remove_hydrogen = remove_hydrogen

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["kinodata_docked_filtered.sdf.gz"]

    @property
    def processed_file_names(self) -> List[str]:
        return "kinodata_docked.pt"

    @property
    def pocket_dir(self) -> Path:
        return Path(self.raw_dir) / "mol2" / "pocket"

    def download(self):
        # TODO at some point set up public download?
        pass

    def make_df_from_raw(self) -> pd.DataFrame:
        print("Reading data frame..")
        df = PandasTools.LoadSDF(
            self.raw_paths[0],
            smilesName="compound_structures.canonical_smiles",
            molColName="molecule",
            embedProps=True,
        )
        df.set_index("ID", inplace=True)

        print("Checking for missing pocket mol2 files...")
        df["similar.klifs_structure_id"] = df["similar.klifs_structure_id"].astype(int)
        # get pocket mol2 files
        if not self.pocket_dir.exists():
            self.pocket_dir.mkdir(parents=True)

        struc_ids = df['similar.klifs_structure_id'].unique()
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
            if resp.ok:
                fp.write_bytes(resp.content)

        pocket_mol2_files = {
            int(fp.stem.split("_")[0]): fp for fp in (self.pocket_dir).iterdir()
        }
        df["pocket_mol2_file"] = [
            pocket_mol2_files[row["similar.klifs_structure_id"]] for _, row in df.iterrows()
        ]

        return df

    def process(self):

        RDLogger.DisableLog("rdApp.*")
        df = self.make_df_from_raw()

        def atomic_numbers(mol) -> Tensor:
            z = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
            for i, atom in enumerate(mol.GetAtoms()):
                z[i] = atom.GetAtomicNum()
            return z

        def atom_positions(mol) -> Tensor:
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            return torch.from_numpy(pos)

        def add_atoms(mol, data, key):
            data[key].z = atomic_numbers(mol)
            data[key].pos = atom_positions(mol)
            assert data[key].z.size(0) == data[key].pos.size(0)
            return data

        def add_bonds(mol, data, key):
            row, col = list(), list()
            bond_type_indices = []
            num_nodes = mol.GetNumAtoms()
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                type = bond.GetBondType()
                row.append(i)
                col.append(j)
                bond_type_indices.append(BOND_TYPE_TO_IDX[type])

            edge_index = torch.tensor([row, col], dtype=torch.long).view(2, -1)
            bond_type_indices = torch.tensor(bond_type_indices)

            # one-hot encode bond type
            edge_attr = F.one_hot(bond_type_indices, NUM_BOND_TYPES)

            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes)

            data[key, "bond", key].edge_index = edge_index
            data[key, "bond", key].edge_attr = edge_attr

            return data

        data_list = []
        skipped: List[str] = []
        print("Creating PyG data objects..")
        def process_idx(ident):
            print('start', ident)
            data = HeteroData()

            df.loc[ident]
            ligand = row["molecule"]
            data = add_atoms(ligand, data, "ligand")

            pocket = Chem.rdmolfiles.MolFromMol2File(
                str(row["pocket_mol2_file"]),
                removeHs=self.remove_hydrogen,
                sanitize=True,
            )
            if pocket is None:
                return None
            data = add_atoms(pocket, data, "pocket")

            data.y = torch.tensor(row["activities.standard_value"]).view(1)
            data.activity_type = row["activities.standard_type"]
            data.ident = ident

            print('stop', ident)
            return data

        with mp.Pool(os.cpu_count()) as pool:
            data_list = pool.map(process_idx, list(df.index))

        skipped = [ident for ident, data in zip(df.index, data_list) if data is None]
        data_list = [d for d in data_list if d is not None]
        if len(skipped) > 0:
            print(f"Skipped {len(skipped)} unprocessable entries.")
            (Path(self.root) / "skipped_idents.log").write_text("\n".join(skipped))

        if self.pre_filter is not None:
            print("Applying pre filter..")
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print("Applying pre transform..")
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = KinodataDocked(str(_DATA))
    dataset.process()
    dataset = KinodataDocked(transform=AddDistancesAndInteractions(radius=5.0))
