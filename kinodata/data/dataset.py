from pathlib import Path
from typing import Callable, List
import torch

import pandas as pd
import requests
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from torch_geometric.data import InMemoryDataset, HeteroData
from tqdm import tqdm
from kinodata.transform.add_distances import AddDistancesAndInteractions
from rdkit.Chem.rdchem import BondType as BT
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import logging

BOND_TYPE_TO_IDX = defaultdict(int)  # other bonds will map to 0
BOND_TYPE_TO_IDX[BT.SINGLE] = 1
BOND_TYPE_TO_IDX[BT.DOUBLE] = 2
BOND_TYPE_TO_IDX[BT.TRIPLE] = 3
NUM_BOND_TYPES = 4


_DATA = Path(__file__).parents[2] / "data"


class KinodataDocked(InMemoryDataset):
    def __init__(
        self,
        root: str = _DATA,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["data_clean.csv"]  # TODO add pdb and mol2 files as zip at some point

    @property
    def processed_file_names(self) -> List[str]:
        return ["kinodata_docked.pt"]

    def download(self):
        # TODO at some point set up public download?
        pass

    def process(self):

        print("Reading data frame..")
        df = pd.read_csv(self.raw_paths[0], index_col="ident")
        df.columns

        print("Gathering paths to ligand pdb files..")
        ligand_pdb_files = {
            int(fp.stem.split("_")[0]): fp
            for fp in (Path(self.raw_dir) / "pdbs" / "ligand").iterdir()
        }
        df["ligand_pdb_file"] = [
            ligand_pdb_files[ident] if ident in ligand_pdb_files else None
            for ident in df.index
        ]

        # sanity check
        missing_ligands = df["ligand_pdb_file"].isna()
        if missing_ligands.any():
            print(df[missing_ligands].head())
            raise RuntimeError

        print("Checking for missing pocket mol2 files...")
        # for some reason pandas reads structure ID as a float
        # we drop those
        df = df[df["structure_ID"].notna()]
        df["structure_ID"] = df["structure_ID"].astype(int)
        # get pocket mol2 files
        pocket_dir = Path(self.raw_dir) / "mol2" / "pocket"
        if not pocket_dir.exists():
            pocket_dir.mkdir(parents=True)

        pbar = tqdm(df.iterrows(), total=len(df))
        for ident, row in pbar:
            structure_id = row["structure_ID"]
            fp = pocket_dir / f"{structure_id}_pocket.mol2"
            if fp.exists():
                continue
            resp = requests.get(
                "https://klifs.net/api/structure_get_pocket",
                params={"structure_ID": structure_id},
            )
            if resp.ok:
                fp.write_bytes(resp.content)

        pocket_mol2_files = {
            int(fp.stem.split("_")[0]): fp
            for fp in (Path(self.raw_dir) / "mol2" / "pocket").iterdir()
        }
        df["pocket_mol2_file"] = [
            pocket_mol2_files[row["structure_ID"]] for _, row in df.iterrows()
        ]

        def atomic_numbers(mol) -> torch.LongTensor:
            z = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
            for i, atom in enumerate(mol.GetAtoms()):
                z[i] = atom.GetAtomicNum()
            return z

        def coordinates(mol) -> torch.FloatTensor:
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            return torch.from_numpy(pos)

        def add_atoms(mol, data, key):
            data[key].z = atomic_numbers(mol)
            data[key].pos = coordinates(mol)
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
        removeHs = False
        skipped = []
        print("Creating PyG data objects..")
        for ident, row in tqdm(df.sample(100).iterrows(), total=len(df)):
            data = HeteroData()

            ligand = Chem.MolFromPDBFile(
                str(row["ligand_pdb_file"]), removeHs=removeHs, sanitize=True
            )
            data = add_atoms(ligand, data, "ligand")

            if ligand is None:
                skipped.append(str(ident))
                continue
            ligand_template = Chem.MolFromSmiles(
                row["compound_structures.canonical_smiles"]
            )
            try:
                ligand = AllChem.AssignBondOrdersFromTemplate(ligand_template, ligand)
                data = add_bonds(ligand, data, "ligand")
            except (Chem.rdchem.AtomValenceException, ValueError):
                skipped.append(str(ident))
                continue

            pocket = Chem.rdmolfiles.MolFromMol2File(
                str(row["pocket_mol2_file"]), removeHs=removeHs, sanitize=False
            )
            if pocket is None:
                skipped.append(str(ident))
                continue
            data = add_atoms(pocket, data, "pocket")

            data.y = torch.tensor(row["activities.standard_value"]).view(1)

            data_list.append(data)

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
    dataset = KinodataDocked(_DATA)
    dataset.process()
    dataset = KinodataDocked(transform=AddDistancesAndInteractions(radius=5.0))

    print(len(dataset))
    data = dataset[0]
    _, et = data.metadata()
    for a, e, b in et:
        if e == "interacts":
            print((a,e,b), data[a, e, b].dist.max())
