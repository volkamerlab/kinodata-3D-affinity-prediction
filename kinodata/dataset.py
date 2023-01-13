from pathlib import Path
from typing import Callable, List
import torch
import pandas as pd
import rdkit.Chem as Chem
from torch_geometric.data import InMemoryDataset, HeteroData
from tqdm import tqdm

_DATA = Path(__file__).parents[1] / "data"

class KinodataDocked(InMemoryDataset):
    def __init__(
        self,
        root: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self) -> List[str]:
        return ["data_clean.csv"]  # TODO add pdb and mol2 files as zip at some point

    def processed_file_names(self) -> List[str]:
        return ["kinodata_docked.pt"]
    
    def download(self):
        # TODO at some point set up public download?
        pass

    def process(self):

        df = pd.read_csv(self.raw_paths[0], index_col="ident")
        df.columns

        ligand_pdb_files = {
            int(fp.stem.split("_")[0]): fp
            for fp in (Path(self.raw_dir) / "pdbs" / "ligand").iterdir()
        }
        df["ligand_pdb_file"] = [
            ligand_pdb_files[ident] if ident in ligand_pdb_files else None
            for ident in df.index
        ]

        # sanity check
        assert df["ligand_pdb_file"].notna().all()
        df = df[df["structure_ID"].notna()]
        df["structure_ID"] = df["structure_ID"].astype(int)

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

        data_list = []
        removeHs = False
        skipped = []
        for ident, row in tqdm(df.iterrows(), total=len(df)):
            data = HeteroData()

            ligand = Chem.MolFromPDBFile(
                str(row["ligand_pdb_file"]), removeHs=removeHs, sanitize=False
            )
            if ligand is None:
                skipped.append(str(ident))
                continue
            data = add_atoms(ligand, data, "ligand")

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
            (Path(self.root) / "skipped_idents.log").write_text(
                "\n".join(skipped)
            )
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = KinodataDocked(_DATA)
    dataset.process()
    