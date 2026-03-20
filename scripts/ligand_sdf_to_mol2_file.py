from kinodata.data.dataset import process_raw_data

from pathlib import Path
from rdkit import Chem

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TARGET_DIR = Path(__file__).parent.parent / "data" / "docktgrid"


def write_mol2_file(molecule, properties: dict, output_path: Path):
    mol2_block = Chem.MolToMol2Block(molecule)
    with open(output_path, "w") as f:
        f.write(mol2_block)
        for key, value in properties.items():
            f.write(f"@<TRIPOS>COMMENT\n{key}: {value}\n")


if __name__ == "__main__":
    if not TARGET_DIR.exists():
        TARGET_DIR.mkdir()
    kinodata3d_df = process_raw_data(
        RAW_DIR,
        remove_hydrogen=False,
    )
