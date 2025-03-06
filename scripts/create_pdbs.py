from collections import namedtuple
from pathlib import Path
from typing import Iterable
from rdkit import Chem
import subprocess as sp
import gzip
from kinodata.data import RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing as mp
import os.path as osp
import biopandas.pdb as PDB

KlifsPocket = namedtuple("KlifsPocket", ["klifs_structure_id", "pocket_mol2_file"])
DockedLigand = namedtuple("BoundLigand", ["chembl_activity_id", "mol"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def join_pdbs_obabel(
    pocket_pdb_file: Path,
    ligand_pdb_file: Path,
    out: Path,
):
    pocket_klifs_structure_id = pocket_pdb_file.stem.split("_")[0]
    chembl_activity_id = ligand_pdb_file.stem.split("_")[0]
    complex_pdb_file = (
        out / f"{pocket_klifs_structure_id}_{chembl_activity_id}_complex.pdb"
    )
    if not complex_pdb_file.exists():
        sp.run(
            [
                "obabel",
                str(pocket_pdb_file),
                str(ligand_pdb_file),
                "-O",
                str(complex_pdb_file),
                "--join",
            ]
        )
    return complex_pdb_file


def join_pdbs_biopandas(
    pocket_pdb_file: Path,
    ligand_pdb_file: Path,
    out: Path,
):
    raise NotImplementedError


def create_complex_pdbs(
    pocket_pdb_files: Iterable[Path],
    ligand_pdb_files: Iterable[Path],
    out: Path | None,
):
    if out is None:
        out = Path.cwd()
    if not out.exists():
        out.mkdir()
    args = zip(pocket_pdb_files, ligand_pdb_files, [out] * len(pocket_pdb_files))
    logger.info("Creating complex PDB files...")
    with mp.Pool(12) as pool:
        complex_pdb_files = list(
            tqdm(pool.starmap(join_pdbs_obabel, args), total=len(pocket_pdb_files))
        )
    return complex_pdb_files


def create_pocket_pdbs(klifs_pockets: Iterable[KlifsPocket], out: Path | None):
    if out is None:
        out = Path.cwd()
    if not out.exists():
        out.mkdir()
    logger.info("Creating pocket PDB files...")
    pocket_pdb_files = []
    for pocket in tqdm(klifs_pockets):
        pocket_pdb_file = out / f"{pocket.klifs_structure_id}_pocket.pdb"
        pocket_pdb_files.append(pocket_pdb_file)
        if pocket_pdb_file.exists():
            continue
        sp.run(
            [
                "obabel",
                str(pocket.pocket_mol2_file),
                "-O",
                str(pocket_pdb_file),
            ]
        )
    return pocket_pdb_files


def create_ligand_pdbs(ligands: Iterable[DockedLigand], out: Path | None):
    if out is None:
        out = Path.cwd()
    if not out.exists():
        out.mkdir()
    logger.info("Creating ligand PDB files...")
    ligand_pdb_files = []
    for ligand in tqdm(ligands):
        ligand_pdb_file = out / f"{ligand.chembl_activity_id}_ligand.pdb"
        ligand_pdb_files.append(ligand_pdb_file)
        if ligand_pdb_file.exists():
            continue
        Chem.MolToPDBFile(ligand.mol, str(ligand_pdb_file))
    return ligand_pdb_files


def main():
    kinodata3d = pd.read_csv(PROCESSED_DATA_DIR / "kinodata3d_v2.csv")
    kinodata3d["activities.activity_id"] = kinodata3d["activities.activity_id"].astype(
        int
    )
    ligand_sdf = RAW_DATA_DIR / "kinodata_docked_v2.sdf.gz"
    mols = []
    idents = []

    logger.info("Reading ligand SDF file...")
    with gzip.open(ligand_sdf) as f:
        for mol in tqdm(Chem.ForwardSDMolSupplier(f, removeHs=False, sanitize=False)):
            if mol is None:
                continue
            idents.append(mol.GetIntProp("activities.activity_id"))
            mols.append(mol)
    kinodata3d = kinodata3d.merge(
        pd.DataFrame({"activities.activity_id": idents, "ligand_mol": mols}),
        on="activities.activity_id",
    )
    klifs_pockets = []
    docked_ligands = []
    for _, row in kinodata3d.iterrows():
        klifs_pockets.append(
            KlifsPocket(
                klifs_structure_id=row["similar.klifs_structure_id"],
                pocket_mol2_file=Path(row["pocket_mol2_file"]),
            )
        )
        docked_ligands.append(
            DockedLigand(
                chembl_activity_id=row["activities.activity_id"],
                mol=row["ligand_mol"],
            )
        )
    ligand_pdb_files = create_ligand_pdbs(docked_ligands, RAW_DATA_DIR / "pdb")
    pocket_pdb_files = create_pocket_pdbs(klifs_pockets, RAW_DATA_DIR / "pdb")
    complex_pdb_files = create_complex_pdbs(
        pocket_pdb_files=pocket_pdb_files,
        ligand_pdb_files=ligand_pdb_files,
        out=RAW_DATA_DIR / "pdb",
    )
    logger.info(f"Done! Created {len(complex_pdb_files)} complex PDB files.")


if __name__ == "__main__":
    main()
