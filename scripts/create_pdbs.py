import gzip
import logging
import multiprocessing as mp
import subprocess as sp
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Iterable

import biopandas.mol2 as MOL2
import biopandas.pdb as PDB
import pandas as pd
from rdkit import Chem
from rdkit.Chem import CombineMols, MolFromPDBFile, MolToPDBFile
from resmo.protein_model import Mol2ProteinModel
from tqdm import tqdm

from kinodata.data import PROCESSED_DATA_DIR, RAW_DATA_DIR
from kinodata.data.voxel.klifs_parser import klifs_mol2_columns

KlifsPocket = namedtuple("KlifsPocket", ["klifs_structure_id", "pocket_mol2_file"])
DockedLigand = namedtuple("BoundLigand", ["chembl_activity_id", "mol"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OUT_DIR = RAW_DATA_DIR / "pdb"
if not OUT_DIR.exists():
    OUT_DIR.mkdir()


def join_pdbs_obabel(
    pocket_pdb_file: Path,
    ligand_pdb_file: Path,
    out: Path,
):
    raise RuntimeError("obabel is broken af")
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


def join_pdbs_rdkit(
    pocket_pdb_file: Path,
    ligand_pdb_file: Path,
    out: Path,
):
    pocket_klifs_structure_id = pocket_pdb_file.stem.split("_")[0]
    chembl_activity_id = ligand_pdb_file.stem.split("_")[0]
    complex_pdb_file = (
        out / f"{pocket_klifs_structure_id}_{chembl_activity_id}_complex.pdb"
    )
    if complex_pdb_file.exists():
        return complex_pdb_file
    pocket = MolFromPDBFile(str(pocket_pdb_file), sanitize=False, removeHs=False)
    ligand = MolFromPDBFile(str(ligand_pdb_file), sanitize=False, removeHs=False)
    complex_mol = CombineMols(pocket, ligand)
    MolToPDBFile(complex_mol, str(complex_pdb_file))
    return complex_pdb_file


def create_complex_pdbs(
    pocket_pdb_files: Iterable[Path],
    ligand_pdb_files: Iterable[Path],
    out: Path,
):
    args = zip(pocket_pdb_files, ligand_pdb_files, [out] * len(pocket_pdb_files))
    logger.info("Creating complex PDB files...")
    complex_pdb_files = []
    for pocket_pdb_file, ligand_pdb_file, out_dir in tqdm(args):
        complex_pdb_file = join_pdbs_rdkit(pocket_pdb_file, ligand_pdb_file, out_dir)
        complex_pdb_files.append(complex_pdb_file)
    return complex_pdb_files


def fix_obabel_converted_pdb_file(pocket: KlifsPocket, out: Path):
    structure_id = pocket.klifs_structure_id
    origin_mol2 = Mol2ProteinModel.from_file(str(pocket.pocket_mol2_file))
    pdb_lines = (out / f"{structure_id}_pocket.pdb").read_text().splitlines()
    num_pdb_atoms = sum(
        1 for line in pdb_lines if line.startswith("ATOM") or line.startswith("HETATM")
    )
    assert num_pdb_atoms == len(origin_mol2.molecule.atoms)
    fixed_pdb_lines = []
    itr_mol2_atoms = iter(origin_mol2.molecule.atoms)
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom = next(itr_mol2_atoms)
            mol2_resname = atom.subst_name[:3]
            line = line.replace("HETATM", "ATOM  ")
            line = line[:17] + mol2_resname + line[20:]
        fixed_pdb_lines.append(line)
    (out / f"{structure_id}_pocket.pdb").write_text("\n".join(fixed_pdb_lines))


def create_pocket_pdbs(klifs_pockets: Iterable[KlifsPocket], out: Path):
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
        fix_obabel_converted_pdb_file(pocket, out)

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


def create_index_mapping(
    klifs_structure_id: int,
    perform_sanity_checks: bool = True,
    out: Path | None = None,
) -> pd.DataFrame | None:
    mol2_file = RAW_DATA_DIR / "mol2" / "pocket" / f"{klifs_structure_id}_pocket.mol2"
    file_pattern = f"{klifs_structure_id}_*_complex.pdb"
    if perform_sanity_checks:
        sanity_check = defaultdict(list)
        for complex_file in (RAW_DATA_DIR / "pdb").glob(file_pattern):
            lines = tuple(
                [
                    "".join(line.strip().split())
                    for line in complex_file.read_text().split("\n")
                    if line.strip().startswith("ATOM")
                ]
            )
            sanity_check[lines].append(complex_file)
        if len(sanity_check) != 1:
            logger.warning(
                f"Sanity check failed! Found {len(sanity_check)} different complex PDB files for KLIFS structure {klifs_structure_id}."
            )
            for file_list in list(sanity_check.values())[:3]:
                logger.warning(f"Example: {file_list[0]}")
            return None
    complex_file = list((RAW_DATA_DIR / "pdb").glob(file_pattern))[0]
    pdb_complex = PDB.PandasPdb().read_pdb(str(complex_file))
    resnr_pdb_complex = (
        pdb_complex.df["ATOM"][["residue_name", "residue_number", "chain_id"]]
        .drop_duplicates()
        .rename(
            columns={
                "residue_name": "residue_name_pdb",
                "residue_number": "residue_number_pdb",
                "chain_id": "chain_id_pdb",
            }
        )
    )

    mol2_pocket = MOL2.pandas_mol2.PandasMol2()
    mol2_pocket.read_mol2(mol2_file, columns=klifs_mol2_columns)
    df = mol2_pocket.df.drop_duplicates(subset=["subst_id", "subst_name"])
    resnr_mol2_complex = df[["subst_id", "subst_name"]].rename(
        columns={"subst_id": "residue_number_mol2", "subst_name": "residue_name_mol2"}
    )
    if perform_sanity_checks:
        resname_mol2 = resnr_mol2_complex["residue_name_mol2"].str[:3]
        resname_pdb = resnr_pdb_complex["residue_name_pdb"]
        if not (match := (resname_mol2 == resname_pdb)).all():
            logger.warning(
                f"Sanity check failed! Mismatch between residue names in PDB and MOL2 files for KLIFS structure {klifs_structure_id}."
            )
            for i, (r_mol2, r_pdb) in enumerate(zip(resname_mol2, resname_pdb)):
                if r_mol2 != r_pdb:
                    logger.warning(
                        f"Residue {i}: MOL2: {r_mol2}, PDB: {r_pdb}, Match: {match[i]}"
                    )
            return None
    resnr_mol2_complex["residue_number_uniprot"] = resnr_mol2_complex[
        "residue_name_mol2"
    ].str[3:]
    mapping_df = pd.concat([resnr_pdb_complex, resnr_mol2_complex], axis=1)
    if out is not None:
        mapping_df.to_csv(
            out / f"{klifs_structure_id}_residue_mapping.csv", index=False
        )
    return mapping_df


def create_index_mapping_dfs(klifs_pockets: Iterable[int], out: Path | None = None):
    if out is None:
        out = RAW_DATA_DIR / "pdb" / "resnr_mapping"
    if not out.exists():
        out.mkdir()
    logger.info(f"Creating residue mapping CSV files at {out}...")
    for klifs_structure_id in tqdm(klifs_pockets):
        create_index_mapping(klifs_structure_id, out=out)


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
    ligand_pdb_files = create_ligand_pdbs(docked_ligands, OUT_DIR)
    pocket_pdb_files = create_pocket_pdbs(klifs_pockets, OUT_DIR)
    complex_pdb_files = create_complex_pdbs(pocket_pdb_files, ligand_pdb_files, OUT_DIR)
    logger.info(f"Done! Created {len(complex_pdb_files)} complex PDB files.")
    create_index_mapping_dfs(kinodata3d["similar.klifs_structure_id"].unique())


if __name__ == "__main__":
    main()
