import subprocess as sp
from pathlib import Path
from kinodata.data import RAW_DATA_DIR
import os
import shutil
import logging
import multiprocessing as mp
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_plip_text_report(
    klifs_structure_id: int,
    chembl_activity_id: int,
):
    plip_raw = f"data/plip/raw/{chembl_activity_id}"
    plip_processed = f"data/plip/text_reports/{chembl_activity_id}.plip.txt"
    input_complex = (
        f"data/raw/pdb/{klifs_structure_id}_{chembl_activity_id}_complex.pdb"
    )
    os.mkdir(plip_raw)
    sp.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{Path.cwd()}:/results",
            "-w",
            "/results",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "pharmai/plip:latest",
            "-tq",
            "-o",
            plip_raw,
            "-f",
            input_complex,
        ]
    )
    if os.path.exists(plip_raw):
        shutil.move(f"{plip_raw}/report.txt", plip_processed)
    else:
        Path(plip_processed).write_text("FAILED")
    shutil.rmtree(plip_raw)
    return plip_processed


def plip_many(
    klifs_structure_ids: list[int],
    chembl_activity_ids: list[int],
):
    with mp.Pool(12) as pool:
        results = pool.starmap(
            create_plip_text_report, zip(klifs_structure_ids, chembl_activity_ids)
        )
    return results


if __name__ == "__main__":
    print(Path.cwd())
    indir = RAW_DATA_DIR / "pdb"
    complex_files = list(indir.glob("*complex.pdb"))
    logger.info(f"Found {len(complex_files)} complex files")
    klifs_structure_ids = [int(f.name.split("_")[0]) for f in complex_files]
    chembl_activity_ids = [int(f.name.split("_")[1]) for f in complex_files]
    plip_many(klifs_structure_ids, chembl_activity_ids)
