from collections import defaultdict
import random
import time
from pandas import DataFrame
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tqdm import tqdm


def _is_separator(line: str) -> bool:
    return all([c in ["+", "-", "="] for c in line.strip()])


def _parse_data_type(value: str) -> any:
    if value in ("True", "False"):
        return value == "True"
    if len(parts := value.split(".")) == 2:
        a, b = parts
        if a.isnumeric() and b.isnumeric():
            return float(value)
    if value.isnumeric():
        return int(value)
    if len(parts := value.split()) == 3:
        parts = [p.replace(",", "") for p in parts]
        try:
            return tuple([float(p) for p in parts])
        except ValueError:
            pass
    return value


def parse_rst_table(table: str | None = None, lines: list[str] | None = None) -> dict:
    if lines is None:
        if table is None:
            raise ValueError("Either table or lines must be provided")
        lines = table.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line for line in lines if not _is_separator(line)]
    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    data = []
    for line in lines[1:]:
        parts = [c.strip() for c in line.split("|")[1:-1]]
        data.append({k: _parse_data_type(v) for k, v in zip(header, parts)})
    return DataFrame(data)


def parse_rst_table_file(
    path: str,
    table_names: list[str] = None,
    max_num_ligands: int = None,
) -> dict[str, DataFrame] | None:
    with open(path, "r") as f:
        table = f.read()

    num_ligands = table.count("SMALLMOLECULE")
    if max_num_ligands is not None and num_ligands > max_num_ligands:
        return None

    lines = table.split("\n")
    iter_lines = iter(lines)
    result = defaultdict(list)
    skipped_line = None
    while True:
        if skipped_line:
            line = skipped_line
            skipped_line = None
        else:
            try:
                line = next(iter_lines)
            except StopIteration:
                break
        if any([name in line for name in table_names]):
            table_name = line.strip().replace("*", "").lower().replace(" ", "_")
            lines = []
            while True:
                try:
                    line = next(iter_lines)
                    if not _is_separator(line) and "|" not in line:
                        skipped_line = line
                        break
                    lines.append(line)
                except StopIteration:
                    break
            result[table_name].append(parse_rst_table(lines=lines))
    for key, df_list in result.items():
        if len(df_list) == 1:
            result[key] = df_list[0]
            continue
        result[key] = pd.concat(df_list, axis=0)
    return result


def test_known_table():
    table = """+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| RESNR | RESTYPE | RESCHAIN | RESNR_LIG | RESTYPE_LIG | RESCHAIN_LIG | SIDECHAIN | DIST_H-A | DIST_D-A | DON_ANGLE | PROTISDON | DONORIDX | DONORTYPE | ACCEPTORIDX | ACCEPTORTYPE | LIGCOO                 | PROTCOO               | 
+=======+=========+==========+===========+=============+==============+===========+==========+==========+===========+===========+==========+===========+=============+==============+========================+=======================+
| 19    | GLN     | A        | 283       | NFT         | A            | True      | 2.16     | 3.11     | 160.05    | True      | 153      | Nam       | 1649        | N2           | 2.820, 18.145, 6.806   | 3.976, 15.409, 7.712  | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| 25    | CYS     | A        | 283       | NFT         | A            | False     | 2.21     | 3.09     | 146.90    | True      | 183      | Nam       | 1649        | N2           | 2.820, 18.145, 6.806   | 3.306, 18.620, 9.817  | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| 61    | ASP     | A        | 283       | NFT         | A            | True      | 2.26     | 2.99     | 134.61    | True      | 451      | O3        | 1645        | N3           | -9.805, 23.545, 10.596 | -9.236, 20.949, 9.223 | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| 61    | ASP     | A        | 283       | NFT         | A            | True      | 1.98     | 2.99     | 170.22    | False     | 1645     | N3        | 451         | O3           | -9.805, 23.545, 10.596 | -9.236, 20.949, 9.223 | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| 66    | GLY     | A        | 283       | NFT         | A            | False     | 2.33     | 3.18     | 140.42    | False     | 1631     | N3        | 473         | O2           | 0.027, 24.446, 5.449   | 0.194, 25.010, 8.576  | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| 66    | GLY     | A        | 283       | NFT         | A            | False     | 2.05     | 2.95     | 150.59    | True      | 470      | Nam       | 1638        | O2           | 0.422, 22.065, 7.006   | -1.810, 23.106, 8.621 | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+
| 158   | ASN     | A        | 283       | NFT         | A            | False     | 1.95     | 2.92     | 167.45    | False     | 1629     | Nam       | 1199        | O2           | 1.645, 21.274, 5.306   | 3.137, 21.570, 2.817  | 
+-------+---------+----------+-----------+-------------+--------------+-----------+----------+----------+-----------+-----------+----------+-----------+-------------+--------------+------------------------+-----------------------+"""
    df = parse_rst_table(table)
    print(df.head())


def test_known_plip_report():
    plip_report_path = Path(__file__).parent.parent.parent / "tests" / "plip_report.txt"
    print(
        parse_rst_table_file(
            plip_report_path, table_names=["Hydrogen Bonds", "Hydrophobic Interactions"]
        )
    )


def test_random_plip_report():
    plip_path = Path(__file__).parent.parent.parent / "data" / "plip"
    plip_report_path = plip_path / "text_reports"
    random.seed(time.time())
    plip_report_file = random.choice(list(plip_report_path.glob("*.txt")))
    dfs = parse_rst_table_file(
        plip_report_file, table_names=["Hydrogen Bonds", "Hydrophobic Interactions"]
    )
    for key, df in dfs.items():
        print(key, df.shape)


def main():
    plip_path = Path(__file__).parent.parent.parent / "data" / "plip"
    plip_report_path = plip_path / "text_reports"
    logger.info(f"Processing PLIP reports in {plip_report_path}")
    interactions = defaultdict(list)
    for report_file in tqdm(plip_report_path.glob("*.txt")):
        dfs = parse_rst_table_file(
            report_file,
            table_names=["Hydrogen Bonds", "Hydrophobic Interactions"],
            max_num_ligands=1,
        )
        if dfs is None:
            continue
        activity_id = int(report_file.stem.split(".")[0])
        for key, df in dfs.items():
            df["activity_id"] = activity_id
            interactions[key].append(df)
    pd.concat(interactions["hydrogen_bonds"], axis=0).to_csv(
        plip_path / "processed" / "hydrogen_bonds.csv", index=False
    )
    pd.concat(interactions["hydrophobic_interactions"], axis=0).to_csv(
        plip_path / "processed" / "hydrophobic_interactions.csv", index=False
    )


if __name__ == "__main__":
    main()
