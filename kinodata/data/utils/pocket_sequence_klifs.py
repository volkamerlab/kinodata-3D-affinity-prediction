from typing import Iterable, List, Optional, Union
import requests
import json
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import PandasTools as PD
from pathlib import Path


def klifs_structure_ids(cache: Path, raw_data_fp: Path) -> list[str]:
    cache = Path(cache)
    if not cache.exists():
        print("Reading data frame...")
        df = PD.LoadSDF(
            str(raw_data_fp),
            molColName=None,
            smilesName=None,
            embedProps=True,
        )
        #print(df.head())
        df[["similar.klifs_structure_id"]].to_csv(cache)
    return pd.read_csv(cache)["similar.klifs_structure_id"].unique().tolist()


class CachedSequences:
    def __init__(
        self,
        sequence_file: Union[str, Path],
    ) -> None:
        sequence_file = Path(sequence_file)
        self.sequence_file = sequence_file
        #print(sequence_file)

    def __setitem__(self, klifs_id, sequence: Optional[str] = None):
        #print('printing the sequence from the pocket_sequence_klifs')
        print('aaaaa')
        print(str(sequence[0]))
        print(str(sequence))
        print('aaaaa')
        assert isinstance(str(sequence[0]), str)
        #assert isinstance(sequence[0], str) --bnefore changed by me
        if sequence is None:
            sequence = get_pocket_sequence([klifs_id])[0]
        if not hasattr(self, "sequences"):
            raise RuntimeError(
                "Must enter context managed sequence cache before adding sequences."
            )
        self.sequences[klifs_id] = str(sequence)
        #self.sequences[klifs_id] = sequence

    def write_to_cache(self):
        self.to_data_frame().to_csv(self.sequence_file, index=False)

    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "similar.klifs_structure_id": list(self.sequences.keys()),
                "structure.pocket_sequence": list(self.sequences.values()),
            }
        )

    def __contains__(self, klifs_id):
        if not hasattr(self, "sequences"):
            raise RuntimeError("Must enter context managed sequence cache first.")
        return klifs_id in self.sequences

    def __enter__(self):
        self.sequences = dict()
        if self.sequence_file.exists():
            df = pd.read_csv(self.sequence_file)
            #print('printing this from pocket_sequence_klifs')
            #print(df)
            self.sequences = {
                klifs_id: klifs_sequence
                for klifs_id, klifs_sequence in zip(
                    df["similar.klifs_structure_id"], df["structure.pocket_sequence"]
                )
            }
        return self

    def __exit__(self, *args):
        self.write_to_cache()
        print(f"Exiting with {len(self.sequences)} cached sequences.")


def get_pocket_sequence(structure_ids: Iterable[int]) -> List[str]:
    """Retrieve pocket sequences from KLIFS based on structure ID.

    Parameters
    ----------
    structure_ids : Iterable[int]
        Iterable over structure IDs.

    Returns
    -------
    List[str]
        List of corresponding pocket sequences.
    """
    sequences = []
    for structure_id in structure_ids:
        resp = requests.get(
            "https://klifs.net/api_v2/structure_list",
            params={"structure_ID": structure_id},
        )
        resp.raise_for_status()
        pocket_seq = json.loads(resp.content)[0]["pocket"]
        sequences.append(pocket_seq)
    return sequences


def add_pocket_sequence(
    target: pd.DataFrame,
    structure_id_key: str = "similar.klifs_structure_id",
    pocket_sequence_key: str = "structure.pocket_sequence",
) -> pd.DataFrame:
    """Given a dataframe with KLIFS structures as entries,
    add/extend it with an additional column that stores corresponding
    pocket sequences.

    Will query KLIFS based on structure ID.

    Parameters
    ----------
    target : pd.DataFrame
        The target data frame that should be modified.
    structure_id_key : str, optional
        The name of the column that stores structure IDs.
    pocket_sequence_key : str, optional
        The name of the newly created column that will store pocket sequences.

    Returns
    -------
    pd.DataFrame
        New data frame with modifications as descrived above.
    """
    klifs_structure_id = target[structure_id_key].unique()
    seq_raw = get_pocket_sequence(
        tqdm(klifs_structure_id, desc="Retrieving pocket sequences from KLIFS...")
    )
    df_seq = pd.DataFrame(
        {pocket_sequence_key: seq_raw, structure_id_key: klifs_structure_id}
    )
    extended = target.merge(df_seq, how="left", on=structure_id_key)
    return extended
