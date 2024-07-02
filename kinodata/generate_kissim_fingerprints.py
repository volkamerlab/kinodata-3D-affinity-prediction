import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# TODO fix environment
import kissim.encoding.fingerprint as fp


def main(
    klifs_path=Path("data") / "unique_klifs_structure_ids.csv",
    raw_kissim_dir=Path("data") / "raw" / "json" / "kissim",
    processed_kissim_dir=Path("data") / "processed" / "kissim",
):
    id_data = pd.read_csv(klifs_path)
    fingerprints = []
    for sid in tqdm(id_data["similar.klifs_structure_id"]):
        fname = raw_kissim_dir / f"kissim_{sid}.json"
        if fname.exists():
            fingerprint = fp.Fingerprint.from_json(fname)
        else:
            try:
                fingerprint = fp.Fingerprint.from_structure_klifs_id(sid)
                fingerprint.to_json(str(fname))
            except Exception as e:
                logging.exception(e)
        if fingerprint is None:
            continue
        fingerprints.append(fingerprint)
    

    fingerprint_data_frames = [
        pd.concat((fp.physicochemical, fp.distances), axis=1) for fp in fingerprints
    ]
    fingerprint_data_frames = [df.fillna(0.0) for df in fingerprint_data_frames]
    for fingerprint, data in zip(fingerprints, fingerprint_data_frames):
        fname = processed_kissim_dir / f"kissim_{fingerprint.structure_klifs_id}.csv"
        print(fname)
        data.to_csv(fname)


if __name__ == "__main__":
    main()
