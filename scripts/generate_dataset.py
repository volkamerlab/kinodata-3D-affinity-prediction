from kinodata.data.dataset import process_raw_data, KinodataDocked, Filtered
from kinodata.transform import FilterDockingRMSD
from pathlib import Path


if __name__ == "__main__":
    full_dataset = KinodataDocked()
    for rmsd_cutoff in [2.0, 4.0, 6.0, 100.0]:
        cls_dataset = Filtered(full_dataset, FilterDockingRMSD(rmsd_cutoff))
        dataset = cls_dataset()
        print(f"RMSD cutoff: {rmsd_cutoff} -- Dataset size: {len(dataset)}")
