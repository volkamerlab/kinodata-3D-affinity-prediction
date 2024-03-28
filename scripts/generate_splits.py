from pathlib import Path
from kinodata.data.dataset import KinodataDocked, Filtered, repr_filter
from kinodata.transform import FilterDockingRMSD
from kinodata.data.grouped_split import KinodataKFoldSplit

DATA = Path("data")


def skippable(
    rmsd_cutoff=None,
    split_type="random-k-fold",
):
    rmsd_cutoff = float(rmsd_cutoff)
    split_type = str(split_type)
    p = DATA / "processed" / f"filter_predicted_rmsd_le{rmsd_cutoff:.2f}" / split_type
    if not p.exists():
        return False
    assert p.is_dir()
    return len(list(p.iterdir())) > 0


if __name__ == "__main__":
    dataset = KinodataDocked()
    all = Filtered(dataset, FilterDockingRMSD(100.0))()
    split = KinodataKFoldSplit("pocket-k-fold", 5)
    _ = split.split(all)
    exit()
    split_types = (
        "random-k-fold",
        "scaffold-k-fold",
        "pocket-k-fold",
    )
    for split_type in split_types:
        print(f"\tSplitting full dataset based on type {split_type}...")
        splitter = KinodataKFoldSplit(split_type, 5)
        splits = splitter.split(dataset)

    for rmsd_cutoff in (2.0, 4.0, 6.0):
        if all(skippable(rmsd_cutoff, split_type) for split_type in split_types):
            print(f"Skipping {rmsd_cutoff}")
            continue
        print(f"Loading dataset for RMSD <= {rmsd_cutoff}")
        sub_dataset = Filtered(dataset, FilterDockingRMSD(rmsd_cutoff))()
        for split_type in split_types:
            print(f"\tSplitting data based on type {split_type}...")
            splitter = KinodataKFoldSplit(split_type, 5)
            splits = splitter.split(sub_dataset)
