from typing import List, Optional
from kinodata.data.dataset import KinodataDocked, Filtered
from kinodata.transform.filter_metadata import FilterDockingRMSD
from kinodata.data.grouped_split import KinodataKFoldSplit


class TestingDataset(KinodataDocked):
    def prefix(self) -> Optional[str]:
        return "testing"

    @property
    def raw_file_names(self) -> List[str]:
        return ["test_kinodata.sdf"]


if __name__ == "__main__":
    # shutil.rmtree("data/processed/testing")
    dataset = KinodataDocked()
    print(dataset)
    filter2 = Filtered(dataset, FilterDockingRMSD(2.0))
    filter4 = Filtered(dataset, FilterDockingRMSD(4.0))
    filter6 = Filtered(dataset, FilterDockingRMSD(6.0))

    for filtered in [filter2, filter4, filter6]:
        splitter = KinodataKFoldSplit("scaffold-k-fold", 5)
        splits = splitter.split(filtered)
        for split in splits:
            print(split)

        splitter = KinodataKFoldSplit("pocket-k-fold", 5)
        splits = splitter.split(filtered)
        for split in splits:
            print(split)

        splitter = KinodataKFoldSplit("random-k-fold", 5)
        splits = splitter.split(filtered)
        for split in splits:
            print(split)
