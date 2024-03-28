from kinodata.data.dataset import process_raw_data, KinodataDocked, Filtered
from kinodata.transform import FilterDockingRMSD
from pathlib import Path


if __name__ == "__main__":
    dataset = KinodataDocked()
    print(dataset[420])
    cls_dataset_2 = Filtered(dataset, FilterDockingRMSD(2.0))
    dataset_2 = cls_dataset_2()
    print(len(dataset_2))
    cls_dataset_4 = Filtered(dataset, FilterDockingRMSD(4.0))
    dataset_4 = cls_dataset_4()
    print(len(dataset_4))
    cls_dataset_6 = Filtered(dataset, FilterDockingRMSD(6.0))
    dataset_6 = cls_dataset_6()
    print(len(dataset_6))
