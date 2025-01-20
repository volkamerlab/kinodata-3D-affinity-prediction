from itertools import count
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import Conv3d, MaxPool3d, ReLU, Sequential, BatchNorm3d, Identity
from torch.optim import Adam
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from kinodata.data.voxel.dataset import make_voxel_dataset_split
from kinodata.data.grouped_split import _generator_to_list
from kinodata.data.data_split import Split
from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.regression import PearsonCorrCoef
from docktgrid.molecule import MolecularData, MolecularComplex


DATA_DIR = Path(__file__).parent.parent / "data"

_DEBUG_IDS = [
    "2043754",
    "22798937",
    "1491221",
    "17626997",
    "2547512",
    "17886447",
    "2564660",
    "18420015",
    "18918195",
    "17772440",
    "3294285",
    "24789104",
    "17633369",
    "17676447",
    "18673767",
    "18287192",
    "18861740",
    "3440562",
    "17612316",
    "24769569",
    "17739970",
    "20640644",
    "16498887",
    "15765340",
    "17682649",
    "17642881",
    "1759182",
    "17625721",
    "16506353",
    "2239320",
    "18939646",
    "17601869",
    "7956440",
    "17680676",
    "12631725",
    "17776709",
    "22955424",
    "17763327",
    "23152101",
    "7991181",
    "17625120",
    "17723116",
    "17767123",
    "12046199",
    "1077022",
    "17638429",
    "22802112",
    "17749761",
    "11018075",
    "2082093",
    "16371338",
    "12195443",
    "16739505",
    "2465597",
    "17727939",
    "17704488",
    "7578026",
    "17623089",
    "6280872",
    "17687221",
    "18439835",
    "17641042",
    "17663155",
    "17740461",
    "15651885",
    "22435303",
    "16871453",
    "16883355",
    "17799330",
    "17625677",
    "17613833",
    "6299710",
    "22960812",
    "2397038",
    "19432498",
    "12099345",
    "16336960",
    "16760025",
    "3203149",
    "16906654",
    "19245602",
    "16566146",
    "18402842",
    "17738760",
    "2901443",
    "18859714",
    "17737491",
    "3442995",
    "17625317",
    "2260964",
    "17614719",
    "10975117",
    "17693641",
    "23288957",
    "16746178",
    "12175199",
    "16309562",
    "16895201",
    "5246429",
    "17698102",
]


def _collect_vdw_key_errors(train_data, val_data, test_data):
    key_errors = []
    aids = []
    for data in (train_data, val_data, test_data):
        for index, (ligand_data, protein_data) in tqdm(
            enumerate(zip(data.lig_files, data.ptn_files))
        ):
            assert isinstance(ligand_data, MolecularData)
            assert isinstance(protein_data, MolecularData)
            activity_id = data.metadata[index]
            try:
                cplx = MolecularComplex(ligand_data, protein_data)
            except KeyError as e:
                key_errors.append(str(e))
                aids.append(activity_id)

    pd.DataFrame({"key_errors": key_errors, "activity_id": aids}).to_csv(
        "docktgrid_vdw_key_errors.csv"
    )


class VoxelModel(LightningModule):

    def __init__(self, in_channels: int, hidden_size: int = 32):
        super().__init__()
        self.corr_metrics = {key: PearsonCorrCoef() for key in ["train", "val", "test"]}
        self.save_hyperparameters()

        def block(din, dout, kernel_size, stride, padding, act=True, norm=True):
            return Sequential(
                Conv3d(din, dout, kernel_size, stride, padding),
                ReLU() if act else Identity(),
                BatchNorm3d(dout) if norm else Identity(),
            )

        self.cnn_model = Sequential(
            block(in_channels, hidden_size, 1, 1, 0),
            block(hidden_size, hidden_size, 3, 1, 1),
            MaxPool3d(2),
            block(hidden_size, hidden_size * 2, 3, 1, 1),
            block(hidden_size * 2, hidden_size * 2, 3, 1, 1),
            MaxPool3d(2),
            block(hidden_size * 2, hidden_size * 4, 3, 1, 1),
            block(hidden_size * 4, hidden_size * 4, 3, 1, 1),
            MaxPool3d(2),
            block(hidden_size * 4, hidden_size * 2, 3, 1, 1),
            block(hidden_size * 2, 1, 1, 1, 0, act=False, norm=False),
        )

    def forward(self, x: torch.Tensor):
        z: torch.Tensor = self.cnn_model(x)
        pred = z.view(z.size(0), -1).sum(1)
        return pred

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def _step(self, batch, key, *args, **kwargs):
        x = batch[0]
        y = batch[1]
        pred = self(x)
        loss = (pred.flatten() - y.flatten()).pow(2).mean()
        mae = (pred.flatten() - y.flatten()).abs().mean()
        self.log(f"{key}/loss", loss)
        self.log(f"{key}/mae", mae)
        self.corr_metrics[key](pred.flatten(), y.flatten())
        return x, y, pred, loss

    def _shared_on_epoch_end(self, key):
        self.log(f"{key}/corr", self.corr_metrics[key].compute())
        self.corr_metrics[key].reset()

    def on_train_epoch_end(self):
        return self._shared_on_epoch_end("train")

    def on_validation_epoch_end(self):
        return self._shared_on_epoch_end("val")

    def on_test_epoch_end(self):
        return self._shared_on_epoch_end("test")

    def training_step(self, batch, *args, **kwargs):
        x, y, pred, loss = self._step(batch, "train", *args, **kwargs)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y, pred, loss = self._step(batch, "val", *args, **kwargs)
        return loss

    def test_step(self, batch, *args, **kwargs):
        x, y, pred, loss = self._step(batch, "test", *args, **kwargs)
        return loss


def make_k_fold_split(
    groups: np.ndarray,
    activity_ids: np.ndarray,
    seed: int = 0,
    cache: Path = None,
    force_recomputation: bool = False,
):
    if np.any([group is None for group in groups]):
        raise ValueError("Groups must not contain NaNs.")
    if np.any(np.isnan(activity_ids)):
        raise ValueError("Activity IDs must not contain NaNs.")
    if cache is not None:
        assert cache.suffix == ".csv"
    if cache is not None and not cache.parent.exists():
        cache.parent.mkdir(parents=True)
    if not force_recomputation and cache is not None and cache.exists():
        k_fold_split = pd.read_csv(cache)
        return k_fold_split

    # make sure groups are int labels
    group_to_label = dict()
    label_counter = count()
    for group in groups:
        if group not in group_to_label:
            group_to_label[group] = next(label_counter)
    groups = np.array([group_to_label[group] for group in groups])

    group_k_fold = GroupKFold(5)
    index_generator = group_k_fold.split(
        np.zeros((groups.shape[0], 1)),
        groups=groups,
    )
    split_ids = np.array([], dtype=activity_ids.dtype)
    splits = []
    folds = []
    for fold_id, split in enumerate(_generator_to_list(index_generator)):
        assert (
            len(split.train_split) + len(split.val_split) + len(split.test_split)
            == groups.shape[0]
        )
        split_ids = np.append(split_ids, activity_ids[split.train_split])
        splits.extend(["train"] * len(split.train_split))

        split_ids = np.append(split_ids, activity_ids[split.val_split])
        splits.extend(["val"] * len(split.val_split))

        split_ids = np.append(split_ids, activity_ids[split.test_split])
        splits.extend(["test"] * len(split.test_split))

        folds.extend([fold_id] * groups.shape[0])

    df = pd.DataFrame(
        {
            "activity_id": split_ids,
            "split": splits,
            "fold": folds,
        }
    )
    if cache is not None:
        df.to_csv(cache, index=False)
    return df


def train(
    batch_size: int = 1,
    split_type: Literal[
        "scaffold-k-fold", "random-k-fold", "pocket-k-fold"
    ] = "scaffold-k-fold",
    seed: int = 0,
    fold: int = 0,
):
    kinodata3d_df = pd.read_csv(DATA_DIR / "docktgrid" / "kinodata3d.csv")
    df_split = None
    if split_type == "scaffold-k-fold":
        df_split = make_k_fold_split(
            kinodata3d_df["scaffold"].values,
            kinodata3d_df["activities.activity_id"].values,
            seed=seed,
            cache=DATA_DIR / "processed" / "chembl_splits" / "scaffold_k_fold.csv",
        )
    elif split_type == "random-k-fold":
        raise NotImplementedError("Random k-fold not implemented.")
    elif split_type == "pocket-k-fold":
        raise NotImplementedError("Pocket k-fold not implemented.")
    else:
        raise ValueError(f"Unknown split type {split_type}")
    assert isinstance(df_split, pd.DataFrame)
    df_split = df_split[df_split["fold"] == fold]
    train_split = df_split[df_split["split"] == "train"]["activity_id"].values
    val_split = df_split[df_split["split"] == "val"]["activity_id"].values
    test_split = df_split[df_split["split"] == "test"]["activity_id"].values

    train_data, val_data, test_data = make_voxel_dataset_split(
        DATA_DIR, train_split, val_split, test_split
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    _ = train_data[42]
    batch = next(iter(train_loader))
    _, in_channels, grid_size, _, _ = batch[0].size()

    class DataModule(LightningDataModule):

        def train_dataloader(self):
            return DataLoader(train_data, batch_size=batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(val_data, batch_size=batch_size, shuffle=False)

        def test_dataloader(self):
            return DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = VoxelModel(in_channels=in_channels)
    pred = model(batch[0])
    print(pred.size())

    logger = CSVLogger("logs", name="voxel_model")

    callbacks = [
        ModelCheckpoint(monitor="val/loss"),
        EarlyStopping(monitor="val/mae", min_delta=1e-2, patience=5),
    ]

    trainer = Trainer(max_epochs=10, accelerator="auto", logger=logger)

    trainer.fit(model, DataModule())
    trainer.test(model, DataModule(), ckpt_path="best")


if __name__ == "__main__":
    print("DATA_DIR", DATA_DIR)
    train(32)
