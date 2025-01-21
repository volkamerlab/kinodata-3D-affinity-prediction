from argparse import ArgumentParser
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import os
import torch
import wandb
from docktgrid.molecule import MolecularComplex, MolecularData
from docktgrid.transforms import RandomRotation
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import GroupKFold
from torch.nn import BatchNorm3d, Conv3d, Identity, MaxPool3d, ReLU, Sequential
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from torchmetrics.regression import PearsonCorrCoef
from tqdm import tqdm

from kinodata.data.data_split import Split
from kinodata.data.grouped_split import _generator_to_list
from kinodata.data.voxel.dataset import make_voxel_dataset_split

torch.multiprocessing.set_start_method("spawn")
DATA_DIR = Path(__file__).parent.parent / "data"


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

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        lr: float = 1e-4,
        lr_decay: float = 1e-5,
    ):
        super().__init__()
        self.corr_metrics = {key: PearsonCorrCoef() for key in ["train", "val", "test"]}
        self.save_hyperparameters()

        def block(din, dout, kernel_size, stride, padding, act=True, norm=True):
            return Sequential(
                Conv3d(din, dout, kernel_size, stride, padding),
                ReLU() if act else Identity(),
                BatchNorm3d(dout) if norm else Identity(),
            )

        nhid = self.hparams.hidden_channels
        self.cnn_model = Sequential(
            block(in_channels, nhid, 1, 1, 0),
            block(nhid, nhid, 3, 1, 1),
            MaxPool3d(2),
            block(nhid, nhid * 2, 3, 1, 1),
            block(nhid * 2, nhid * 2, 3, 1, 1),
            MaxPool3d(2),
            block(nhid * 2, nhid * 4, 3, 1, 1),
            block(nhid * 4, nhid * 4, 3, 1, 1),
            MaxPool3d(2),
            block(nhid * 4, nhid * 2, 3, 1, 1),
            block(nhid * 2, 1, 1, 1, 0, act=False, norm=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.cnn_model(x)
        pred = z.view(z.size(0), -1).sum(1)
        return pred

    def configure_optimizers(self):
        return AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.lr_decay
        )

    def _step(self, batch, key, *args, **kwargs):
        x = batch[0]
        y = batch[1]
        pred = self(x)
        loss = (pred.flatten() - y.flatten()).pow(2).mean()
        mae = (pred.flatten() - y.flatten()).abs().mean()
        self.log(f"{key}/loss", loss)
        self.log(f"{key}/mae", mae)
        self.corr_metrics[key](pred.flatten().detach().cpu(), y.flatten().cpu())
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
    batch_size: int = 64,
    split_type: str = "scaffold-k-fold",
    seed: int = 0,
    fold: int = 0,
    wandb_mode: str = "online",
    hidden_channels: int = 32,
    lr: float = 3e-4,
    lr_decay: float = 2e-5,
    random_rotation_augmentations: bool = False,
    data_sample: int = 0,
    compile_model: bool = True,
    num_workers: int = os.cpu_count(),
):
    wandb.init(
        project="kinodata-voxel",
        mode=wandb_mode,
        config=dict(
            batch_size=batch_size,
            split_type=split_type,
            seed=seed,
            fold=fold,
        ),
    )

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

    if data_sample > 0:
        df_split = df_split.sample(data_sample, random_state=seed)

    train_split = df_split[df_split["split"] == "train"]["activity_id"].values
    val_split = df_split[df_split["split"] == "val"]["activity_id"].values
    test_split = df_split[df_split["split"] == "test"]["activity_id"].values

    train_transform = None
    inference_transform = None
    if random_rotation_augmentations:
        train_transform = RandomRotation()

    train_data, val_data, test_data = make_voxel_dataset_split(
        DATA_DIR,
        train_split,
        val_split,
        test_split,
        train_transform=train_transform,
        inference_transform=inference_transform,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    _ = train_data[42]
    batch = next(iter(train_loader))
    _, in_channels, grid_size, _, _ = batch[0].size()

    class DataModule(LightningDataModule):

        def train_dataloader(self):
            return DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

        def val_dataloader(self):
            return DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        def test_dataloader(self):
            return DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

    model = VoxelModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        lr=lr,
        lr_decay=lr_decay,
    )
    if compile_model:
        model = torch.compile(model)
    logger = WandbLogger(log_model=True)
    callbacks = [
        ModelCheckpoint(monitor="val/loss"),
        EarlyStopping(monitor="val/mae", min_delta=1e-2, patience=5),
    ]
    trainer = Trainer(
        max_epochs=100, accelerator="cpu", logger=logger, callbacks=callbacks
    )
    data_module = DataModule()
    trainer.fit(model, data_module)
    trainer.test(model, data_module, ckpt_path="best")


from inspect import Parameter, signature

train_sig = signature(train)
parser = ArgumentParser()
for name, arg in train_sig.parameters.items():
    if arg.annotation != Parameter.empty:
        arg_type = arg.annotation
    elif arg.default == None:
        arg_type = str
    elif arg.default != Parameter.empty:
        arg_type = type(arg.default)
    else:
        arg_type = str
    parser_arg = parser.add_argument(
        f"--{arg.name}",
        type=arg_type,
        required=arg.default == Parameter.empty,
        default=arg.default if arg.default != Parameter.empty else None,
    )


if __name__ == "__main__":
    parser.print_usage()
    print("DATA_DIR", DATA_DIR)

    print("Parsing args...")
    args = parser.parse_args()

    print("Binding args to train signature...")
    bound_args = train_sig.bind(**vars(args))

    print("Training...")
    train(*bound_args.args, **bound_args.kwargs)
