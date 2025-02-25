import torch

from kinodata.training.predict import predict_df


from argparse import ArgumentParser
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
from docktgrid.molecule import MolecularComplex, MolecularData
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from kinodata.data.grouped_split import _generator_to_list
from kinodata.data.voxel.dataset import make_voxel_dataset_split
from kinodata.transform.voxel import PerturbPosition, RandomRotation
from kinodata.model.voxel import PafnucyIshVoxelModel

DATA_DIR = Path(__file__).parent.parent / "data"

torch.multiprocessing.set_start_method("spawn", force=True)


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


def get_data_split_activity_ids(
    split_type: str,
    seed: int,
    fold: int,
    data_sample: int = 0,
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

    if data_sample > 0:
        df_split = df_split.sample(data_sample, random_state=seed)

    train_split = df_split[df_split["split"] == "train"]["activity_id"].values
    val_split = df_split[df_split["split"] == "val"]["activity_id"].values
    test_split = df_split[df_split["split"] == "test"]["activity_id"].values
    return train_split, val_split, test_split


def train(
    batch_size: int = 32,
    split_type: str = "scaffold-k-fold",
    seed: int = 0,
    fold: int = 0,
    wandb_mode: str = "online",
    hidden_channels: int = 32,
    dense_channels: int = 512,
    kernel_sizes: str = "4333",
    pooling_type: str = "max",
    lr: float = 1e-4,
    lr_decay: float = 1e-3,
    random_rotation_augmentations: bool = True,
    perturb_complex_positions: float = 0.08,
    data_sample: int = 0,
    compile_model: bool = False,
    num_workers: int = 0,
    use_val_for_testing: bool = False,
    accelerator: str = "auto",
    min_epochs: int = 100,
    max_epochs: int = 500,
    acc_grad_batches: int = 1,
):
    kernel_sizes = [int(k) for k in kernel_sizes]

    wandb.init(
        project="kinodata-voxel",
        mode=wandb_mode,
        config=dict(
            batch_size=batch_size,
            effective_batch_size=batch_size * acc_grad_batches,
            split_type=split_type,
            seed=seed,
            fold=fold,
            hidden_channels=hidden_channels,
            dense_channels=dense_channels,
            kernel_sizes=kernel_sizes,
            pooling_type=pooling_type,
            lr=lr,
            lr_decay=lr_decay,
            random_rotation_augmentations=random_rotation_augmentations,
            perturb_complex_positions=perturb_complex_positions,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
        ),
    )

    train_transform = []
    inference_transform = None
    if random_rotation_augmentations:
        train_transform.append(RandomRotation())
    if perturb_complex_positions > 0:
        train_transform.append(PerturbPosition(std=perturb_complex_positions))

    train_split, val_split, test_split = get_data_split_activity_ids(
        split_type,
        seed,
        fold,
        data_sample=data_sample,
    )

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

        def _dataloader(self, data, **kwargs):
            if "persistent_workers" not in kwargs:
                kwargs["persistent_workers"] = num_workers > 0
            return DataLoader(
                data,
                batch_size=batch_size,
                num_workers=num_workers,
                **kwargs,
            )

        def train_dataloader(self):
            return self._dataloader(train_data, shuffle=True)

        def val_dataloader(self):
            return self._dataloader(val_data, shuffle=False)

        def test_dataloader(self):
            if use_val_for_testing:
                return self._dataloader(val_data, shuffle=False)
            return self._dataloader(test_data, shuffle=False)

    model = PafnucyIshVoxelModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dense_channels=dense_channels,
        kernel_sizes=kernel_sizes,
        lr=lr,
        lr_decay=lr_decay,
        pooling_type=pooling_type,
    )
    print(model)
    if compile_model:
        model = torch.compile(model)
    logger = WandbLogger(log_model=True)
    callbacks = [
        ModelCheckpoint(monitor="val/loss"),
        EarlyStopping(monitor="val/corr", min_delta=5e-3, patience=15, mode="max"),
    ]
    trainer = Trainer(
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=acc_grad_batches,
        gradient_clip_val=10.0,
    )
    data_module = DataModule()
    trainer.fit(model, data_module)
    trainer.test(model, data_module.val_dataloader(), ckpt_path="best")

    # log all predictions of best model
    df_train = predict_df(model, data_module.train_dataloader(), trainer, "best")
    df_val = predict_df(model, data_module.val_dataloader(), trainer, "best")
    df_test = predict_df(model, data_module.test_dataloader(), trainer, "best")
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_val, df_test])
    table = wandb.Table(dataframe=df)
    wandb.log({"all_predictions": table})


from inspect import Parameter, signature  # noqa: E402

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
