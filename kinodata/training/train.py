import os.path as osp
from typing import Callable
import json

import pandas as pd
import torch

try:
    import pathlib

    torch.serialization.add_safe_globals(
        [pathlib.PosixPath, pathlib.WindowsPath, pathlib.Path]
    )
    if hasattr(pathlib, "_local"):
        torch.serialization.add_safe_globals(
            [pathlib._local.PosixPath, pathlib._local.WindowsPath]
        )
except (ImportError, AttributeError):
    pass
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch_geometric.data.lightning import LightningDataset

import kinodata.transform as T
import wandb
from kinodata.configuration import Config
from kinodata.data.data_module import make_kinodata_module
from kinodata.model.regression import RegressionModel, enable_target_normalization

from ..evaluation.predict import predict_df
from .callbacks.crocodoc import (
    CrocodocCallback,
    load_pli_reference,
    remove_augmentation_transforms_from_data_module,
)
from .callbacks.integrated_gradients import IntegratedGradientsCallback
from .callbacks.representation import StoreModelRepresentation


def log_large_table(df, name):
    artifact = wandb.Artifact(name, type="large_table")
    file_name = osp.join(wandb.run.dir, f"{name}.csv.gz")
    print(f"Saving large table to {file_name}")
    df.to_csv(file_name, index=False, compression="infer")
    artifact.add_file(file_name)
    wandb.log_artifact(artifact)


def train(
    config: Config,
    fn_data: Callable[[Config], LightningDataset] = make_kinodata_module,
    fn_model: Callable[[Config], RegressionModel] = None,
):
    logger = WandbLogger(log_model=True)
    model = fn_model(config)
    data_module = fn_data(config)
    if config.get("normalize_target", False):
        model = enable_target_normalization(model)
        train_dataset = model.fit_normalize_target(data_module.train_dataset)
        data_module.train_dataset = train_dataset

    validation_checkpoint = ModelCheckpoint(
        monitor="val/mae",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor("epoch")
    callbacks = [validation_checkpoint, lr_monitor]
    if config.get("early_stopping", False):
        early_stopping = EarlyStopping(
            monitor=config.get("early_stopping_metric", "val/mae"),
            patience=config.early_stopping_patience,
            mode=config.get("early_stopping_mode", "min"),
        )
        callbacks.append(early_stopping)
    if config.get("store_model_representation", False):
        repr_callback = StoreModelRepresentation(
            datasets={
                "train": data_module.train_dataset,
                "val": data_module.val_dataset,
                "test": data_module.test_dataset,
            },
            batch_size=config.get("batch_size", 16)
            * 2,  # x 2 should be fine since no backpropagation is required,
            alias=config.get("representation_alias", None),
        )
        callbacks.append(repr_callback)
    if config.get("run_crocodoc", False) and config.get(
        "run_integrated_gradients", False
    ):
        raise ValueError(
            "Cannot run Crocodoc and Integrated Gradients at the same time. "
            "Please set one of them to False."
        )
    if config.get("run_crocodoc", False):
        crocodoc_callback = CrocodocCallback(
            datasets={
                # "train": data_module.train_dataset,
                "val": data_module.val_dataset,
                "test": data_module.test_dataset,
            },
            mask_type=config.get("mask_type", None),
            start_epoch=config.get("crocodoc_start_epoch", 0),
            frequency=config.get("crocodoc_frequency", 25),
            pli_reference=load_pli_reference(
                config.get("pli_reference_path", None),
            ),
        )
        callbacks.append(crocodoc_callback)
    if config.get("run_integrated_gradients", False):
        ig_callback = IntegratedGradientsCallback(
            datasets={
                # "train": data_module.train_dataset,
                "val": data_module.val_dataset,
                "test": data_module.test_dataset,
            },
            start_epoch=config.get("ig_start_epoch", 0),
            frequency=config.get("ig_frequency", 0),
        )
        callbacks.append(ig_callback)

    trainer = pl.Trainer(
        logger=logger,
        devices="auto",
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        gradient_clip_val=config.clip_grad_value,
        gradient_clip_algorithm="norm" if config.clip_grad_value else None,
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)

    data_module = remove_augmentation_transforms_from_data_module(data_module)
    transforms_during_prediction = [None]
    if config.get("also_predict_with_protein_masked", True):
        transforms_during_prediction.append(T.MaskProtein())
    predict_dfs = []
    for transform in transforms_during_prediction:
        for split_key, loader_method in {
            "train": data_module.train_dataloader,
            "val": data_module.val_dataloader,
            "test": data_module.test_dataloader,
        }.items():
            loader = loader_method()
            df_ = predict_df(
                model, loader, trainer, "best", additional_transform=transform
            )
            df_["split"] = split_key
            df_["transform"] = str(transform) if transform else "none"
            predict_dfs.append(df_)

    prediction_df: pd.DataFrame = pd.concat(predict_dfs)
    if (prediction_path := config.get("store_predictions_locally", None)) is not None:
        prediction_df.to_csv(prediction_path)
    if (model_path := config.get("store_model_locally", None)) is not None:
        torch.save(model.state_dict(), model_path)
    if (config_path := config.get("store_config_locally", None)) is not None:
        with open(config_path, "w") as json_file:
            json.dump(config, json_file, default=str)
    table = wandb.Table(dataframe=prediction_df)
    wandb.log({"all_predictions": table})
