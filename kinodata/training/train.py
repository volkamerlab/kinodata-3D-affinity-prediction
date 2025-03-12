import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from kinodata.data.data_module import make_kinodata_module
import kinodata.transform as T
from .predict import predict_df
from .crocodoc import crocodoc_cgnn


def _remove_augementation_transforms_from_dataset(dataset):
    if isinstance(dataset.transform, T.Compose):
        dataset.transform = T.Compose(
            [
                t
                for t in dataset.transform.transforms
                if not isinstance(t, T.PerturbAtomPositions)
            ]
        )
    elif isinstance(dataset.transform, T.PerturbAtomPositions):
        dataset.transform = None
    return dataset


def _remove_augmentation_transforms_from_data_module(data_module):
    data_module.train_dataset = _remove_augementation_transforms_from_dataset(
        data_module.train_dataset
    )
    data_module.val_dataset = _remove_augementation_transforms_from_dataset(
        data_module.val_dataset
    )
    data_module.test_dataset = _remove_augementation_transforms_from_dataset(
        data_module.test_dataset
    )
    return data_module


def train(config, fn_data=make_kinodata_module, fn_model=None):
    logger = WandbLogger(log_model="all")
    model = fn_model(config)
    data_module = fn_data(config)
    validation_checkpoint = ModelCheckpoint(
        monitor="val/mae",
        mode="min",
    )
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/mae", patience=config.early_stopping_patience, mode="min"
    )

    trainer = pl.Trainer(
        logger=logger,
        devices="auto",
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[validation_checkpoint, lr_monitor, early_stopping],
        gradient_clip_val=config.clip_grad_value,
        gradient_clip_algorithm="norm" if config.clip_grad_value else None,
        overfit_batches=config.get("overfit_batches", 0),
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)

    data_module = _remove_augmentation_transforms_from_data_module(data_module)

    # log crocodoc results if enabled
    if config.get("run_crocodoc", False):
        mask_type = config.get("mask_type", None)
        crocodoc_train = crocodoc_cgnn(
            model,
            data_module.test_dataset,
            trainer,
            mask_type=mask_type,
            ckpt_path=config.get("crocodoc_model", None),
        )
        crocodoc_train["split"] = "train"
        crocodoc_test = crocodoc_cgnn(
            model,
            data_module.test_dataset,
            trainer,
            mask_type=mask_type,
            ckpt_path=config.get("crocodoc_model", None),
        )
        crocodoc_test["split"] = "test"
        crocodoc_table = wandb.Table(
            dataframe=pd.concat([crocodoc_train, crocodoc_test])
        )
        wandb.log({"crocodoc_results": crocodoc_table})

    # log all predictions of the best model
    df_train = predict_df(model, data_module.train_dataloader(), trainer, "best")
    df_val = predict_df(model, data_module.val_dataloader(), trainer, "best")
    df_test = predict_df(model, data_module.test_dataloader(), trainer, "best")
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_val, df_test])
    table = wandb.Table(dataframe=df)
    wandb.log({"all_predictions": table})
