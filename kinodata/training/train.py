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
from .predict import predict_df


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
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)

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
