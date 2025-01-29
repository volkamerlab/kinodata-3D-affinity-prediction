import numpy as np
from ..model.regression import RegressionModel, cat_many
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
import pandas as pd


def predict_df(
    model: RegressionModel,
    loader: DataLoader,
    trainer: Trainer | None = None,
    ckpt_path: str | None = "best",
) -> pd.DataFrame:
    if trainer is None:
        trainer = Trainer()
    dict_list = trainer.predict(model, loader, ckpt_path=ckpt_path)
    return pd.DataFrame(
        {key: np.array(value) for key, value in cat_many(dict_list).items()}
    )
