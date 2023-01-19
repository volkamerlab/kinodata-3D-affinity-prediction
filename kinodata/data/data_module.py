import wandb

from typing import Optional
import torch
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch.utils.data import random_split
import wandb


def make_data_module(
    dataset,
    batch_size: int,
    num_workers: int,
    train_size: float = 1.0,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
    seed: int = 0,
    log_seed: bool = False,
) -> LightningDataset:
    if log_seed:
        wandb.config.data_seed = seed
    split = dict()
    split["train_dataset"] = int(train_size * len(dataset))
    if val_size is not None:
        split["val_dataset"] = int(val_size * len(dataset))
    if test_size is not None:
        split["test_dataset"] = int(test_size * len(dataset))
    split["train_dataset"] += (len(dataset) - sum(split.values()))
    
    kwargs = dict(zip(
        split.keys(), 
        random_split(
            dataset,
            list(split.values()), 
            generator=torch.Generator().manual_seed(seed)
        )
    ))
    kwargs["batch_size"] = batch_size
    kwargs["num_workers"] = num_workers
    
    return LightningDataset(**kwargs)