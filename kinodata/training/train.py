#%%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from kinodata.data.data_module import make_kinodata_module


import json
from pathlib import Path
from typing import Any
from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")


import torch

import kinodata.configuration as cfg
from kinodata.model import ComplexTransformer, DTIModel, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph

import kinodata.configuration as configuration
#from kinodata.training import train
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.types import NodeType
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform.to_complex_graph import TransformToComplexGraph



#%%
# Log in to your W&B account

import wandb
#wandb.login() #pass key and host?
wandb.init(project='extended_kinodata')

#%%
torch.cuda.is_available()

#%%

data_module = make_kinodata_module(
    cfg.get("data", "training").update(
        dict(

            batch_size=32,
            split_type="scaffold-k-fold",
            filter_rmsd_max_value=4.0,
            split_index=0,
        )
    ),
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)


#%%
data_module[1]


#%%

#trying to treat everything as a large dataset

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        """
        dataset1: Dataset A (Activity dataset)
        dataset2: Dataset B (Pose dataset)
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.min_len = min(self.len1, self.len2)

    def __len__(self):
        return self.min_len  # Adjust this if you want equal-size batches or oversampling

    def __getitem__(self, idx):
        """
        Returns a combined batch containing activity and pose data
        """
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        return {'activity_batch': item1, 'pose_batch': item2}


def custom_collate(batch):
    """
    Custom collate function to handle activity and pose batches separately.
    """
    activity_batches = [item['activity_batch'] for item in batch]
    pose_batches = [item['pose_batch'] for item in batch]

    # Create PyTorch Geometric batches from activity and pose data
    activity_batch = Batch.from_data_list(activity_batches)
    pose_batch = Batch.from_data_list(pose_batches)

    return activity_batch, pose_batch


class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, datamodule1, datamodule2, batch_size=32, num_workers=0):
        """
        CombinedDataModule for handling two datasets (activity and pose).
        
        datamodule1: DataModule for Activity dataset (Dataset A)
        datamodule2: DataModule for Pose dataset (Dataset B)
        """
        super().__init__()
        self.datamodule1 = datamodule1
        self.datamodule2 = datamodule2
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Setup function to combine training, validation, and test datasets.
        """
        # Combine training datasets
        self.train_dataset = CombinedDataset(self.datamodule1.train_dataset, self.datamodule2.train_dataset)

        # Combine validation datasets
        self.val_dataset = CombinedDataset(self.datamodule1.val_dataset, self.datamodule2.val_dataset)

        # Combine test datasets
        self.test_dataset = CombinedDataset(self.datamodule1.test_dataset, self.datamodule2.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False, 
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False, 
            collate_fn=custom_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False, 
            collate_fn=custom_collate
        )

combined_data_module = CombinedDataModule(data_module[0], data_module[1], batch_size=16)



def train(config, fn_data, fn_model=None):
    logger = WandbLogger(project="kinodata_extended", log_model="all")
    model = fn_model(config)
    data_module = fn_data

    print(data_module)
    #fn_data(config)
    #data_module=data_module
    validation_checkpoint = ModelCheckpoint(
        monitor="val/mae",
        #monitor="train/loss_activity",
        mode="min",
    )
    #print(data_module)
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/mae", 
        #monitor="train/loss_activity", 
    
        patience=config.early_stopping_patience, mode="min"
    )

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
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
    #trainer.test(ckpt_path="best", datamodule=data_module)




configuration.register(
        "sparse_transformer",
        max_num_neighbors=16,
        hidden_channels=256,
        num_attention_blocks=3,
        num_heads=8,
        act="silu",
        edge_attr_size=4,
        ln1=True,
        ln2=True,
        ln3=True,
        graph_norm=False,
        interaction_modes=["covalent", "structural"],
        #optim='adam',
    )


config = configuration.get("data", "training", "sparse_transformer")
#print(config)

config = config.update_from_file()

   # config = config.update_from_args()


config["need_distances"] = False


config["perturb_ligand_positions"] = 0.0
config["perturb_pocket_positions"] = 0.0
config["perturb_complex_positions"] = 0.1


config["node_types"] = [NodeType.Complex]


#config.register(optimizer="adam")

#%%
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

# %%
train(
        config,
        fn_model=make_model,
        fn_data=combined_data_module
        #partial(
            #make_kinodata_module,
            #data_module,
            #one_time_transform=partial(
            #    apply_transform_instance_permament,
            #    transform=TransformToComplexGraph(
            #        remove_heterogeneous_representation=True
            #    ),
            #),
        #),
    )


#%%
