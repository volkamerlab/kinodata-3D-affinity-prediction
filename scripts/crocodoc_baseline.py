from functools import partial

from torch_geometric.transforms import Compose

from kinodata.configuration import Config
from kinodata.data.data_module import make_kinodata_module
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform import TransformToComplexGraph
from kinodata.transform.baseline import RemoveLigand, RemovePocket
from kinodata.transform.impute_metadata import ImputeMetdata

from inference_utils import load_model_from_checkpoint

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger


def make_config() -> Config:
    config = Config(
        rmsd_threshold=2,
        split_type="scaffold-k-fold",
        split_fold=0,
        model_type="CGNN-3D",
        mask_ligand=False,
        mask_pocket=False,
        remove_new_features=True,
    )
    config.update_from_args()
    return config


if __name__ == "__main__":
    config = make_config()
    model, model_config = config.call(load_model_from_checkpoint)
    pre_transform = [TransformToComplexGraph()]
    model_config.pretty_print()

    if config.mask_ligand and config.mask_pocket:
        raise ValueError("Should not mask ligand and pocket simultaneously")
    if config.mask_ligand:
        pre_transform = pre_transform + [RemoveLigand()]
    if config.mask_pocket:
        pre_transform = pre_transform + [RemovePocket()]
    # pre_transform = pre_transform + [ImputeMetdata()]
    pre_transform = Compose(pre_transform)

    model_config["num_workers"] = 0

    data_module = make_kinodata_module(
        model_config,
        one_time_transform=partial(
            apply_transform_instance_permament,
            transform=pre_transform,
        ),
    )

    logger = WandbLogger()
    trainer = Trainer(logger=logger, auto_select_gpus=True, num_processes=1)
    trainer.test(model, data_module.test_dataloader())
