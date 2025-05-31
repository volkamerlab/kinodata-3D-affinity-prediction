from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")
import wandb
import torch
from torch_geometric.transforms import Compose

import kinodata.configuration as configuration
from kinodata.training import train
from kinodata.model.dimenet import DimeNetWrapper
from kinodata.data.data_module import make_kinodata_module
from kinodata.types import NodeType, RelationType
from kinodata.data.dataset import apply_transform_instance_permament, _DATA
from kinodata.transform.to_complex_graph import TransformToComplexGraph
from kinodata.transform.ligand_only import ToLigandOnlyComplex
from kinodata.data.featurization.atoms import AtomFeatures
from kinodata.data.featurization.bonds import NUM_BOND_TYPES


class FeatureSelection:
    def __init__(self, mask, bond_mask=None):
        self.mask = mask
        self.bond_mask = bond_mask

    def __call__(self, data):
        data[NodeType.Complex].x = data[NodeType.Complex].x[:, self.mask]
        if self.bond_mask is not None:
            data[
                NodeType.Complex, RelationType.Covalent, NodeType.Complex
            ].edge_attr = data[
                NodeType.Complex, RelationType.Covalent, NodeType.Complex
            ].edge_attr[:, self.bond_mask]
        return data


DEBUG = False


def _debug_config(config):
    config["hidden_channels"] = 16
    config["num_attention_blocks"] = 1
    config["num_heads"] = 1
    config["max_num_neighbors"] = 8
    config["epochs"] = 3
    config["crocodoc_frequency"] = 0
    config["run_crocodoc"] = True
    config["store_model_representation"] = True
    config["representation_alias"] = "last"
    return config


if __name__ == "__main__":
    config = configuration.get("data", "training", "dimenet")
    config["mask_pl_edges"] = False
    config["perturb_ligand_positions"] = 0.0
    config["perturb_pocket_positions"] = 0.0
    config["perturb_complex_positions"] = 0.1
    config["node_types"] = [NodeType.Complex]
    config["atom_attr_size"] = AtomFeatures.size

    config["run_crocodoc"] = True
    config["crocodoc_model"] = "best"
    config["mask_type"] = "atom_objects"
    config["crocodoc_start_epoch"] = 0
    config["crocodoc_frequency"] = 0

    config["store_model_representation"] = True
    config["representation_alias"] = "last"

    config["early_stopping"] = False
    config["epochs"] = 200
    config["num_workers"] = 0
    config["simplified_features"] = True
    config["pli_reference_path"] = (
        _DATA / "processed" / "stability_seleciton_pli_reference.csv"
    )

    tags = []
    parser = config.argparser(overwrite_default_values=False)
    args = parser.parse_args()
    config = config.update(vars(args))
    if DEBUG:
        config = _debug_config(config)
        tags.append("debugging")

    if config.get("covalent_only", False):
        config["interaction_modes"] = ["covalent"]
        config["covalent_only"] = True
    else:
        config["interaction_modes"] = ["covalent", "structural"]
        config["covalent_only"] = False

    for key, value in sorted(config.items(), key=lambda i: i[0]):
        print(f"{key}: {value}")

    ott = TransformToComplexGraph(remove_heterogeneous_representation=True)
    if config.get("simplified_features", False):
        print("Using simplified features")
        mask = torch.zeros(AtomFeatures.size, dtype=torch.bool)
        is_hydrogen = list(range(5))
        mask[is_hydrogen] = True  # num hydrogens 0-4
        mask[-1] = True  # gasteiger charge
        bond_mask = torch.zeros(NUM_BOND_TYPES, dtype=torch.bool)
        bond_mask[[0, 1, 2, -1]] = True  # single, double, triple, other
        select_transform = FeatureSelection(mask, bond_mask)
        ott = Compose([ott, select_transform])
        config["atom_attr_size"] = mask.sum().item()
        config["edge_size"] = bond_mask.sum().item()
        tags.append("simplified_features")

    if config.get("ligand_only_3d", False):
        ott = Compose([ott, ToLigandOnlyComplex()])

    wandb.init(
        config=config,
        project="kinodata-docked-rescore",
        tags=tags + ["transformer"] + config.get("tags", []),
        mode=config.get("wandb_mode", "online"),
    )
    train(
        config,
        fn_model=DimeNetWrapper,
        fn_data=partial(
            make_kinodata_module,
            one_time_transform=partial(
                apply_transform_instance_permament,
                transform=ott,
            ),
            subset_data=100 if DEBUG else 0,
        ),
    )
