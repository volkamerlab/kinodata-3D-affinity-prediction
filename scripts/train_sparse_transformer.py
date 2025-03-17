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
from kinodata.data.data_module import make_kinodata_module
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.types import NodeType, RelationType
from kinodata.data.dataset import apply_transform_instance_permament
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
if __name__ == "__main__":
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
        covalent_only=False,
        ablate_binding_features=False,
    )
    config = configuration.get("data", "training", "sparse_transformer")
    config["mask_pl_edges"] = False
    config["perturb_ligand_positions"] = 0.0
    config["perturb_pocket_positions"] = 0.0
    config["perturb_complex_positions"] = 0.1
    config["node_types"] = [NodeType.Complex]
    config["atom_attr_size"] = AtomFeatures.size
    config["run_crocodoc"] = True
    config["crocodoc_model"] = "best"
    config["mask_type"] = "atom_objects"
    config["num_workers"] = 0
    config["simplified_features"] = True

    if DEBUG:
        config["hidden_channels"] = 16
        config["num_attention_blocks"] = 1
        config["num_heads"] = 1
        config["max_num_neighbors"] = 8
        config["overfit_batches"] = 0.1
        config["epochs"] = 1

    tags = []
    parser = config.argparser(overwrite_default_values=False)
    args = parser.parse_args()
    print(args)
    config = config.update(vars(args))

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
        fn_model=make_model,
        fn_data=partial(
            make_kinodata_module,
            one_time_transform=partial(
                apply_transform_instance_permament,
                transform=ott,
            ),
        ),
    )
