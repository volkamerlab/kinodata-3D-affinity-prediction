from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")
import wandb
from torch_geometric.transforms import Compose

import kinodata.configuration as configuration
from kinodata.training import train
from kinodata.data.data_module import make_kinodata_module
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.types import NodeType
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform.to_complex_graph import TransformToComplexGraph
from kinodata.transform.ligand_only import ToLigandOnlyComplex


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
    )
    config = configuration.get("data", "training", "sparse_transformer")
    config = config.update_from_file()
    config = config.update_from_args()
    config["need_distances"] = False
    config["mask_pl_edges"] = False
    config["ligand_only_3d"] = True
    config["perturb_ligand_positions"] = 0.0
    config["perturb_pocket_positions"] = 0.0
    config["perturb_complex_positions"] = 0.1

    config["node_types"] = [NodeType.Complex]

    for key, value in sorted(config.items(), key=lambda i: i[0]):
        print(f"{key}: {value}")
    
    ott = TransformToComplexGraph(remove_heterogeneous_representation=True) 
    if config.get("ligand_only_3d", None) is not None:
        ott = Compose([ott, ToLigandOnlyComplex()])


    wandb.init(config=config, project="kinodata-docked-rescore", tags=["transformer"])
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
