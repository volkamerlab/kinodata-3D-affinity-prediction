from itertools import combinations
from typing import List, Callable, Optional, Any, Union, Dict

from tqdm import tqdm
from rdkit.Chem import PandasTools
import torch
from torch_geometric.data import Data, InMemoryDataset, collate, HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage

from kinodata.types import NodeType, RelationType
from kinodata.data.dataset import KinodataDocked


def split_graph(
    nodes: NodeStorage, edges: EdgeStorage
) -> (NodeStorage, EdgeStorage, NodeStorage, EdgeStorage):
    keys = list(nodes.to_dict().keys())
    mask = nodes.batch == 0
    size_s = nodes.ptr[1]
    nodes_s = NodeStorage(
        {k: nodes[k][mask] for k in keys if nodes[k].size(0) == len(mask)}
    )
    nodes_t = NodeStorage(
        {k: nodes[k][~mask] for k in keys if nodes[k].size(0) == len(mask)}
    )
    del nodes_s["batch"]
    del nodes_t["batch"]

    mask = edges.edge_index < size_s
    edges_s = {
        "edge_index": (edges.edge_index[:, mask[0]]),
        "edge_attr": edges.edge_attr[mask[0]],
    }
    edges_t = {
        "edge_index": (edges.edge_index[:, ~mask[1]] - size_s),
        "edge_attr": edges.edge_attr[~mask[0]],
    }

    return nodes_s, edges_s, nodes_t, edges_t


def split_graphs(combined: HeteroData):
    paired = HeteroData()
    s, bond_s, t, bond_t = split_graph(
        combined[NodeType.Ligand],
        combined[(NodeType.Ligand, RelationType.Covalent, NodeType.Ligand)],
    )

    def set_subgraph(
        nodes: NodeStorage, edges: Dict[str, torch.Tensor], node_type: NodeType
    ):
        paired[node_type] = nodes
        paired[(node_type, RelationType.Covalent, node_type)].edge_index = edges[
            "edge_index"
        ]
        paired[(node_type, RelationType.Covalent, node_type)].edge_attr = edges[
            "edge_attr"
        ]

    set_subgraph(s, bond_s, NodeType.LigandA)
    set_subgraph(t, bond_t, NodeType.LigandB)

    s, bond_s, t, bond_t = split_graph(
        combined[NodeType.Pocket],
        combined[(NodeType.Pocket, RelationType.Covalent, NodeType.Pocket)],
    )
    set_subgraph(s, bond_s, NodeType.PocketA)
    set_subgraph(t, bond_t, NodeType.PocketB)

    return paired


def join_metadata(paired: HeteroData, combined: HeteroData, matching_properties):
    paired.y = combined.y[0] - combined.y[1]
    for key in [
        "scaffold",
        "ident",
        "klifs_structure_id",
        "posit_prob",
        "assay_ident",
        "pocket_sequence",
        "predicted_rmsd",
        "activity_type",
    ]:
        if key in matching_properties:
            paired[key] = combined[key][0]
        else:
            paired["metadata"][key] = combined[key]

    return paired


def pair_data(primary: HeteroData, secondary: HeteroData, matching_properties: List[str]) -> HeteroData:
    combined, _, _ = collate.collate(HeteroData, [primary, secondary], add_batch=True)
    paired = split_graphs(combined)
    paired = join_metadata(paired, combined, matching_properties)

    return paired


class PropertyPairing(Callable):
    def __init__(
        self,
        matching_properties: List[str] = [],
        non_matching_properties: List[str] = [],
    ):
        self.matching_properties = matching_properties
        self.non_matching_properties = non_matching_properties

    def __call__(self, x):
        return all(
            getattr(x[0], p) == getattr(x[1], p) for p in self.matching_properties
        ) and all(
            getattr(x[0], p) != getattr(x[1], p) for p in self.non_matching_properties
        )


class KinodataDockedPairs(KinodataDocked):
    def __init__(
        self,
        matching_properties: List[str] = [],
        non_matching_properties: List[str] = [],
        **kwargs
    ):
        self.matching_properties = matching_properties
        self.non_matching_properties = non_matching_properties
        super().__init__(**kwargs)

    @property
    def processed_file_names(self) -> List[str]:
        return ["kinodata_docked_v2_paired.pt"]

    def process(self):
        data_list = super().make_data_list()
        data_list = super().filter_transform(data_list)

        pair_filter = PropertyPairing(
            self.matching_properties,
            self.non_matching_properties,
        )
        data_list = [
            pair_data(a, b, self.matching_properties)
            for a, b in tqdm(
                filter(pair_filter, combinations(data_list, 2)),
            )
        ]

        self.persist(data_list)
