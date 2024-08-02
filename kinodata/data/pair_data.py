from itertools import combinations
from typing import List, Callable, Optional, Any, Union

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
    edges_s = EdgeStorage(
        {
            "edge_index": (edges.edge_index[:, mask[0]]).T,
            "edge_attr": edges.edge_attr[mask[0]],
        }
    )
    edges_t = EdgeStorage(
        {
            "edge_index": (edges.edge_index[:, ~mask[1]] - size_s).T,
            "edge_attr": edges.edge_attr[~mask[0]],
        }
    )

    return nodes_s, edges_s, nodes_t, edges_t


def pair_data(primary: HeteroData, secondary: HeteroData) -> HeteroData:
    paired, _, _ = collate.collate(HeteroData, [primary, secondary], add_batch=True)
    s, bond_s, t, bond_t = split_graph(
        paired[NodeType.Ligand],
        paired[(NodeType.Ligand, RelationType.Covalent, NodeType.Ligand)],
    )
    del paired[NodeType.Ligand]
    del paired[(NodeType.Ligand, RelationType.Covalent, NodeType.Ligand)]

    def set_subgraph(nodes: NodeStorage, edges: EdgeStorage, node_type: NodeType):
        paired[node_type] = nodes
        paired[(node_type, RelationType.Covalent, node_type)] = edges

    set_subgraph(s, bond_s, NodeType.LigandA)
    set_subgraph(t, bond_t, NodeType.LigandB)

    s, bond_s, t, bond_t = split_graph(
        paired[NodeType.Pocket],
        paired[(NodeType.Pocket, RelationType.Covalent, NodeType.Pocket)],
    )
    del paired[NodeType.Pocket]
    del paired[(NodeType.Pocket, RelationType.Covalent, NodeType.Pocket)]
    set_subgraph(s, bond_s, NodeType.PocketA)
    set_subgraph(t, bond_t, NodeType.PocketB)

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
        pair_filter: Optional[Callable[[tuple[HeteroData, HeteroData]], bool]] = None,
        **kwargs
    ):
        self.pair_filter = pair_filter
        super().__init__(**kwargs)

    @property
    def processed_file_names(self) -> List[str]:
        return ["kinodata_docked_v2_paired.pt"]

    def process(self):
        data_list = super().make_data_list()
        data_list = super().filter_transform(data_list)

        n_data = len(data_list)
        pair_list = [
            pair_data(a, b)
            for a, b in tqdm(
                filter(self.pair_filter, combinations(data_list, 2)),
            )
        ]

        self.persist(pair_list)
