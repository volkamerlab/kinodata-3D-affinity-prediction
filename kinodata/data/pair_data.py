import pandas as pd
from rdkit.Chem import PandasTools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Any, Union
from kinodata.data.dataset import KinodataDocked
from torch_geometric.data import Data, InMemoryDataset, collate, HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage
from tqdm import tqdm
from itertools import combinations
import torch


def pair_data(primary: HeteroData, secondary: HeteroData) -> HeteroData:
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

        mask = edges.edge_index < size_s
        edges_s = EdgeStorage(
            {
                "edge_index": edges.edge_index[:, mask[0]],
                "edge_attr": edges.edge_attr[mask[0]],
            }
        )
        edges_t = EdgeStorage(
            {
                "edge_index": edges.edge_index[:, ~mask[1]] - size_s,
                "edge_attr": edges.edge_attr[~mask[0]],
            }
        )

        return nodes_s, edges_s, nodes_t, edges_t

    paired, _, _ = collate.collate(HeteroData, [primary, secondary], add_batch=True)
    s, bond_s, t, bond_t = split_graph(
        paired["ligand"], paired[("ligand", "bond", "ligand")]
    )
    del paired["ligand"]
    del paired[("ligand", "bond", "ligand")]

    def set_subgraph(nodes: NodeStorage, edges: EdgeStorage, name: str):
        paired[name] = nodes
        paired[(name, "bond", name)] = edges

    set_subgraph(s, bond_s, "ligand_s")
    set_subgraph(t, bond_t, "ligand_t")

    s, bond_s, t, bond_t = split_graph(
        paired["pocket"], paired[("pocket", "bond", "pocket")]
    )
    del paired["pocket"]
    del paired[("pocket", "bond", "pocket")]
    set_subgraph(s, bond_s, "pocket_s")
    set_subgraph(t, bond_t, "pocket_t")

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

        pair_list = [
            pair_data(a, b)
            for a, b in tqdm(filter(self.pair_filter, combinations(data_list, 2)))
        ]

        self.persist(pair_list)
