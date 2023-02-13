from typing import Optional, Tuple
import torch
from torch import Tensor 

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import (
    remove_self_loops,
    to_dense_adj,
    to_undirected,
    coalesce,
)
from torch_cluster import radius
from itertools import product


def interactions_and_distances(
    pos1: Tensor,
    pos2: Optional[Tensor] = None,
    r: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    if pos2 is None:
        pos2 = pos1
    y_ind, x_ind = radius(pos1, pos2, r)
    dist = (pos1[x_ind] - pos2[y_ind]).pow(2).sum(dim=1).sqrt()
    edge_index = torch.stack((x_ind, y_ind))
    return edge_index, dist


class AddDistancesAndInteractions(BaseTransform):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius

    def __call__(self, data: HeteroData) -> HeteroData:
        if isinstance(data, HeteroData):
            node_types, edge_types = data.metadata()
            for nt_a, nt_b in product(node_types, node_types):
                edge_index, dist = interactions_and_distances(
                    data[nt_a].pos,
                    data[nt_b].pos,
                    self.radius,
                )
                if nt_a == nt_b:
                    num_nodes = data[nt_a].num_nodes
                    edge_index, dist = to_undirected(
                        edge_index, dist, num_nodes, reduce="min"
                    )
                    edge_index, dist = remove_self_loops(edge_index, dist)

                if nt_a == nt_b and (nt_a, "bond", nt_b) in edge_types:
                    num_nodes = data[nt_a].num_nodes
                    bond_adj = to_dense_adj(
                        data[nt_a, "bond", nt_a].edge_index,
                        edge_attr=data[nt_a, "bond", nt_a].edge_attr,
                        max_num_nodes=num_nodes,
                    ).squeeze(0)
                    row, col = edge_index
                    data[nt_a, "interacts", nt_b].edge_attr = bond_adj[row, col]

                data[nt_a, "interacts", nt_b].edge_index = edge_index
                data[nt_a, "interacts", nt_b].dist = dist.view(-1, 1).to(torch.float32)

            return data
        else:
            raise NotImplementedError
            ...


class ForceSymmetricInteraction(BaseTransform):
    def __init__(self, edge_type: Tuple[str, str, str]) -> None:
        super().__init__()
        self.edge_type = edge_type
        nt_a, relation, nt_b = self.edge_type
        assert nt_a != nt_b
        assert relation == "interacts"

    def _reverse_direction(self, edge_index):
        row, col = edge_index
        return torch.stack((col, row))

    def __call__(self, data: HeteroData) -> HeteroData:

        nt_a, relation, nt_b = self.edge_type

        for source, target in ([nt_a, nt_b], [nt_b, nt_a]):
            edge_index_1 = data[source, relation, target].edge_index
            dist_1 = data[source, relation, target].dist
            edge_index_2 = data[target, relation, source].edge_index
            dist_2 = data[target, relation, source].dist

            merged_edge_index = torch.cat(
                (edge_index_1, self._reverse_direction(edge_index_2)), dim=1
            )
            merged_dist = torch.cat((dist_1, dist_2))
            merged_edge_index, merged_dist = coalesce(
                merged_edge_index, merged_dist, reduce="mean"
            )

            data[source, relation, target].edge_index = merged_edge_index
            data[source, relation, target].dist = merged_dist

        return data


if __name__ == "__main__":

    x = torch.tensor([1.0, 2.1, 3.8]).float().unsqueeze(1)
    y = torch.tensor([0, 3, 4.7]).float().unsqueeze(1)

    edge_index, dist = interactions_and_distances(x, y, 1.0)
    print(edge_index, dist)
    edge_index, dist = interactions_and_distances(y, x, 1.0)
    print(edge_index, dist)
