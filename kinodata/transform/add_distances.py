from typing import Optional, Tuple
import torch
from torch import FloatTensor, LongTensor

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_undirected
from torch_cluster import radius
from itertools import product


def interactions_and_distances(
    pos1: FloatTensor,
    pos2: Optional[FloatTensor] = None,
    r: float = 1.0,
) -> Tuple[LongTensor, FloatTensor]:
    if pos2 is None:
        pos2 = pos1
    y_ind, x_ind = radius(pos1, pos2, r)
    dist = (pos1[x_ind] - pos2[y_ind]).pow(2).sum(dim=1).sqrt()
    if dist.numel() > 0:
        print(dist.max().item())
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


if __name__ == "__main__":

    x = torch.tensor([1.0, 2.1, 3.8]).float().unsqueeze(1)
    y = torch.tensor([0, 3, 4.7]).float().unsqueeze(1)

    edge_index, dist = interactions_and_distances(x, y, 1.0)
    print(edge_index, dist)
    edge_index, dist = interactions_and_distances(y, x, 1.0)
    print(edge_index, dist)
