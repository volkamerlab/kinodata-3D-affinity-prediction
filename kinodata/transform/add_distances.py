from typing import Tuple
import torch
from torch import FloatTensor, LongTensor

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from torch_cluster import radius
from itertools import product


def interactions_and_distances(
    pos1: FloatTensor, pos2: FloatTensor, r: float
) -> Tuple[LongTensor, FloatTensor]:
    y_ind, x_ind = radius(pos1, pos2, r)
    dist = (pos1[x_ind] - pos2[y_ind]).pow(2).sum(dim=1).sqrt()
    return torch.stack((x_ind, y_ind)), dist


class AddDistancesAndInteractions(BaseTransform):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:
        if isinstance(data, HeteroData):
            node_types, _ = data.metadata()
            for nt_a, nt_b in product(node_types, node_types):
                edge_index, dist = interactions_and_distances(
                    data[nt_a].pos, data[nt_b].pos, self.radius
                )
                data[nt_a, "interacts", nt_b].edge_index = edge_index
                data[nt_a, "interacts", nt_b].dist = dist.view(-1, 1).to(torch.float32)
            return data
        else:
            raise NotImplementedError
            ...


if __name__ == "__main__":

    x = torch.randn(5, 2)
    y = torch.randn(3, 2)
    edge_index, dist = interactions_and_distances(x, y, 1.5)
    print(dist)
