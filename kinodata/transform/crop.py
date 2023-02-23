from typing import Optional
import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph

from torch_cluster import knn

from kinodata.typing import NodeType, EdgeType


class RandomCropComplex(BaseTransform):
    def __init__(self, min_frac_ligand: float = 0.2, min_frac_pocket: float = 0.2):
        """Crops a ligand/protein(pocket) complex in euclidean space.

        Randomly select a ligand atom as crop origin and crop the complex
        by only keeping the `k`-nearest neighrbor ligand and `k`-nearest neighbor pocket atoms/nodes.

        `k` is chosen randomly as
        `int(u * N_ligand_atoms)` and
        `int(u * N_pocket_atoms)` respectively,
        where `u` is a standard uniform random variable.


        Parameters
        ----------
        min_frac_ligand : float, optional
            minimum fraction of ligand atoms to keep, by default 0.2
        min_frac_pocket : float, optional
            minimum fraction of pocket atoms to keep, by default 0.2
        """
        self.min_frac_ligand = min_frac_ligand
        self.min_frac_pocket = min_frac_pocket

    def __call__(self, data: HeteroData) -> HeteroData:

        frac = torch.rand(1).item()
        frac_ligand = max(frac, self.min_frac_ligand)
        frac_pocket = max(frac, self.min_frac_pocket)
        ligand_pos = data["ligand"].pos
        pocket_pos = data["pocket"].pos
        n_ligand = ligand_pos.size(0)
        n_ligand_subset = int(frac_ligand * n_ligand)
        n_pocket = pocket_pos.size(0)
        n_pocket_subset = int(frac_pocket * n_pocket)

        # select a random origin point
        oidx = torch.randint(low=0, high=n_ligand, size=(1,))
        origin = ligand_pos[oidx].view(1, 3)

        # keep fraction of closest ligand atoms
        _, ligand_subset = knn(ligand_pos, origin, k=n_ligand_subset)
        data = self._subset_molecule(
            data, ligand_subset, "ligand", ("ligand", "bond", "ligand")
        )

        # keep fraction of closest pocket atoms
        _, pocket_subset = knn(pocket_pos, origin, k=n_pocket_subset)
        data = self._subset_molecule(data, pocket_subset, "pocket")

        return data

    def _subset_molecule(
        self,
        data: HeteroData,
        subset: Tensor,
        node_key: NodeType,
        edge_key: Optional[EdgeType] = None,
    ) -> HeteroData:

        # subset atoms/nodes
        data[node_key].pos = data[node_key].pos[subset]
        data[node_key].z = data[node_key].z[subset]

        # subset edges/bonds
        if edge_key is not None:
            size = data[node_key].pos.size(0)
            subset_size = subset.size(0)
            edge_store = data[edge_key[0], edge_key[1], edge_key[2]]
            node_relabel = -torch.ones(size, dtype=torch.long)
            node_relabel[subset] = torch.arange(subset_size)
            masked_edge_index, masked_edge_attr = subgraph(
                subset,
                edge_store.edge_index,
                edge_store.edge_attr,
            )
            masked_edge_index[0, :] = node_relabel[masked_edge_index[0, :]]
            masked_edge_index[1, :] = node_relabel[masked_edge_index[1, :]]
            edge_store.edge_index = masked_edge_index
            edge_store.edge_attr = masked_edge_attr

        return data
