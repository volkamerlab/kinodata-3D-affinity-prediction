from torch_geometric.transforms import BaseTransform
from kinodata.types import NodeType, RelationType


class FeatureMask(BaseTransform):
    def __init__(self, mask, bond_mask=None):
        self.mask = mask
        self.bond_mask = bond_mask

    def forward(self, data):
        data[NodeType.Complex].x = data[NodeType.Complex].x[:, self.mask]
        if self.bond_mask is not None:
            data[
                NodeType.Complex, RelationType.Covalent, NodeType.Complex
            ].edge_attr = data[
                NodeType.Complex, RelationType.Covalent, NodeType.Complex
            ].edge_attr[:, self.bond_mask]
        return data
