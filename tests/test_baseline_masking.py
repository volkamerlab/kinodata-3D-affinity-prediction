import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets.graph_generator import ERGraph
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import tqdm
from kinodata.transform.baseline_masking import MaskLigand, MaskLigandPosition
from kinodata.transform import TransformToComplexGraph
from kinodata.types import NodeType, RelationType
from kinodata.data import KinodataDocked
from kinodata.model.complex_transformer import StructuralInteractions


def dummy_data():
    gen = ERGraph(num_nodes=18, edge_prob=0.3)
    edge_index = gen().edge_index
    x = torch.randn(18, 32)
    z = torch.randint(0, 20, (18,))
    pos = torch.randn(18, 3)
    pocket_size = 12
    is_pocket_atom = torch.zeros(18, dtype=torch.bool)
    is_pocket_atom[:pocket_size] = True
    data = HeteroData()
    data[NodeType.Complex].x = x
    data[NodeType.Complex].z = z
    data[NodeType.Complex].pos = pos
    data[NodeType.Complex].is_pocket_atom = is_pocket_atom
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_index = (
        edge_index
    )
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_attr = (
        torch.randn(edge_index.size(1), 4)
    )
    return data


def test_mask_ligand():
    data = dummy_data()
    print(data)
    num_ligand = (
        data[NodeType.Complex].x.size(0) - data[NodeType.Complex].is_pocket_atom.sum()
    )
    mask = MaskLigand()
    masked = mask(data)
    assert masked[NodeType.Complex].x.size(0) == num_ligand


@torch.inference_mode()
def test_no_interactions_after_ligand_position_masking():
    dataset = KinodataDocked(
        transform=Compose([TransformToComplexGraph(), MaskLigandPosition()])
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    intr_module = StructuralInteractions(hidden_channels=2)
    for idx, batch in tqdm.tqdm(enumerate(loader)):
        if idx > 100:
            break
        edge_index, _ = intr_module.forward(batch)
        batch[NodeType.Complex].is_pocket_atom
        row, col = edge_index
        is_pl_interaction = torch.logical_xor(
            batch[NodeType.Complex].is_pocket_atom[row],
            batch[NodeType.Complex].is_pocket_atom[col],
        )
        assert torch.all(is_pl_interaction == False)


if __name__ == "__main__":
    test_mask_ligand()
    test_no_interactions_after_ligand_position_masking()
