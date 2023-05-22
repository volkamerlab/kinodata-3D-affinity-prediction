import torch
from torch import Tensor
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from .bonds import BOND_TYPE_TO_IDX, NUM_BOND_TYPES


def atomic_numbers(mol) -> Tensor:
    z = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for i, atom in enumerate(mol.GetAtoms()):
        z[i] = atom.GetAtomicNum()
    return z


def atom_positions(mol) -> Tensor:
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    return torch.from_numpy(pos).to(torch.float)


def add_atoms(mol, data, key):
    data[key].z = atomic_numbers(mol)
    data[key].pos = atom_positions(mol)
    assert data[key].z.size(0) == data[key].pos.size(0)
    return data


def add_bonds(mol, data, key):
    row, col = list(), list()
    bond_type_indices = []
    num_nodes = mol.GetNumAtoms()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        type = bond.GetBondType()
        row.append(i)
        col.append(j)
        bond_type_indices.append(BOND_TYPE_TO_IDX[type])

    edge_index = torch.tensor([row, col], dtype=torch.long).view(2, -1)
    bond_type_indices = torch.tensor(bond_type_indices)

    # one-hot encode bond type
    edge_attr = F.one_hot(bond_type_indices, NUM_BOND_TYPES)

    edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes)

    data[key, "bond", key].edge_index = edge_index
    data[key, "bond", key].edge_attr = edge_attr

    return data
