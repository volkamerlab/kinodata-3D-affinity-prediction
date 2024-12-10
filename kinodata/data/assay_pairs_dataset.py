from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import multiprocessing as mp
from itertools import count, batched, product
from pathlib import Path
from typing import Callable, Generator, Iterable, List, Sequence, Tuple, Union

import torch
import numpy as np
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.data.data import BaseData

from kinodata.data.featurization.rdkit import (
    set_atoms,
    set_bonds,
    append_atoms_and_bonds,
)
from kinodata.types import NodeType

from .dataset import ComplexInformation

ComplexPair = tuple[ComplexInformation, ComplexInformation]


@dataclass
class ProcessLigandArgs:
    add_spatial_edges: bool
    radius: float


@dataclass
class ProcessKinaseArgs:
    add_spatial_edges: bool
    radius: float


def process_ligand_pair_to_pyg(
    complex_pair: tuple[ComplexInformation],
    process_ligand_args: ProcessLigandArgs,
    data: HeteroData | None = None,
) -> HeteroData:
    if data is None:
        data = HeteroData()
    cplx1, cplx2 = complex_pair
    data = set_atoms(cplx1.ligand, data, NodeType.Ligand)
    data = set_bonds(cplx1.ligand, data, NodeType.Ligand)
    ligand_one_size = data[NodeType.Ligand].x.size(0)
    data = append_atoms_and_bonds(cplx2.ligand, data, NodeType.Ligand)
    ligand_two_size = data.x.size(0) - ligand_one_size
    data[NodeType.Ligand].component_index = torch.cat(
        (
            torch.zeros(ligand_one_size, dtype=torch.long),
            torch.ones(ligand_two_size, dtype=torch.long),
        ),
        dim=0,
    )
    return data


def process_kinase_to_pyg(
    complex: ComplexInformation,
    process_kinase_args: ProcessKinaseArgs,
    data: HeteroData | None = None,
) -> HeteroData:
    if data is None:
        data = HeteroData()
    pocket = complex.pocket
    data = set_atoms(pocket, data, NodeType.Pocket)
    data = set_bonds(pocket, data, NodeType.Pocket)
    return data


def filter_intra_assay(complex_pairs: Iterable[ComplexPair]) -> list[ComplexPair]:
    return [
        (cplx_0, cplx_1)
        for cplx_0, cplx_1 in complex_pairs
        if cplx_0.assay_id == cplx_1.assay_id
    ]


def iter_intra_assay_pairs(
    complexes: ComplexInformation,
    processes: int | None = None,
    processing_batch_size: int = 512,
) -> Generator[ComplexPair, None, None]:
    pairs = product(complexes, repeat=2)
    if processes is None:
        for cplx_0, cplx_1 in pairs:
            if cplx_0.assay_id != cplx_1.assay_id:
                continue
            yield (cplx_0, cplx_1)
    with mp.Pool(processes) as p:
        for pair_batch in p.apply_async(
            filter_intra_assay, batched(pairs, processing_batch_size)
        ):
            for pair in pair_batch:
                yield pair


def fast_pair_index_numpy(
    groupby: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    groupby = np.array(groupby)
    # TODO assert is int: ids.dtype
    is_intra_assay = groupby.reshape(-1, 1) - groupby.reshape(1, -1) == 0
    return np.where(is_intra_assay)


def fast_pair_index(groupby: Iterable[int]) -> Generator[tuple[int, int], None, None]:
    buckets = defaultdict(list)
    for index, group in enumerate(groupby):
        for other_index in buckets[group]:
            yield other_index, index
            yield index, other_index
        buckets[group].append(index)


def union_hetero_graph_data(
    data1: HeteroData, data2: HeteroData, cat_dim: int
) -> HeteroData:
    node_types1, edge_types1 = data1.metadata()
    node_types2, edge_types2 = data2.metadata()
    for shared_node_type in set(node_types1).intersection(node_types2):
        node_store1 = data1[shared_node_type]
        node_store2 = data2[shared_node_type]
        for attr in node_store1.keys():
            if attr not in node_store2:
                continue
            val1 = node_store1[attr]
            val2 = node_store2[attr]
            if isinstance(val1, list):
                assert isinstance(val2, list)
                node_store1[attr] = val1.extend(val2)
            if isinstance(val1, torch.Tensor):
                assert isinstance(val2, torch.Tensor)
                assert val1.size() == val2.size()
                node_store1[attr] = torch.cat((val1, val2), dim=cat_dim)
        # data1[shared_node_type] = node_store1 TODO is this needed?
    for specific_node_type in set(node_types2).difference(node_types1):
        # TODO does this work?
        data1[specific_node_type] = data2[specific_node_type]

    return data1


def generate_distance_based_edges(
    pos: torch.Tensor,
    radius: float,
    max_num_neighbors: int,
) -> torch.Tensor:
    edge_index, distance


@dataclass
class ProcessingResultInformation:
    processed_activities: list[int]
    failed_activities: list[tuple[int, str]]

    def save(self, fp: Path) -> Path:
        dict_repr = asdict(self)
        with open(fp, "w") as f:
            json.dump(dict_repr, f)

    @classmethod
    def load(cls, fp: Path) -> "ProcessingResultInformation":
        with open(fp, "r") as f:
            dict_repr = json.load(fp)
        return cls(**dict_repr)

    def log_processing_failure(
        self, cplx: ComplexInformation, reason: Exception | None = None
    ):
        aid = cplx.chembl_activity_id
        self.failed_activities.append((aid, str(reason)))


class PairwiseIntraAssayDataset(Dataset):

    def __init__(
        self,
        root: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        self.processing_result = None
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    def process(self) -> None:
        processing_result = ProcessingResultInformation()
        kinodata3d_data_frame = self.make_df()
        complexes = self.make_complexes(kinodata3d_data_frame)
        assay_ids = [cplx.assay_id for cplx in complexes]
        pair_counter = count()
        for i, j in fast_pair_index(assay_ids):
            complex_pair = (complexes[i], complexes[j])
            try:
                data = process_ligand_pair_to_pyg(complex_pair)
            except Exception as excpt:
                processing_result.log_processing_failure(complex_pair, excpt)
            if data is None:
                processing_result.log_processing_failure(complex_pair, None)
            index = next(pair_counter)
            self.persist_pair(data, index)

        kinase_data = dict()
        for complex in complexes:
            if (kinase_id := complex.klifs_structure_id) in kinase_data:
                continue
            data = process_kinase_to_pyg(complex)
            kinase_data[kinase_id] = data
        self.persist_kinases(kinase_data)

    def get(self, idx: int) -> BaseData:
        ligand_pair = self.load_ligand_pair(idx)
        pocket_data = self.get_pocket_data(ligand_pair.klifs_structure_id)
        complex_data = union_hetero_graph_data(ligand_pair, pocket_data)
        complex_data = add_distance_based_edges(complex_data)
        return complex_data

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        raise NotImplementedError

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError
