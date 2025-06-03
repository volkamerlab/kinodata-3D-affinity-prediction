import torch
from torch_cluster import knn
from torch_scatter import scatter
from torch_geometric.utils import coalesce
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _impl_torch_cluster(
    u: torch.Tensor,
    v: torch.Tensor,
    query: torch.Tensor | None = None,
    k: int = 1,
):
    uy = u
    vy = v
    if query is not None:
        uy = uy[query]
        vy = vy[query]

    knns_u = knn(u, uy, k=k + 1)
    knns_v = knn(v, vy, k=k + 1)

    (query_index, _), is_mnn = coalesce(
        torch.cat((knns_u, knns_v), dim=1),
        torch.ones(knns_u.size(1) + knns_v.size(1)),
        reduce="sum",
    )
    # (is_mnn == 2) means that the point is a mutual nearest neighbor
    is_mnn = (is_mnn == 2).to(torch.int64)
    num_mutual_nns = scatter(is_mnn, query_index, dim=0, reduce="sum")
    return (num_mutual_nns.view(-1, 1) - 1).cpu()


def _impl_sklearn(u: torch.Tensor, v: torch.Tensor, query: torch.Tensor | None, k: int):
    n = u.shape[0]
    if query is None:
        query_u = u
        query_v = v
        query_idx = np.arange(n)
    else:
        query_u = u[query]
        query_v = v[query]
        query_idx = query

    # Get indices of k nearest neighbors
    knn_u = NearestNeighbors(n_neighbors=k + 1).fit(u)
    knn_v = NearestNeighbors(n_neighbors=k + 1).fit(v)

    indices_u = knn_u.kneighbors(query_u, return_distance=False)
    indices_v = knn_v.kneighbors(query_v, return_distance=False)

    # Exclude the self-match (first neighbor = self)
    indices_u = indices_u[:, 1:]
    indices_v = indices_v[:, 1:]

    mutual_counts = np.zeros(n, dtype=float)

    for i, q in enumerate(query_idx):
        set_u = set(indices_u[i])
        set_v = set(indices_v[i])
        mutual_counts[q] = len(set_u & set_v)

    return torch.tensor(
        mutual_counts.reshape(-1, 1)[query_idx]
        if query is not None
        else mutual_counts.reshape(-1, 1)
    )


def num_mutual_nearest_neighbors(
    u: torch.Tensor,
    v: torch.Tensor,
    query: torch.Tensor | None = None,
    k: int = 1,
    impl: str = "sklearn",
) -> torch.Tensor:
    """Compute the number of mutual nearest neighbors between two sets of embeddings.

    Args:
        u (torch.Tensor): Embeddings of the first set.
        v (torch.Tensor): Embeddings of the second set.
        query (torch.Tensor | None):  A tensor of dtype long. Specifies the indices of the embeddings to use as roots for querying neighbor sets.
        If None, all embeddings are used as a query.
        k (int): Number of nearest neighbors to consider.
        impl (str, optional): Implemenation to use. Defaults to "sklearn".

    Raises:
        ValueError: If `u` and `v` do not have the same number of rows.
        ValueError: If an unknown implementation is specified.

    Returns:
        torch.Tensor: A tensor containing the number of mutual nearest neighbors for each query.
    """
    if u.shape[0] != v.shape[0]:
        raise ValueError("u and v must have the same number of rows.")
    if impl == "torch_cluster":
        return _impl_torch_cluster(u, v, query, k)
    elif impl == "sklearn":
        return _impl_sklearn(u, v, query, k)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def expected_num_mutual_neighbors(n: int, k: int) -> float:
    """Compute the expected number of mutual nearest neighbors for a given number of points and k.

    Args:
        n (int): Number of points.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: The expected number of mutual nearest neighbors.
    """
    if k > n:
        raise ValueError("k must be less than or equal to n.")
    return (k**2) / n
