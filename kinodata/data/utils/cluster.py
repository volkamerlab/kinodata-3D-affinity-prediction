import pandas as pd
import numpy as np
from typing import Generic, Sequence, TypeVar

import sklearn.cluster as cluster

T = TypeVar("T")


class Clustering(Generic[T]):
    cluster_key: str = "cluster_index"
    similarities: np.ndarray
    unique_items: Sequence[T]

    def __call__(
        self,
        df: pd.DataFrame,
        key: str,
        similarities: np.ndarray,
    ) -> pd.DataFrame:
        self.unique_items: Sequence[T] = df[key].unique()
        self.similarities = similarities
        cluster_labels = self.solve()
        df_cluster = pd.DataFrame(
            {
                key: self.unique_items,
                self.cluster_key: cluster_labels,
            }
        )
        return pd.merge(df, df_cluster, on=key)

    def solve(self) -> Sequence[int]:
        raise NotImplementedError


class AffinityPropagation(Clustering):
    def __init__(self) -> None:
        super().__init__()
        self.model = cluster.AffinityPropagation(affinity="precomputed")

    def solve(self):
        self.model.fit(self.similarities)
        return self.model.labels_
