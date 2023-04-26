from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from kinodata.data.data_split import Split
from kinodata.data.utils.cluster import AffinityPropagation
from kinodata.data.utils.solve_split_assignment import solve_random


class ColdSplit:
    def __init__(
        self,
        train_size: float,
        val_size: float,
        test_size: float,
        attribute_key: str,
        solver=solve_random,
        clustering=AffinityPropagation(),
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.attribute_key = attribute_key
        self.solver = solver

        self.clustering = clustering

    def __call__(
        self,
        dataset: pd.DataFrame,
        seed: int,
        similarities: np.ndarray = None,
    ) -> Split:
        assert self.attribute_key in dataset.columns
        if self.clustering is not None:
            assert similarities is not None
            df_clustered = self.clustering(dataset, self.attribute_key, similarities)
            df_split = self.solver(
                df_clustered,
                self.clustering.cluster_key,
                {
                    "train": self.train_size,
                    "val": self.val_size,
                    "test": self.test_size,
                },
                seed,
            )
        else:
            df_split = solve_random(
                dataset,
                self.attribute_key,
                {
                    "train": self.train_size,
                    "val": self.val_size,
                    "test": self.test_size,
                },
                seed,
            )
        return Split.from_data_frame(df_split)
