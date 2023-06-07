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
        """A class used for generating "cold" data splits
        where two data points can only be part of different splits,
        if they differ in some attribute (e.g. scaffold or pocket cluster membership).

        Parameters
        ----------
        train_size : float
            Fraction of data to be used for training
        val_size : float
            Fraction of data to be used for validation
        test_size : float
            Fraction of data to be used for testing
        attribute_key : str
            The name used to index the attribute the split is based on.
        solver : _type_, optional
            A function that solves the split, ie assigns items to train,
            val and test split such that the "cold" split property
            is satisfied and they roughly match the given split sizes.
            By default solve_random
        clustering : _type_, optional
            Further cluster data points based on some similarity measure
            before performing the split.
            By default AffinityPropagation()
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.attribute_key = attribute_key
        self.split_solver = solver

        self.clustering = clustering

    def __call__(
        self,
        dataset: pd.DataFrame,
        seed: int,
        similarities: np.ndarray = None,
    ) -> Split:
        """Split a given dataset using the internal solver with the given seed.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset that should be split
        seed : int
            Random seed for the split solver
        similarities : np.ndarray, optional
            Precomputed, pairwise similarities of unique items in the dataset.
            Must only be provided if the split uses some clustering method.

            # TODO: similarities should be indexable by attribute pairs, rather than some integer-index?

        Returns
        -------
        Split
            A data split object that stores the splits as separate lists of data point indices.
        """
        splittable_data = dataset
        attribute_key = self.attribute_key
        if self.clustering is not None:
            assert (
                similarities is not None
            ), "Must provide precomputed similarities for clustering"
            splittable_data = self.clustering(
                splittable_data, self.attribute_key, similarities
            )
            # base splits on clusters instead of directly on attribute
            attribute_key = self.clustering.cluster_key

        df_split = self.split_solver(
            splittable_data,
            attribute_key,
            {
                "train": self.train_size,
                "val": self.val_size,
                "test": self.test_size,
            },
            seed,
        )
        return Split.from_data_frame(df_split)
