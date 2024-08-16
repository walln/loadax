"""Dataset transformations that represent sampling items from the source dataset."""

import random
from typing import TypeVar

from loadax.dataset.protocol import Dataset

DatasetItem = TypeVar("DatasetItem")


class SampledDatasetWithoutReplacement(Dataset[DatasetItem]):
    """Sample a subset of the items in the source dataset without replacement.

    This dataset allows sampling a subset of the items in the source dataset without
    replacement. The sampling eagerly shuffles the indices of the source dataset and
    then selects the first `sample_size` indices. This is a simple implementation
    that does not perform any kind of sampling with replacement.

    The sampling does not alter the underlying storage of the source dataset. If your
    underlying storage does not perform well with random access, you may want to
    consider performing the sampling in advance. However, for almost all use cases,
    this should not be necessary.

    The lazy sampling enables better reproducibility and determinism, while also
    being extremely efficient as only the indices need to be sampled. This also means
    that sampling is not IO bound as the indices can be stored in memory.

    Example:
        >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
        >>> sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3)
        >>> print(sampled_dataset.get(0))

    Attributes:
        dataset (Dataset): The underlying dataset to sample from.
        sample_size (int): The size of the sample to take.
        indices (list[int]): The indices to sample from.
    """

    def __init__(self, dataset: Dataset[DatasetItem], sample_size: int):
        """Sample a subset of the items in the source dataset without replacement.

        This dataset allows sampling a subset of the items in the source dataset without
        replacement. The sampling eagerly shuffles the indices of the source dataset and
        then selects the first `sample_size` indices. This is a simple implementation
        that does not perform any kind of sampling with replacement.

        The sampling does not alter the underlying storage of the source dataset. If
        your underlying storage does not perform well with random access, you may want
        to consider performing the sampling in advance. However, for almost all use
        cases,this should not be necessary.

        The lazy sampling enables better reproducibility and determinism, while also
        being extremely efficient as only the indices need to be sampled. This also
        means that sampling is not IO bound as the indices can be stored in memory.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3)
            >>> print(sampled_dataset.get(0))

        Args:
            dataset (Dataset): The dataset to sample from.
            sample_size (int): The size of the sample to take.
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.indices = []

    def __len__(self):
        """Get the length of the dataset.

        In the case of a sampled dataset, the length is the sample size.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3)
            >>> print(len(sampled_dataset))

        Returns:
            int: The length of the dataset.
        """
        return self.sample_size

    def _index(self) -> int:
        if len(self.indices) == 0:
            self.indices = list(range(len(self.dataset)))
            # shuffle the indices
            random.shuffle(self.indices)
        return self.indices.pop()

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3)
            >>> print(sampled_dataset.get(0))

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index >= self.sample_size or len(self.dataset) == 0:
            return None
        return self.dataset.get(self._index())


# TODO: Look at the usage of sample_size. This does not seem correct.
class SampledDatasetWithReplacement(Dataset[DatasetItem]):
    """Sample a subset of the items in the source dataset with replacement.

    This dataset allows sampling a subset of the items in the source dataset with
    replacement. The sampling lazily selects random indices from the source dataset.

    The sampling does not alter the underlying storage of the source dataset. If your
    underlying storage does not perform well with random access, you may want to
    consider performing the sampling in advance. However, for almost all use cases,
    this should not be necessary.

    Example:
        >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
        >>> sampled_dataset = SampledDatasetWithReplacement(dataset, 3)
        >>> print(sampled_dataset.get(0))

    Attributes:
        dataset (Dataset): The underlying dataset to sample from.
        sample_size (int): The size of the sample to take.
    """

    def __init__(self, dataset: Dataset[DatasetItem], sample_size: int):
        """Sample a subset of the items in the source dataset with replacement.

        This dataset allows sampling a subset of the items in the source dataset with
        replacement. The sampling lazily selects random indices from the source dataset.

        The sampling does not alter the underlying storage of the source dataset. If
        your underlying storage does not perform well with random access, you may want
        to consider performing the sampling in advance. However, for almost all use
        cases, this should not be necessary.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> sampled_dataset = SampledDatasetWithReplacement(dataset, 3)
            >>> print(sampled_dataset.get(0))

        Args:
            dataset (Dataset): The dataset to sample from.
            sample_size (int): The size of the sample to take.
        """
        self.dataset = dataset
        self.sample_size = sample_size

    def __len__(self):
        """Get the length of the dataset.

        In the case of a sampled dataset, the length is the sample size.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> sampled_dataset = SampledDatasetWithReplacement(dataset, 3)
            >>> print(len(sampled_dataset))

        Returns:
            int: The length of the dataset.
        """
        return self.sample_size

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> sampled_dataset = SampledDatasetWithReplacement(dataset, 3)
            >>> print(sampled_dataset.get(0))

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index >= self.sample_size or len(self.dataset) == 0:
            return None

        # TODO: Look at using jax PRNG random sampling
        random_index = random.randint(0, len(self.dataset) - 1)
        return self.dataset.get(random_index)
