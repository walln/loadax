"""Dataset transformations that represent sampling items from the source dataset."""

from typing import TypeVar

import jax.random

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
        ```python
        import jax
        from loadax import SampledDatasetWithoutReplacement, InMemoryDataset

        dataset = InMemoryDataset([1, 2, 3, 4, 5])
        key = jax.random.PRNGKey(0)
        sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3, key)
        ```

    Attributes:
        dataset (Dataset): The underlying dataset to sample from.
        sample_size (int): The size of the sample to take.
        indices (list[int]): The indices to sample from.
    """

    def __init__(self, dataset: Dataset[DatasetItem], sample_size: int, key: jax.Array):
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
            ```python
            import jax
            from loadax import SampledDatasetWithoutReplacement, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3, key)
            ```

        Args:
            dataset (Dataset): The dataset to sample from.
            sample_size (int): The size of the sample to take.
            key (jax.random.KeyArray): The key to use for sampling.
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.indices: list[int] = []
        self.key = key

    def __len__(self) -> int:
        """Get the length of the dataset.

        In the case of a sampled dataset, the length is the sample size.

        Example:
            ```python
            import jax
            from loadax import SampledDatasetWithoutReplacement, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3, key)
            print(len(sampled_dataset))
            #> 3
            ```

        Returns:
            int: The length of the dataset.
        """
        return self.sample_size

    def _index(self) -> int:
        if not self.indices:
            self.key, subkey = jax.random.split(self.key)
            permuted = jax.random.permutation(subkey, len(self.dataset))
            self.indices = list(permuted[: self.sample_size])
        return self.indices.pop()

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            ```python
            import jax
            from loadax import SampledDatasetWithoutReplacement, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3, key)
            ```

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index >= self.sample_size or len(self.dataset) == 0:
            return None
        return self.dataset.get(self._index())


class SampledDatasetWithReplacement(Dataset[DatasetItem]):
    """Sample a subset of the items in the source dataset with replacement.

    This dataset allows sampling a subset of the items in the source dataset with
    replacement. The sampling lazily selects random indices from the source dataset.

    The sampling does not alter the underlying storage of the source dataset. If your
    underlying storage does not perform well with random access, you may want to
    consider performing the sampling in advance. However, for almost all use cases,
    this should not be necessary.

    Example:
        ```python
        import jax
        from loadax import SampledDatasetWithReplacement, InMemoryDataset

        dataset = InMemoryDataset([1, 2, 3, 4, 5])
        key = jax.random.PRNGKey(0)
        sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)
        ```

    Attributes:
        dataset (Dataset): The underlying dataset to sample from.
        sample_size (int): The size of the sample to take.
    """

    def __init__(self, dataset: Dataset[DatasetItem], sample_size: int, key: jax.Array):
        """Sample a subset of the items in the source dataset with replacement.

        This dataset allows sampling a subset of the items in the source dataset with
        replacement. The sampling lazily selects random indices from the source dataset.

        The sampling does not alter the underlying storage of the source dataset. If
        your underlying storage does not perform well with random access, you may want
        to consider performing the sampling in advance. However, for almost all use
        cases, this should not be necessary.

        Example:
            ```python
            import jax
            from loadax import SampledDatasetWithReplacement, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)
            ```

        Args:
            dataset (Dataset): The dataset to sample from.
            sample_size (int): The size of the sample to take.
            key (jax.random.KeyArray): The key to use for sampling.
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.key = key

    def __len__(self) -> int:
        """Get the length of the dataset.

        In the case of a sampled dataset, the length is the sample size.

        Example:
            ```python
            import jax
            from loadax import SampledDatasetWithReplacement, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)
            print(len(sampled_dataset))
            #> 3
            ```

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
            ```python
            import jax
            from loadax import SampledDatasetWithReplacement, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)
            print(sampled_dataset.get(0))
            #> 1
            ```

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index >= self.sample_size or len(self.dataset) == 0:
            return None

        self.key, subkey = jax.random.split(self.key)
        random_index = jax.random.randint(subkey, (), 0, len(self.dataset))
        return self.dataset.get(int(random_index))
