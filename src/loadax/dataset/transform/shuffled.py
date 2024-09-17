"""Shuffled dataset that shuffles the items in the source dataset."""

from typing import TypeVar

import jax
import jax.random

from loadax.dataset.protocol import Dataset

DatasetItem = TypeVar("DatasetItem")


class ShuffledDataset(Dataset[DatasetItem]):
    """Shuffle the items in the source dataset.

    The shuffling is performed lazily and does not actually shuffle the underlying
    storage. If your underlying storage does not perform well with random access,
    you may want to consider performing the shuffling in advance. However, for
    almost all use cases, this should not be necessary.

    The lazy shuffling enables  better reproducibility and determinism, while also
    being extremely efficient as only the indices need to be shuffled. This also
    means that shuffling is not IO bound as the indices can be stored in memory.

    Example:
        ```python
        import jax
        from loadax import ShuffledDataset, InMemoryDataset

        dataset = InMemoryDataset([1, 2, 3, 4, 5])
        key = jax.random.PRNGKey(0)
        shuffled_dataset = ShuffledDataset(dataset, key)
        ```

    Attributes:
        dataset (Dataset): The underlying dataset to shuffle.
        indices (list[int]): The indices to shuffle.
    """

    def __init__(self, dataset: Dataset[DatasetItem], key: jax.Array):
        """Shuffle the items in the source dataset.

        The shuffling is performed lazily and does not actually shuffle the underlying
        storage. If your underlying storage does not perform well with random access,
        you may want to consider performing the shuffling in advance. However, for
        almost all use cases, this should not be necessary.

        The lazy shuffling enables better reproducibility and determinism, while also
        being extremely efficient as only the indices need to be shuffled. This also
        means that shuffling is not IO bound as the indices can be stored in memory.

        Example:
            ```python
            import jax
            from loadax import ShuffledDataset, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            shuffled_dataset = ShuffledDataset(dataset, key)
            ```

        Args:
            dataset (Dataset): The dataset to shuffle.
            key (jax.random.KeyArray): The key to use for shuffling.
        """
        self.dataset = dataset
        self.key = key
        self.indices = list(jax.random.permutation(self.key, len(dataset)))

    def __len__(self) -> int:
        """Get the length of the dataset.

        In the case of a shuffled dataset, the length is the length of the underlying
        dataset.

        Example:
            ```python
            import jax
            from loadax import ShuffledDataset, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            shuffled_dataset = ShuffledDataset(dataset, key)
            print(len(shuffled_dataset))
            #> 5
            ```

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            ```python
            import jax
            from loadax import ShuffledDataset, InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            key = jax.random.PRNGKey(0)
            shuffled_dataset = ShuffledDataset(dataset, key)
            print(shuffled_dataset.get(0))
            #> 3
            ```

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index < 0 or index >= len(self.indices):
            return None
        return self.dataset.get(self.indices[index])

    def __repr__(self) -> str:
        """Get a string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return f"ShuffledDataset(dataset={self.dataset})"
