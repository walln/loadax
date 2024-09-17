"""Dataset that applies a function to the items of another dataset."""

from collections.abc import Callable
from typing import TypeVar

from loadax.dataset.protocol import Dataset

DatasetItem = TypeVar("DatasetItem")
MappedItem = TypeVar("MappedItem")


class MappedDataset(Dataset[MappedItem]):
    """Dataset that applies a function to the items of another dataset.

    This dataset transformation lazily applies a function to the items of another
    dataset. This is useful for cases where you need to augment the data before using
    it in a training loop.

    Example:
        ```python
        from loadax.dataset import RangeDataset
        from loadax.dataset.transform import MappedDataset

        def slow_fn(x):
            time.sleep(0.1)
            return x * 2

        dataset = MappedDataset(RangeDataset(0, 20), slow_fn)
        ```

    Attributes:
        dataset: The dataset to apply the function to.
        map_fn: The function to apply to the items of the dataset.
    """

    def __init__(
        self, dataset: Dataset[DatasetItem], map_fn: Callable[[DatasetItem], MappedItem]
    ):
        """Dataset that applies a function to the items of another dataset.

        This dataset transformation lazily applies a function to the items of another
        dataset. This is useful for cases where you need to augment the data before
        usingit in a training loop.

        Example:
            ```python
            from loadax.dataset import RangeDataset
            from loadax.dataset.transform import MappedDataset

            def slow_fn(x):
                time.sleep(0.1)
                return x * 2

            dataset = MappedDataset(RangeDataset(0, 20), slow_fn)
            ```

        Args:
            dataset: The dataset to apply the function to.
            map_fn: The function to apply to the items of the dataset.
        """
        self.dataset = dataset
        self.map_fn = map_fn

    def get(self, index: int) -> MappedItem | None:
        """Get the item at the given index.

        Args:
            index: The index of the item to get.

        Returns:
            The item at the given index, or None if the index is out of range.
        """
        item = self.dataset.get(index)
        if item is None:
            return None
        return self.map_fn(item)

    def __len__(self) -> int:
        """Get the length of the dataset.

        In the case of a MappedDataset, the length of the dataset is the same as the
        length of the underlying dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.dataset)

    def __repr__(self) -> str:
        """Get a string representation of the dataset.

        Returns:
            A string representation of the dataset.
        """
        return f"MappedDataset(dataset={self.dataset}, map_fn={self.map_fn})"
