"""InMemoryDataset is a simple in-memory dataset."""

from typing import TypeVar

from loadax.dataset.protocol import Dataset

DatasetItem = TypeVar("DatasetItem")


class InMemoryDataset(Dataset[DatasetItem]):
    """InMemoryDataset is a simple in-memory dataset.

    This dataset stores all underlying items in a list in memory. This is a simple
    implementation, in your training loop if you are using non-trivial dataset
    it is better to use another dataset. This dataset type does not get much benefit
    from dataloaders, but it is useful for debugging and for simple use cases.

    Example:
        >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
        >>> print(dataset.get(0))

    Attributes:
        items (list[DatasetItem]): The underlying items in the dataset.
    """

    def __init__(self, items: list[DatasetItem]):
        """InMemoryDataset is a simple in-memory dataset.

        This dataset stores all underlying items in a list in memory. This is a simple
        implementation, in your training loop if you are using non-trivial dataset
        it is better to use another dataset. This dataset type does not get much benefit
        from dataloaders, but it is useful for debugging and for simple use cases.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> print(dataset.get(0))

        Args:
            items (list[DatasetItem]): The underlying items in the dataset.
        """
        self.items = items

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> print(dataset.get(0))

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index >= len(self.items):
            return None

        if index < 0:
            index = len(self.items) + index
        return self.items[index]

    def __len__(self) -> int:
        """Get the length of the dataset.

        In the case of an in-memory dataset, the length is the number of items in
        the dataset.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> print(len(dataset))

        Returns:
            int: The length of the dataset.
        """
        return len(self.items)

    def __repr__(self) -> str:
        """Get a string representation of the dataset.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> print(repr(dataset))

        Returns:
            str: The string representation of the dataset.
        """
        return f"InMemoryDataset(items={self.items[:2]}...)"

    # TODO: implement from different file types, ex: from_csv, from_json, from_parquet
