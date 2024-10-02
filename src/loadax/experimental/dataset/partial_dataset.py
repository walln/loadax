from collections.abc import Iterator
from typing import Generic

from loadax.experimental.dataset.dataset import Dataset, Example


class PartialDataset(Dataset[Example], Generic[Example]):
    """A dataset that represents a range of another dataset.

    This dataset type allows you to create a new dataset that contains
    a subset of elements from an existing dataset, specified by a start
    and end index.
    """

    def __init__(self, dataset: Dataset[Example], start: int, end: int):
        """Initialize the PartialDataset.

        Args:
            dataset: The original dataset to create a partial view of.
            start: The starting index of the range (inclusive).
            end: The ending index of the range (exclusive).
        """
        self.dataset = dataset
        self.start = start
        self.end = end

        if start < 0 or end > len(dataset) or start >= end:
            raise ValueError("Invalid start or end index")

    def __iter__(self) -> Iterator[Example]:
        """Return an iterator over the partial dataset."""
        return (self.dataset[i] for i in range(self.start, self.end))

    def __len__(self) -> int:
        """Return the number of samples in the partial dataset."""
        return self.end - self.start

    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index from the partial dataset.

        Args:
            index: The index of the example to retrieve.

        Returns:
            The data example at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        return self.dataset[self.start + index]

    @staticmethod
    def split_dataset(
        dataset: Dataset[Example], num_partitions: int
    ) -> list["PartialDataset[Example]"]:
        """Split a dataset into a number of partial datasets.

        Args:
            dataset: The original dataset to split.
            num_partitions: The number of partitions to create.

        Returns:
            A list of PartialDataset objects.

        Raises:
            ValueError: If num_partitions is less than 1 or greater than the
                dataset size.
        """
        if num_partitions < 1:
            raise ValueError("Number of partitions must be at least 1")
        if num_partitions > len(dataset):
            raise ValueError("Number of partitions cannot exceed dataset size")

        partition_size = len(dataset) // num_partitions
        remainder = len(dataset) % num_partitions

        partials = []
        start = 0
        for i in range(num_partitions):
            end = start + partition_size + (1 if i < remainder else 0)
            partials.append(PartialDataset(dataset, start, end))
            start = end

        return partials
