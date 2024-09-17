"""Partial dataset that allows accessing a subset of the dataset."""

from collections.abc import Sequence
from typing import TypeVar

from loadax.dataset import Dataset

DatasetItem = TypeVar("DatasetItem")


class PartialDataset(Dataset[DatasetItem]):
    """Create a partial subset of an existing dataset."""

    def __init__(
        self, dataset: Dataset[DatasetItem], start_index: int, end_index: int
    ) -> None:
        """Create a partial subset of an existing dataset.

        This dataset allows accessing a subset of the original dataset. The
        subset is defined by the start and end indices.

        Example:
            ```python
            from loadax import InMemoryDataset, PartialDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            partial_dataset = PartialDataset(dataset, 1, 4)
            ```

        Args:
            dataset (Dataset): The dataset to create the partial subset from.
            start_index (int): The start index of the subset.
            end_index (int): The end index of the subset.
        """
        self.dataset = dataset
        self.start_index = start_index
        self.end_index = min(end_index, len(dataset))

    @staticmethod
    def split(
        dataset: Dataset[DatasetItem], num_parts: int
    ) -> Sequence[Dataset[DatasetItem]]:
        """Split a dataset into multiple parts.

        This method splits a dataset into multiple parts of equal size. The
        number of parts is determined by the `num_parts` argument. The last
        part may contain fewer items if the dataset is not evenly divisible
        by the number of parts.

        Example:
            ```python
            from loadax import InMemoryDataset, PartialDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            partial_datasets = PartialDataset.split(dataset, 3)

            for partial_dataset in partial_datasets:
                print(partial_dataset.get(0))

            #> 1
            #> 3
            #> 5
            ```

        Args:
            dataset (Dataset): The dataset to split.
            num_parts (int): The number of parts to split the dataset into.

        Returns:
            list[Dataset[DatasetItem]]: A list of partial datasets.
        """
        total_len = len(dataset)
        batch_size = total_len // num_parts
        remainder = total_len % num_parts
        datasets: list[Dataset[DatasetItem]] = []

        current = 0
        for i in range(num_parts):
            start = current
            end = start + batch_size + (1 if i < remainder else 0)
            datasets.append(PartialDataset[DatasetItem](dataset, start, end))
            current = end

        return datasets

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            ```python
            from loadax import InMemoryDataset, PartialDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            partial_dataset = PartialDataset(dataset, 1, 4)
            print(partial_dataset.get(0))

            #> 2
            ```

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            return None
        return self.dataset.get(index + self.start_index)

    def __len__(self) -> int:
        """Get the length of the dataset.

        In a partial dataset, the length is number of items between the start
        and end indices.

        Example:
            ```python
            from loadax import InMemoryDataset, PartialDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            partial_dataset = PartialDataset(dataset, 1, 4)
            print(len(partial_dataset))

            #> 3
            ```

        Returns:
            int: The length of the dataset.
        """
        return max(0, self.end_index - self.start_index)
