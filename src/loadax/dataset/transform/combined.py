"""Combine multiple datsets into a single dataset."""

from typing import TypeVar

from loadax.dataset.protocol import Dataset

DatasetItem = TypeVar("DatasetItem")


class CombinedDataset(Dataset[DatasetItem]):
    """Combine multiple datasets into a single dataset."""

    def __init__(self, datasets: list[Dataset[DatasetItem]]):
        """Combine multiple datasets into a single dataset.

        This is a simple concatenation of the datasets. The datasets are
        concatenated in the order they are passed to the constructor. If the
        dataset elements need to be interleaved, you can either use a `ShuffledDataset`,
        `SampledDatasetWithoutReplacement` or `SampledDatasetWithReplacement`.

        Example:
            ```python
            from loadax import CombinedDataset, InMemoryDataset

            dataset1 = InMemoryDataset([1, 2, 3])
            dataset2 = InMemoryDataset([4, 5, 6])
            combined_dataset = CombinedDataset([dataset1, dataset2])
            ```

        Args:
            datasets (list[Dataset]): The datasets to combine.
        """
        self.datasets = datasets

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        Example:
            ```python
            from loadax import CombinedDataset, InMemoryDataset

            dataset1 = InMemoryDataset([1, 2, 3])
            dataset2 = InMemoryDataset([4, 5, 6])
            combined_dataset = CombinedDataset([dataset1, dataset2])
            print(combined_dataset.get(0))
            #> 1
            ```

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        current_index = 0
        for dataset in self.datasets:
            if index < current_index + len(dataset):
                return dataset.get(index - current_index)
            current_index += len(dataset)
        return None

    def __len__(self) -> int:
        """Get the length of the dataset.

        In the case of a combined dataset, the length is the sum of the lengths
        of the individual datasets.

        Example:
            ```python
            from loadax import CombinedDataset, InMemoryDataset

            dataset1 = InMemoryDataset([1, 2, 3])
            dataset2 = InMemoryDataset([4, 5, 6])
            combined_dataset = CombinedDataset([dataset1, dataset2])
            print(len(combined_dataset))
            #> 6
            ```

        Returns:
            int: The length of the dataset.
        """
        return sum(len(dataset) for dataset in self.datasets)

    def __repr__(self) -> str:
        """Get a string representation of the dataset.

        Example:
            ```python
            from loadax import CombinedDataset, InMemoryDataset

            dataset1 = InMemoryDataset([1, 2, 3])
            dataset2 = InMemoryDataset([4, 5, 6])
            combined_dataset = CombinedDataset([dataset1, dataset2])
            print(repr(combined_dataset))
            #> CombinedDataset(datasets=[InMemoryDataset(items=[1, 2, 3]),
                                         InMemoryDataset(items=[4, 5, 6])])
            ```

        Returns:
            str: The string representation of the dataset.
        """
        return f"CombinedDataset(datasets={self.datasets})"
