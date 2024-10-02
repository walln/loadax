from collections.abc import Iterator
from typing import Generic

from loadax.experimental.dataset.dataset import Dataset, Example


class CombinedDataset(Dataset[Example], Generic[Example]):
    """A dataset that combines two datasets sequentially.

    This dataset type allows you to concatenate two datasets, creating a new dataset
    that contains all elements from the first dataset followed by all elements from
    the second dataset.
    """

    def __init__(self, dataset1: Dataset[Example], dataset2: Dataset[Example]):
        """Initialize the CombinedDataset.

        Args:
            dataset1: The first dataset to be combined.
            dataset2: The second dataset to be combined.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __iter__(self) -> Iterator[Example]:
        """Return an iterator over the combined dataset."""
        yield from self.dataset1
        yield from self.dataset2

    def __len__(self) -> int:
        """Return the total number of samples in the combined dataset."""
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index from the combined dataset.

        Args:
            index: The index of the example to retrieve.

        Returns:
            The data example at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        if index < len(self.dataset1):
            return self.dataset1[index]
        else:
            return self.dataset2[index - len(self.dataset1)]
