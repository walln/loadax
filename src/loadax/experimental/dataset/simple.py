from collections.abc import Iterator
from typing import Generic

from loadax.experimental.dataset.dataset import Dataset, Example


class SimpleDataset(Dataset[Example], Generic[Example]):
    """A dataset that wraps a list of examples.

    Args:
        data (List[Example]): The list of data examples.
    """

    def __init__(self, data: list[Example]):
        """Initialize a simple dataset in-memory from a list.

        Args:
            data (List[Example]): The list of data examples.
        """
        self.data = data

    def __iter__(self) -> Iterator[Example]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index.

        Args:
            index (int): The index of the example to retrieve.

        Returns:
            Example: The data example at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.data[index]
