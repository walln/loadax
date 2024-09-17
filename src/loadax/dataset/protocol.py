"""Dataset is a protocol for datasets.

Any loadax dataset must implement the Dataset protocol. This protocol defines the
basic functionality needed for a random access dataset. This includes the ability
to get an item at a given index, as well as the ability to iterate over the dataset.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

DatasetItem = TypeVar("DatasetItem", covariant=True)


class DatasetIterator(ABC, Generic[DatasetItem]):
    """DatasetIterator is a protocol for iterating over a dataset.

    A dataset iterator is responsible for iterating over a dataset. This includes
    tracking the state of the iteration.

    Example:
        ```python
        from loadax import InMemoryDataset

        dataset = InMemoryDataset([1, 2, 3, 4, 5])
        iterator = iter(dataset)
        for item in iterator:
            print(item)

        #> 1
        #> 2
        #> 3
        #> 4
        #> 5
        ```

    Attributes:
        dataset (Dataset): The dataset to iterate over.
        current (int): The current index of the iterator.
    """

    def __init__(self, dataset: "Dataset[DatasetItem]"):
        """DatasetIterator is a protocol for iterating over a dataset.

        A dataset iterator is responsible for iterating over a dataset. This includes
        tracking the state of the iteration.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            iterator = iter(dataset)
            for item in iterator:
                print(item)

            #> 1
            #> 2
            #> 3
            #> 4
            #> 5
            ```

        Args:
            dataset (Dataset): The dataset to iterate over.
        """
        self.current = 0
        self.dataset = dataset

    def __next__(self) -> DatasetItem | None:
        """Get the next item in the dataset.

        This method returns the next item in the dataset. If the iterator is at the
        end of the dataset, it returns None.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            iterator = iter(dataset)
            print(next(iterator))

            #> 1
            ```

        Returns:
            DatasetItem | None: The next item in the dataset, or None if the iterator
                is at the end of the dataset.
        """
        item = self.dataset.get(self.current)
        self.current += 1
        return item


class Dataset(ABC, Generic[DatasetItem]):
    """Dataset is a protocol for datasets.

    Any loadax dataset must implement the Dataset protocol. This protocol defines the
    basic functionality needed for a random access dataset. This includes the ability
    to get an item at a given index, as well as the ability to iterate over the dataset.

    Example:
        ```python
        from loadax import InMemoryDataset

        dataset = InMemoryDataset([1, 2, 3, 4, 5])
        print(dataset.get(0))

        #> 1
        ```

    Attributes:
        dataset (Dataset): The dataset to iterate over.
        current (int): The current index of the iterator.
    """

    def __init__(self) -> None:
        """Dataset is a protocol for datasets.

        Any loadax dataset must implement the Dataset protocol. This protocol defines
        thebasic functionality needed for a random access dataset. This includes the
        ability to get an item at a given index, as well as the ability to iterate over
        the dataset.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            print(dataset.get(0))

            #> 1
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        This method returns the item at the given index in the dataset. If the
        index is negative, it respects python's negative indexing semantics.
        If the index is out of bounds, it returns None.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            print(dataset.get(0))

            #> 1
            ```

        Args:
            index (int): The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the index is
                out of bounds.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Check if the dataset is empty.

        This method returns True if the dataset is empty, and False otherwise.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            print(dataset.is_empty())

            #> False
            ```

        Returns:
            bool: True if the dataset is empty, and False otherwise.
        """
        return len(self) == 0

    def __iter__(self) -> DatasetIterator[DatasetItem]:
        """Get an iterator for the dataset.

        This method returns an iterator for the dataset. The iterator is responsible
        for tracking the state of the iteration.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            iterator = iter(dataset)
            for item in iterator:
                print(item)

            #> 1
            #> 2
            #> 3
            #> 4
            #> 5
            ```

        Returns:
            DatasetIterator[DatasetItem]: The iterator for the dataset.
        """
        return DatasetIterator(self)

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset.

        This is just the number of elements in the dataset. This may vary depending
        on the implementation of the dataset.

        Example:
            ```python
            from loadax import InMemoryDataset

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            print(len(dataset))

            #> 5
            ```

        Returns:
            int: The length of the dataset.
        """
        raise NotImplementedError
