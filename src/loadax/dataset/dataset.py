from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

Example = TypeVar("Example")
Transformed = TypeVar("Transformed")


class Dataset(ABC, Generic[Example]):
    """Dataset is the basic protocol that loadax needs to support.

    Any random-access dataset can be used with loadax as long as its size
    can be known and elements can be retrieved by index.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        """Return an iterator over the dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index."""
        pass

    def map(
        self, transform: Callable[[Example], Transformed]
    ) -> "Dataset[Transformed]":
        """Apply a transformation to each element in the dataset."""
        return MappedDataset(self, transform)

    def filter(self, predicate: Callable[[Example], bool]) -> "Dataset[Example]":
        """Filter elements in the dataset based on a predicate."""
        return FilteredDataset(self, predicate)

    def map_batch(
        self, transform: Callable[[list[Example]], Transformed], batch_size: int = 32
    ) -> "Dataset[Transformed]":
        """Apply a transformation to batches of elements in the dataset."""
        return MappedBatchDataset(self, transform, batch_size)


class MappedDataset(Dataset[Transformed], Generic[Example, Transformed]):
    """A dataset that applies a transformation to each element in the dataset.

    The transformation is lazily applied, this means that the underlying data
    is not altered and instead is only applied when iterated over.
    """

    def __init__(
        self, dataset: Dataset[Example], transform: Callable[[Example], Transformed]
    ):
        """Intializes the MappedDataset.

        Args:
            dataset: The underlying dataset to apply the transformation to.
            transform: The transformation to apply to each element in the dataset.
        """
        self.dataset = dataset
        self.transform = transform

    def __iter__(self) -> Iterator[Transformed]:
        for example in self.dataset:
            yield self.transform(example)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Transformed:
        example = self.dataset[index]
        return self.transform(example)


class FilteredDataset(Dataset[Example], Generic[Example]):
    """A dataset that filters elements based on a predicate.

    The predicate is lazily applied, in the case of large datasets
    it is recommended to filter the dataset using eager data cleaning
    and preprocessing techniques.
    """

    def __init__(self, dataset: Dataset[Example], predicate: Callable[[Example], bool]):
        """Intializes the FilteredDataset.

        Args:
            dataset: The underlying dataset to filter.
            predicate: The predicate to filter the dataset.
        """
        self.dataset = dataset
        self.predicate = predicate
        self.length: int | None = None  # Cache for length

    def __iter__(self) -> Iterator[Example]:
        for example in self.dataset:
            if self.predicate(example):
                yield example

    def __len__(self) -> int:
        length = self.length or sum(
            1 for example in self.dataset if self.predicate(example)
        )
        assert length is not None, "Length must be set."
        self.length = length
        return length

    def __getitem__(self, index: int) -> Example:
        # To support indexed access, precompute filtered indices
        if not hasattr(self, "_filtered_indices"):
            self._filtered_indices = [
                i for i, example in enumerate(self.dataset) if self.predicate(example)
            ]
        actual_index = self._filtered_indices[index]
        return self.dataset[actual_index]


class MappedBatchDataset(Dataset[Transformed], Generic[Example, Transformed]):
    """Performs element transformations in batches.

    Just as with MappedDataset, the transformation is lazily applied, this means
    that the underlying data is not altered and instead is only applied when iterated
    over.

    Batched-mapping is useful when you want to apply a transformation that is
    particularly expensive to apply to a large number of elements. For example, if
    you have a dataset that needs to be tokenized, you can apply the tokenization
    to each batch of elements in the dataset to avoid the overhead of tokenizing
    each element individually.
    """

    def __init__(
        self,
        dataset: Dataset[Example],
        transform: Callable[[list[Example]], Transformed],
        batch_size: int = 32,
    ):
        """Intializes the MappedBatchDataset.

        Args:
            dataset: The underlying dataset to apply the transformation to.
            transform: The transformation to apply to each batch of elements in the
                dataset.
            batch_size: The size of each batch.
        """
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Transformed]:
        batch = []
        for example in self.dataset:
            batch.append(example)
            if len(batch) == self.batch_size:
                yield self.transform(batch)
                batch = []
        if batch:
            yield self.transform(batch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> Transformed:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.dataset))
        batch = [self.dataset[i] for i in range(start, end)]
        return self.transform(batch)
