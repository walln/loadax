"""The dataloader protocol defines the interface that a loadax dataloader implements."""

from typing import Generic, Protocol, TypeVar

from loadax.dataloader.progress import Progress

DatasetItem = TypeVar("DatasetItem", covariant=True)
Batch = TypeVar("Batch", covariant=True)


class DataloaderIteratorProtocol(Protocol[DatasetItem, Batch]):
    """The iterator protocol for a dataloader.

    This protocol defines the interface for iterating over a dataloader. It is
    implemented by the dataloader itself, is a stateful indicator of the progress
    of iteration.

    Attributes:
        progress: The progress of the dataloader.
    """

    def progress(self) -> Progress:
        """Get the progress of the dataloader.

        This method returns metadata about the iteration progress. This is useful for
        debugging and monitoring the iteration.

        Returns:
            The progress of the dataloader.
        """
        ...

    def __next__(self) -> Batch:
        """Get the next batch.

        This method returns the next batch of data. It is a blocking call that will
        block until the next batch is available.

        Returns:
            The next batch of data.
        """
        ...


class DataloaderProtocol(Protocol, Generic[DatasetItem, Batch]):
    """The dataloader protocol is the interface that a loadax dataloader implements.

    A loadax dataloader is mostly just configuration of how to create an iterator over
    the dataset. The iterator itself is stateful and implements the iterator protocol.

    Attributes:
        num_items: The number of items in the dataset.
    """

    def __iter__(self) -> DataloaderIteratorProtocol[DatasetItem, Batch]:
        """Get an iterator over the dataset.

        This method returns an iterator over the dataset. The iterator is stateful and
        implements the iterator protocol.

        Returns:
            An iterator over the dataset.
        """
        raise NotImplementedError

    def num_items(self) -> int:
        """Get the number of items in the dataset.

        This method returns the number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        raise NotImplementedError
