"""A naive dataloader that does not offload work to background processes."""

from typing import Generic, TypeVar

from loadax.batcher import Batcher
from loadax.dataloader.progress import Progress
from loadax.dataloader.protocol import DataloaderIteratorProtocol, DataloaderProtocol
from loadax.dataset import Dataset
from loadax.strategy import BatchStrategy

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


class NaiveDataloaderIterator(DataloaderIteratorProtocol[DatasetItem, Batch]):
    """An iterator over a naive dataloader.

    This iterator is stateful and blocks the calling thread until the next batch is
    available.

    Attributes:
        dataloader: The naive dataloader.
        current_index: The current index of the iterator.
    """

    def __init__(self, dataloader: "NaiveDataloader[DatasetItem,Batch]"):
        """An iterator over a naive dataloader.

        This iterator is stateful and blocks the calling thread until the next batch is
        available.

        Args:
            dataloader: The naive dataloader.
        """
        self.dataloader = dataloader
        self.current_index = 0

    def __next__(self) -> Batch:
        """Get the next batch.

        This method returns the next batch of data. It is a blocking call that will
        block until the next batch is available.

        Returns:
            The next batch of data.
        """
        while True:
            item = self.dataloader.dataset.get(self.current_index)
            if item is None:
                break

            self.current_index += 1
            self.dataloader.strategy.add(item)

            items = self.dataloader.strategy.batch(force=False)
            if items is not None:
                return self.dataloader.batcher.batch(items)

        items = self.dataloader.strategy.batch(force=True)
        if items is not None:
            return self.dataloader.batcher.batch(items)

        raise StopIteration

    def __iter__(self) -> "NaiveDataloaderIterator[DatasetItem, Batch]":
        """Get an iterator over the dataloader.

        This method returns an iterator over the dataloader. The iterator is stateful
        and creating a new iterator will not share the same state, such as the current
        index.

        Returns:
            An iterator over the dataloader.
        """
        return self

    def progress(self) -> Progress:
        """Get the progress of the dataloader.

        This method returns metadata about the iteration progress. This is useful for
        debugging and monitoring the iteration.

        Returns:
            The progress of the dataloader.
        """
        return Progress(self.current_index, len(self.dataloader.dataset))


class NaiveDataloader(
    DataloaderProtocol[DatasetItem, Batch], Generic[DatasetItem, Batch]
):
    """A naive dataloader that does not offload work to background processes.

    This dataloader is a simple implementation of a dataloader that does not offload
    work to background processes. It is useful for debugging and testing. This will
    slow down your training loop, but is useful for debugging and really simple
    training loops.

    Attributes:
        dataset: The dataset to load.
        batcher: The batcher to use to collate the data.
        strategy: The batch strategy to use to determine how much data goes into a
            batch.
    """

    def __init__(
        self,
        dataset: Dataset[DatasetItem],
        batcher: Batcher[DatasetItem, Batch],
        strategy: BatchStrategy[DatasetItem],
    ):
        """A naive dataloader that does not offload work to background processes.

        This dataloader is a simple implementation of a dataloader that does not
        offload work to background processes. It is useful for debugging and testing.
        This will slow down your training loop, but is useful for debugging and really
        simple training loops.

        Args:
            dataset: The dataset to load.
            batcher: The batcher to use to collate the data.
            strategy: The batch strategy to use to determine how much data goes into a
                batch.
        """
        self.dataset = dataset
        self.batcher = batcher
        self.strategy = strategy

    def num_items(self) -> int:
        """Get the number of items in the dataset.

        This method returns the number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return len(self.dataset)

    def num_batches(self) -> int:
        """Get the number of batches in the dataset.

        This method returns the number of batches in the dataset based
        on your batch strategy.

        Returns:
            The number of batches in the dataset.
        """
        return self.num_items() // self.strategy.batch_size

    def __len__(self) -> int:
        """Get the length of the dataloader.

        This method returns the length of the dataloader based on the number of batches
        that will be returned.

        Returns:
            The length of the dataloader.
        """
        return self.num_batches()

    def __iter__(self) -> DataloaderIteratorProtocol[DatasetItem, Batch]:
        """Get an iterator over the dataloader.

        This method returns an iterator over the dataloader. The iterator is stateful
        and creating a new iterator will not share the same state, such as the current
        index.

        Returns:
            An iterator over the dataloader.
        """
        return NaiveDataloaderIterator(self)
