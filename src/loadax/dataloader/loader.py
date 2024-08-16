"""Dataloader that leverage multiprocessing for performance."""

import itertools
import multiprocessing
import queue
from typing import TypeVar

from loadax.batcher import Batcher
from loadax.dataloader.naive import NaiveDataLoader
from loadax.dataloader.protocol import DataLoaderIteratorProtocol, Progress
from loadax.dataset import Dataset
from loadax.strategy import BatchStrategy

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


def worker_fn(dataset: Dataset, index_queue: queue.Queue, data_queue: queue.Queue):
    """Worker function for multiprocessing dataloader.

    This function continuously fetches indexes from the index queue, when it receives an
    index, it fetches the corresponding data from the dataset and puts it in the data
    queue. This function runs in a separate process and passses data through queues
    as to not acquite the GIL.

    Args:
        dataset (Dataset): The dataset to fetch data from.
        index_queue (queue.Queue): The queue to fetch indexes from.
        data_queue (queue.Queue): The queue to put data in.
    """
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        data_queue.put((index, dataset.get(index)))


class MultiProcessingDataLoaderIterator(DataLoaderIteratorProtocol[Batch]):
    """Iterator for the multiprocessing dataloader."""

    def __init__(self, dataloader: "MultiProcessingDataLoader[DatasetItem, Batch]"):
        """Iterator for the dataloader.

        A dataloader iterator is stateful and only maintains a reference to the
        the dataloader instance. This means that you must use the same iterator
        instance if you wish to preserve the iteration state.

        This should not be manually instantiated. Instead, use the `iter` method
        of the dataloader to get an iterator.

        Example:
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            dataloader (MultiProcessingDataLoader): The dataloader to iterate over.
        """
        self.current_index = 0
        self.dataloader = dataloader

        self.index_queues = []
        self.data_queue = multiprocessing.Queue()
        self.workers: list[multiprocessing.Process] = []
        self.worker_cycle = itertools.cycle(range(self.dataloader.num_workers))
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(self.dataloader.num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn,
                args=(self.dataloader.dataset, index_queue, self.data_queue),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self._prefetch()

    def _prefetch(self):
        while (
            self.prefetch_index < len(self.dataloader.dataset)
            and self.prefetch_index
            < self.current_index
            + self.dataloader.prefetch_factor
            * self.dataloader.num_workers
            * self.dataloader.strategy.batch_size
        ):
            # if the prefetch index hasnt reached the end of the dataset and it is
            # not 2 batches ahead of the current index, we can add more indexes to
            # be prefetched
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def __next__(self):
        """Get the next batch from the dataloader.

        This method will pull the next batch from the cache and replenish
        it if necessary. If the batch is the last batch in the dataset, it
        will return the last batch and then raise `StopIteration` upon the
        next call to `__next__`.

        It is important to note that the dataloader is offloading the work
        of prefetching and batching to the workers. This means that the
        `__next__` method can spinlock while waiting for the next batch to
        be available. If you are experiencing performance issues, you may
        want to increase the `prefetch_factor` or `num_workers`.

        Example:
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Raises:
            StopIteration: If the dataset is exhausted.

        Returns:
            Batch: The next batch from the dataloader.
        """
        self._prefetch()
        while self.current_index < len(self.dataloader.dataset):
            if self.current_index in self.cache:
                item = self.cache[self.current_index]
                del self.cache[self.current_index]
            else:
                while True:
                    try:
                        index, data = self.data_queue.get(timeout=0)
                    except queue.Empty:
                        continue
                    if index == self.current_index:
                        item = data
                        break
                    else:
                        self.cache[index] = data

            self.current_index += 1
            self.dataloader.strategy.add(item)

            items = self.dataloader.strategy.batch(False)
            if items is not None:
                return self.dataloader.batcher.batch(items)

        # Check for remaining items if the batch is not full
        items = self.dataloader.strategy.batch(True)
        if items is not None:
            return self.dataloader.batcher.batch(items)

        raise StopIteration

    def __iter__(self):
        """Get an iterator for the dataloader.

        This method returns an iterator that can be used to iterate over the
        dataloader. The iterator will automatically handle prefetching and
        batching, and will return the last batch if the dataset is exhausted.

        Example:
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Note that the iterator is stateful and creating a new iterator on the
        same dataloader will not preserve the iteration state.

        Returns:
            MultiProcessingDataLoaderIterator: The iterator for the dataloader.
        """
        self.current_index = 0
        self.cache = {}
        self.prefetch_index = 0
        self._prefetch()
        return self

    def __del__(self):
        """Clean up the dataloader.

        This method will terminate all the workers and close all the queues.
        It is important to call this method when you are done with the dataloader
        to ensure that all resources are properly cleaned up.

        This method should not be called manually and will be called automatically
        when the iterator is garbage collected.
        """
        try:
            # stop each worker by sending None to their index queue
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                # close all the queues
                q.cancel_join_thread()
                q.close()
            self.data_queue.cancel_join_thread()
            self.data_queue.close()
        finally:
            for w in self.workers:
                # manually terminate the workers no matter what
                if w.is_alive():
                    w.terminate()

    def progress(self) -> Progress:
        """Get the progress of the dataloader.

        This method returns a `Progress` object that contains the current
        index and the total number of items in the dataset. This can be used
        to track the progress of the dataloader.

        Example:
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)
            >>> progress = dataloader.progress()
            >>> print(progress.current_index)
            >>> print(progress.total_items)

        Returns:
            Progress: The progress of the dataloader.
        """
        return Progress(self.current_index, len(self.dataset))


class MultiProcessingDataLoader(NaiveDataLoader[DatasetItem, Batch]):
    """Dataloader that leverages multiprocessing for performance.

    This dataloader is designed to not block the main thread while loading data
    and enables efficient parallel data loading and prefetching.

    Example:
        >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
        >>> iterator = iter(dataloader)
        >>> for batch in iterator:
        ...     print(batch)
    """

    def __init__(
        self,
        dataset: Dataset[DatasetItem],
        batcher: Batcher[DatasetItem, Batch],
        strategy: BatchStrategy[DatasetItem],
        num_workers: int,
        prefetch_factor: int,
    ):
        """A dataloader that leverages multiprocessing for performance.

        This dataloader is designed to not block the main thread while loading data
        and enables efficient parallel data loading and prefetching.

        Args:
            dataset (Dataset): The dataset to load data from.
            batcher (Batcher): The batcher to use for batching.
            strategy (BatchStrategy): The batch strategy to use for prefetching.
            num_workers (int): The number of workers to use for parallel data loading.
            prefetch_factor (int): The prefetch factor to use for prefetching.
        """
        super().__init__(dataset, batcher, strategy)

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def __iter__(self):
        """Get an iterator for the dataloader.

        This method returns an iterator that can be used to iterate over the
        dataloader. The iterator is stateful and creating a new iterator on the
        same dataloader will not preserve the iteration state.

        Example:
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Returns:
            MultiProcessingDataLoaderIterator: The iterator for the dataloader.
        """
        return MultiProcessingDataLoaderIterator(self)
