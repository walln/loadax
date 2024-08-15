import itertools
import multiprocessing
from loadax.dataloader.naive import NaiveDataLoader
from loadax.batcher import Batcher
from loadax.dataloader.protocol import DataLoaderIteratorProtocol, Progress
from loadax.dataset import Dataset
from loadax.strategy import BatchStrategy
import queue
from typing import TypeVar

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


def worker_fn(dataset: Dataset, index_queue: queue.Queue, data_queue: queue.Queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        data_queue.put((index, dataset.get(index)))


class MultiProcessingDataLoaderIterator(DataLoaderIteratorProtocol[Batch]):
    def __init__(self, dataloader: "MultiProcessingDataLoader[DatasetItem, Batch]"):
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

        self.prefetch()

    def prefetch(self):
        while (
            self.prefetch_index < len(self.dataloader.dataset)
            and self.prefetch_index
            < self.current_index
            + self.dataloader.prefetch_factor
            * self.dataloader.num_workers
            * self.dataloader.strategy.batch_size
        ):
            # if the prefetch index hasnt reached the end of the dataset and it is not 2 batches ahead of the current index, we can add more indexes to be prefetched
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def __next__(self):
        self.prefetch()
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
        self.current_index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def __del__(self):
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
        return Progress(self.current_index, len(self.dataset))


class MultiProcessingDataLoader(NaiveDataLoader[DatasetItem, Batch]):
    def __init__(
        self,
        dataset: Dataset[DatasetItem],
        batcher: Batcher[DatasetItem, Batch],
        strategy: BatchStrategy[DatasetItem],
        num_workers: int,
        prefetch_factor: int,
    ):
        super().__init__(dataset, batcher, strategy)

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def __iter__(self):
        return MultiProcessingDataLoaderIterator(self)
