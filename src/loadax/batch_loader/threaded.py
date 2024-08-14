import threading
from typing import Generic, TypeVar
import queue

from loadax.batcher import Batcher
from loadax.dataloader import Progress
from loadax.dataset.dataset import Dataset
from loadax.strategy import BatchStrategy

T = TypeVar("T")
P = TypeVar("P")


class ThreadedDataLoaderIterator(Generic[T, P]):
    def __init__(self, data_loader: "ThreadedDataLoader", start_index: int = 0):
        self.data_loader = data_loader
        self.index = start_index
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue(
            maxsize=self.data_loader.prefetch_factor
            * self.data_loader.strategy.batch_size
        )
        self.index_queue = queue.Queue()
        self.lock = threading.Lock()

        self.strategy = self.data_loader.strategy.clone()

        self.workers = [
            threading.Thread(target=self._worker)
            for _ in range(self.data_loader.num_workers)
        ]
        for worker in self.workers:
            worker.start()

        self.prefetch(self.index)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                index = self.index_queue.get(timeout=1)
            except queue.Empty:
                continue
            if self.stop_event.is_set():
                break
            data = self.data_loader.dataset.get(index)
            self.data_queue.put(data)

    def prefetch(self, index):
        while not self.stop_event.is_set():
            with self.lock:
                current_prefetch_size = self.data_queue.qsize()
            if current_prefetch_size >= self.data_queue.maxsize:
                break

            if index >= len(self.data_loader.dataset):
                break

            self.index_queue.put(index)
            index += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data_loader.dataset) and self.data_queue.empty():
            self.stop_event.set()
            for worker in self.workers:
                worker.join()
            raise StopIteration

        batch = []
        while len(batch) < self.strategy.batch_size:
            try:
                data = self.data_queue.get(timeout=5)
                self.strategy.add(data)
                with self.lock:
                    self.index += 1
                    self.prefetch(self.index)
            except queue.Empty:
                if (
                    self.index >= len(self.data_loader.dataset)
                    and self.data_queue.empty()
                ):
                    break
                continue

            new_batch = self.strategy.batch(force=False)
            if new_batch:
                batch.extend(new_batch)

        if len(batch) < self.strategy.batch_size:
            final_batch = self.strategy.batch(force=True)
            if final_batch:
                batch.extend(final_batch)

        if not batch:
            raise StopIteration

        return self.data_loader.batcher(batch)

    def __del__(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join()

    def progress(self) -> Progress:
        with self.lock:
            return Progress(self.index, len(self.data_loader.dataset))


class ThreadedDataLoader(Generic[T, P]):
    def __init__(
        self,
        dataset: Dataset,
        num_workers: int,
        prefetch_factor: int,
        strategy: BatchStrategy[T],
        batcher: Batcher[T, P],
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.strategy = strategy
        self.batcher = batcher

    def __iter__(self):
        return ThreadedDataLoaderIterator(self)

    def get_iterator(self, start_index: int = 0):
        return ThreadedDataLoaderIterator(self, start_index)
