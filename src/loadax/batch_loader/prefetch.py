"""Prefetch loader extends the single batch loader to prefetch data in the background."""

from loadax.batch_loader.single import BatchDataLoader
from loadax.dataset.dataset import Dataset
from loadax.strategy import BatchStrategy
from loadax.batcher import Batcher
import queue


class PrefetchBatchLoader(BatchDataLoader):
    """Prefetch loader extends the single batch loader to prefetch future batches in the background."""

    def __init__(
        self,
        dataset: Dataset,
        batcher: Batcher,
        strategy: BatchStrategy,
        prefetch_factor=2,
    ):
        super().__init__(dataset, batcher, strategy)
        self.prefetch_factor = prefetch_factor
        self.data_queue = queue.Queue(maxsize=prefetch_factor)
        self.index = 0
        self.prefetch(self.index)

    def prefetch(self, index: int):
        end_index = min(index + self.prefetch_factor, len(self.dataset))
        for i in range(index, end_index):
            self.data_queue.put(self.dataset.get(i))

    def get_iterator(self, start_index=0):
        return PrefetchBatchIterator(self, start_index)


class PrefetchBatchIterator(object):
    def __init__(self, data_loader: PrefetchBatchLoader, start_index: int = 0):
        self.data_loader = data_loader
        self.index = start_index

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.index >= len(self.data_loader.dataset)
            and self.data_loader.data_queue.empty()
        ):
            raise StopIteration

        batch = []
        while len(batch) < self.data_loader.strategy.batch_size:
            try:
                data = self.data_loader.data_queue.get(timeout=5)
                self.data_loader.strategy.add(data)
                self.index += 1
                if self.index < len(self.data_loader.dataset):
                    self.data_loader.prefetch(self.index)
            except queue.Empty:
                if (
                    self.index >= len(self.data_loader.dataset)
                    and self.data_loader.data_queue.empty()
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
        self.data_loader.data_queue.queue.clear()
