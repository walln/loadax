from typing import TypeVar
from loadax.strategy import BatchStrategy
from loadax.batcher import Batcher
from loadax.dataloader import DataLoader, DataLoaderIterator, Progress
from loadax.dataset import Dataset

T = TypeVar("T")


class BatchDataLoaderIterator(DataLoaderIterator[T]):
    def __init__(self, dataset: Dataset, strategy: BatchStrategy, batcher: Batcher):
        self.dataset = dataset
        self.strategy = strategy
        self.batcher = batcher
        self.current_index = 0

    def __next__(self) -> T | None:
        while True:
            item = self.dataset.get(self.current_index)
            if item is None:
                break

            self.current_index += 1
            self.strategy.add(item)

            items = self.strategy.batch(False)
            if items is not None:
                # print(f"Batching items: {items} current index: {self.current_index}")
                return self.batcher.batch(items)

        items = self.strategy.batch(True)
        if items is not None:
            return self.batcher.batch(items)

        raise StopIteration

    def __iter__(self):
        return self

    def progress(self) -> Progress:
        return Progress(self.current_index, len(self.dataset))


class BatchDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batcher: Batcher,
        strategy: BatchStrategy,
    ):
        self.dataset = dataset
        self.batcher = batcher
        self.strategy = strategy

    def num_items(self) -> int:
        return len(self.dataset)

    def num_batches(self) -> int:
        return self.num_items() // self.strategy.batch_size

    def __len__(self) -> int:
        return self.num_batches()

    def __iter__(self):
        return BatchDataLoaderIterator(self.dataset, self.strategy, self.batcher)
