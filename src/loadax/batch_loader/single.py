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
        rng: int | None,
    ):
        self.dataset = dataset
        self.batcher = batcher
        self.strategy = strategy
        self.rng = rng

    def num_items(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        # when starting the iterator, we need to check if the loader was created with a shuffling strategy
        # if so, we need to shuffle the dataset before starting the iterator

        if not self.rng:
            dataset = self.dataset
        # elif self.rng:
        #     ShuffledDataset(self.dataset, self.rng)

        return BatchDataLoaderIterator(dataset, self.strategy, self.batcher)
