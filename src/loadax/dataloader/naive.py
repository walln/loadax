from typing import TypeVar, Generic
from loadax.strategy import BatchStrategy
from loadax.batcher import Batcher
from loadax.dataloader.protocol import DataLoaderIteratorProtocol, DataLoaderProtocol
from loadax.dataloader.progress import Progress
from loadax.dataset import Dataset

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


class NaiveDataLoaderIterator(DataLoaderIteratorProtocol[Batch]):
    def __init__(self, dataloader: "NaiveDataLoader[DatasetItem,Batch]"):
        self.dataloader = dataloader
        self.current_index = 0

    def __next__(self) -> Batch:
        while True:
            item = self.dataloader.dataset.get(self.current_index)
            if item is None:
                break

            self.current_index += 1
            self.dataloader.strategy.add(item)

            items = self.dataloader.strategy.batch(False)
            if items is not None:
                return self.dataloader.batcher.batch(items)

        items = self.dataloader.strategy.batch(True)
        if items is not None:
            return self.dataloader.batcher.batch(items)

        raise StopIteration

    def __iter__(self):
        return self

    def progress(self) -> Progress:
        return Progress(
            self.current_index, len(self.dataloader.dataset.get(self.current_index))
        )


class NaiveDataLoader(
    DataLoaderProtocol[DatasetItem, Batch], Generic[DatasetItem, Batch]
):
    def __init__(
        self,
        dataset: Dataset[DatasetItem],
        batcher: Batcher[DatasetItem, Batch],
        strategy: BatchStrategy[DatasetItem],
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
        return NaiveDataLoaderIterator(self)
