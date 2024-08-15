from loadax.strategy import FixedBatchStrategy, BatchStrategy
from loadax.batcher import Batcher
from loadax.dataloader import NaiveDataLoader, MultiProcessingDataLoader
from loadax.dataset import Dataset


class DataLoaderBuilder:
    strategy: BatchStrategy | None = None
    num_workers: int | None = None
    prefetch_factor: int | None = 2

    def __init__(self, batcher: Batcher):
        self.batcher = batcher

    def batch_size(self, batch_size: int):
        self.strategy = FixedBatchStrategy(batch_size)
        return self

    def workers(self, num_workers: int):
        self.num_workers = num_workers
        return self

    def pretech(self, factor: int):
        self.prefetch_factor = factor
        return self

    def build(self, dataset: Dataset):
        strategy = self.strategy if self.strategy else FixedBatchStrategy(1)

        if self.num_workers:
            print("Creating multiprocessing dataloader")
            return MultiProcessingDataLoader(
                dataset=dataset,
                strategy=strategy,
                batcher=self.batcher,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )

        print("Creating single threaded dataloader")
        return NaiveDataLoader(dataset=dataset, batcher=self.batcher, strategy=strategy)
