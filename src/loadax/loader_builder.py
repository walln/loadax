from loadax.strategy import FixedBatchStrategy, BatchStrategy
from loadax.batcher import Batcher
from loadax.batch_loader import BatchDataLoader, PrefetchBatchLoader
from loadax.dataset import Dataset


class DataLoaderBuilder:
    strategy: BatchStrategy | None = None
    num_threads: int | None = None
    prefetch_factor: int | None = None

    def __init__(self, batcher: Batcher):
        self.batcher = batcher

    def batch_size(self, batch_size: int):
        self.strategy = FixedBatchStrategy(batch_size)
        return self

    # def shuffle(self, seed: int):
    #     self.seed = seed
    #     return self

    def num_workers(self, num_threads: int):
        self.num_threads = num_threads
        return self

    def pretech(self, factor: int):
        self.prefetch_factor = factor
        return self

    def build(self, dataset: Dataset):
        strategy = self.strategy if self.strategy else FixedBatchStrategy(1)

        if self.prefetch_factor:
            print("Creating prefetch dataloader")
            return PrefetchBatchLoader(
                dataset=dataset,
                strategy=strategy,
                batcher=self.batcher,
                prefetch_factor=self.prefetch_factor,
            )

        if self.num_threads:
            print("Creating multi threaded dataloader")
            raise NotImplementedError

        print("Creating single threaded dataloader")
        return BatchDataLoader(dataset=dataset, batcher=self.batcher, strategy=strategy)
