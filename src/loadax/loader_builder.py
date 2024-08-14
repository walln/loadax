from loadax.strategy import FixedBatchStrategy, BatchStrategy
from loadax.batcher import Batcher
from loadax.batch_loader import MultiThreadedBatchDataLoader, BatchDataLoader
from loadax.dataset import Dataset
from loadax.transform.partial import PartialDataset


class SingleThreadedBatchDataloader:
    def __init__(
        self, dataset: Dataset, batcher: Batcher, strategy: BatchStrategy, rng: int
    ):
        pass


class DataLoaderBuilder:
    strategy: BatchStrategy | None = None
    seed: int | None = None
    num_threads: int | None = None

    def __init__(self, batcher: Batcher):
        self.batcher = batcher

    def batch_size(self, batch_size: int):
        self.strategy = FixedBatchStrategy(batch_size)
        return self

    def shuffle(self, seed: int):
        self.seed = seed
        return self

    def num_workers(self, num_threads: int):
        self.num_threads = num_threads
        return self

    def build(self, dataset: Dataset):
        strategy = self.strategy if self.strategy else FixedBatchStrategy(1)
        rng = self.seed if self.seed else None

        if self.num_threads:
            print(f"Splitting dataset into {self.num_threads} chunks")
            datasets = PartialDataset.split(dataset, self.num_threads)
            print(f"Created {len(datasets)} datasets")
            print(f"Dataset sizes: {[len(dataset) for dataset in datasets]}")
            # TODO: PRNG key splitting
            rngs = [rng for _ in range(self.num_threads)]
            dataloaders = [
                BatchDataLoader(
                    dataset=dataset, batcher=self.batcher, strategy=strategy, rng=rng
                )
                for (dataset, rng) in zip(datasets, rngs)
            ]
            print("Creating multi threaded dataloader")
            return MultiThreadedBatchDataLoader(
                dataloaders=dataloaders,
            )
        else:
            print("Creating single threaded dataloader")
            return BatchDataLoader(
                dataset=dataset, batcher=self.batcher, strategy=strategy, rng=rng
            )
