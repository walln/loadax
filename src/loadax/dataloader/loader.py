"""Distributed dataloader that shards batches across the entire cluster."""

import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

import jax

from loadax.batcher import Batcher
from loadax.dataloader.naive import NaiveDataloader
from loadax.dataloader.progress import Progress
from loadax.dataloader.protocol import DataloaderIteratorProtocol
from loadax.dataloader.sharding import ShardingStrategy
from loadax.dataset import Dataset
from loadax.strategy import BatchStrategy

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


class DataloaderIterator(DataloaderIteratorProtocol[DatasetItem, Batch]):
    """Iterator for the threaded dataloader."""

    def __init__(self, dataloader: "Dataloader[DatasetItem, Batch]"):
        """Iterator for the dataloader.

        Args:
            dataloader (Dataloader): The dataloader to iterate over.
        """
        self.dataloader = dataloader
        self.executor = ThreadPoolExecutor(max_workers=self.dataloader.num_workers)
        self.current_index = 0
        self.cache: dict[int, DatasetItem] = {}
        self.futures: dict[int, Future[tuple[int, DatasetItem | None]]] = {}
        self.cache_lock = threading.Lock()
        self.index_lock = threading.Lock()

        self.shard_indices = list(self.dataloader.shard_indices)
        self.total_indices = len(self.shard_indices)

        self._shutdown_finalizer = weakref.finalize(self, self._shutdown_executor)

        self._start_prefetching()

    def _start_prefetching(self) -> None:
        """Start prefetching data in the background."""
        with self.index_lock:
            prefetch_end = self.current_index + (
                self.dataloader.strategy.batch_size * self.dataloader.prefetch_factor
            )
            indices_to_prefetch = self.shard_indices[self.current_index : prefetch_end]
        for index in indices_to_prefetch:
            if index not in self.futures and index not in self.cache:
                future = self.executor.submit(self._load_data, index)
                self.futures[index] = future

    def _load_data(self, index: int) -> tuple[int, DatasetItem | None]:
        return index, self.dataloader.dataset.get(index)

    def _get_item(self, index: int) -> DatasetItem | None:
        """Get a single data item from the data queue."""
        while True:
            with self.cache_lock:
                if index in self.cache:
                    return self.cache.pop(index)
                if index in self.futures and self.futures[index].done():
                    try:
                        _, item = self.futures[index].result()
                        del self.futures[index]
                        return item
                    except Exception as e:
                        print(f"Error loading item at index {index}: {e}")
                        del self.futures[index]
                        return None  # skip this index and move on to the next one

            self._start_prefetching()

    def __next__(self) -> Batch:
        """Get the next batch from the dataloader."""
        with self.index_lock:
            if self.current_index >= self.total_indices:
                raise StopIteration

            batch_end = self.current_index + self.dataloader.strategy.batch_size
            batch_indices = self.shard_indices[self.current_index : batch_end]
            self.current_index = batch_end

        self._start_prefetching()

        for index in batch_indices:
            item = self._get_item(index)
            if item is not None:
                self.dataloader.strategy.add(item)

        batch = self.dataloader.strategy.batch(force=True)
        if batch is None:
            raise StopIteration

        return self.dataloader.batcher.batch(batch)

    def __iter__(self) -> "DataloaderIterator[DatasetItem,Batch]":
        """Get an iterator for the dataloader."""
        self.current_index = 0
        self.futures.clear()
        self.cache.clear()
        self._start_prefetching()
        return self

    def _shutdown_executor(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)

    def __del__(self) -> None:
        """Clean up the dataloader."""
        if self._shutdown_finalizer and self._shutdown_finalizer.alive:
            self._shutdown_finalizer()

    def progress(self) -> Progress:
        """Get the progress of the dataloader."""
        return Progress(self.current_index, len(self.dataloader.shard_indices))


class Dataloader(NaiveDataloader[DatasetItem, Batch]):
    """Dataloader that leverages threading for non-blocking data loading."""

    def __init__(
        self,
        dataset: Dataset[DatasetItem],
        batcher: Batcher[DatasetItem, Batch],
        strategy: BatchStrategy[DatasetItem],
        num_workers: int,
        prefetch_factor: int,
        sharding_strategy: ShardingStrategy,
        shard_id: int | None = None,
        num_shards: int | None = None,
    ):
        """A dataloader that leverages threading for non-blocking data loading.

        Example:
            ```python
            from loadax import Dataloader, InMemoryDataset, Batcher

            dataset = InMemoryDataset([1, 2, 3, 4, 5])
            batcher = Batcher(lambda x: x)
            dataloader = Dataloader(
                dataset=dataset,
                batcher=batcher,
                strategy=FixedBatchStrategy(batch_size=2),
                num_workers=2,
                prefetch_factor=2,
                sharding_strategy=NoShardingStrategy(),
            )
            for batch in dataloader:
                print(batch)

            #> [1, 2]
            #> [3, 4]
            #> [5]
            ```

        Args:
            dataset (Dataset): The dataset to load data from.
            batcher (Batcher): The batcher to use for batching.
            strategy (BatchStrategy): The batch strategy to use for prefetching.
            num_workers (int): The number of workers to use for parallel data loading.
            prefetch_factor (int): The prefetch factor to use for prefetching.
            sharding_strategy (JaxShardingStrategy): The sharding strategy for
                distributed data loading.
            shard_id (int | None): The ID of this shard (node).
                If None, then the shard_id will be determined based on the
                current process index and the number of shards.
            num_shards (int | None): The total number of shards (nodes).
                If None, then the number of shards will be determined based on
                the number of processes.
        """
        super().__init__(dataset, batcher, strategy)
        assert num_workers > 0, "num_workers must be at least 1"
        self.num_workers = num_workers

        assert prefetch_factor > 0, "prefetch_factor must be at least 1"
        self.prefetch_factor = prefetch_factor

        self.sharding_strategy = sharding_strategy

        # Check that both shard_id and num_shards are provided or neither are provided

        assert (shard_id is None and num_shards is None) or (
            shard_id is not None and num_shards is not None
        ), "Either both shard_id and num_shards must be provided or neither"

        assert num_shards is None or num_shards > 0, "num_shards must be greater than 0"
        self.num_shards = num_shards or jax.process_count()

        self.shard_id = shard_id or (jax.process_index() % self.num_shards)
        assert (
            self.shard_id < self.num_shards
        ), f"shard_id {self.shard_id} must be in the range [0, {self.num_shards})"

        # Determine the shard indices for this node
        self.shard_indices = self.sharding_strategy.get_shard_indices(
            dataset_size=len(dataset),
            shard_id=self.shard_id,
            num_shards=self.num_shards,
        )

    def __iter__(self) -> DataloaderIteratorProtocol[DatasetItem, Batch]:
        """Get an iterator for the dataloader."""
        return DataloaderIterator(self)
