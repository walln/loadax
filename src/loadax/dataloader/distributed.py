"""Distributed dataloader that shards batches across the entire cluster."""

import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

from jax.sharding import Mesh, PartitionSpec

from loadax.batcher import Batcher
from loadax.dataloader.naive import NaiveDataLoader
from loadax.dataloader.progress import Progress
from loadax.dataloader.protocol import DataLoaderIteratorProtocol
from loadax.dataset import Dataset
from loadax.strategy import BatchStrategy

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


class JaxShardingStrategy:
    """Define a distributed sharding strategy for the dataloader.

    A sharding strategy is responsible for determining which data indices
    should be loaded by each worker node. This is done by dividing the
    dataset into equally sized shards and assigning each worker node a
    subset of the shards.
    """

    def __init__(self, mesh: Mesh, sharding_spec: PartitionSpec):
        """Define a distributed sharding strategy for the dataloader.

        A sharding strategy is responsible for determining which data indices
        should be loaded by each worker node. This is done by dividing the
        dataset into equally sized shards and assigning each worker node a
        subset of the shards.

        Args:
            mesh (Mesh): The mesh to use for sharding.
            sharding_spec (PartitionSpec): The sharding specification to use.
        """
        self.mesh = mesh
        self.sharding_spec = sharding_spec

    def get_shard(self, dataset_size: int, shard_id: int, num_shards: int) -> list[int]:
        """Get the list of data indices for a specific shard."""
        shard_size = dataset_size // num_shards
        start_index = shard_id * shard_size
        end_index = (
            start_index + shard_size if shard_id < num_shards - 1 else dataset_size
        )
        return list(range(start_index, end_index))


class DistributedDataLoaderIterator(DataLoaderIteratorProtocol[DatasetItem, Batch]):
    """Iterator for the threaded dataloader."""

    def __init__(self, dataloader: "DistributedDataLoader[DatasetItem, Batch]"):
        """Iterator for the dataloader.

        Args:
            dataloader (ThreadedDataLoader): The dataloader to iterate over.
        """
        self.dataloader = dataloader
        self.executor = ThreadPoolExecutor(max_workers=self.dataloader.num_workers)
        self.current_index = 0
        self.cache: dict[int, DatasetItem] = {}
        self.futures: dict[int, Future[tuple[int, DatasetItem | None]]] = {}
        self.cache_lock = threading.Lock()
        self.index_lock = threading.Lock()

        self._shutdown_finalizer = weakref.finalize(self, self._shutdown_executor)

        self._start_prefetching()

    def _start_prefetching(self) -> None:
        """Start prefetching data in the background."""
        with self.index_lock:
            prefetch_end = min(
                self.current_index
                + (
                    self.dataloader.strategy.batch_size
                    * self.dataloader.prefetch_factor
                ),
                len(self.dataloader.shard_indices),
            )
            indices_to_prefetch = self.dataloader.shard_indices[
                self.current_index : prefetch_end
            ]
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
            if self.current_index >= len(self.dataloader.shard_indices):
                raise StopIteration

            batch_end = min(
                self.current_index + self.dataloader.strategy.batch_size,
                len(self.dataloader.shard_indices),
            )
            batch_indices = self.dataloader.shard_indices[
                self.current_index : batch_end
            ]
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

    def __iter__(self) -> "DistributedDataLoaderIterator[DatasetItem,Batch]":
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
        if self._shutdown_finalizer.alive:
            self._shutdown_finalizer()

    def progress(self) -> Progress:
        """Get the progress of the dataloader."""
        return Progress(self.current_index, len(self.dataloader.shard_indices))


class DistributedDataLoader(NaiveDataLoader[DatasetItem, Batch]):
    """Dataloader that leverages threading for non-blocking data loading."""

    def __init__(
        self,
        dataset: Dataset[DatasetItem],
        batcher: Batcher[DatasetItem, Batch],
        strategy: BatchStrategy[DatasetItem],
        num_workers: int,
        prefetch_factor: int,
        sharding_strategy: JaxShardingStrategy,
        shard_id: int,
        num_shards: int,
    ):
        """A dataloader that leverages threading for non-blocking data loading.

        Args:
            dataset (Dataset): The dataset to load data from.
            batcher (Batcher): The batcher to use for batching.
            strategy (BatchStrategy): The batch strategy to use for prefetching.
            num_workers (int): The number of workers to use for parallel data loading.
            prefetch_factor (int): The prefetch factor to use for prefetching.
            sharding_strategy (JaxShardingStrategy): The sharding strategy for
                distributed data loading.
            shard_id (int): The ID of this shard (node).
            num_shards (int): The total number of shards (nodes).
        """
        super().__init__(dataset, batcher, strategy)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.sharding_strategy = sharding_strategy
        self.shard_id = shard_id
        self.num_shards = num_shards

        # Determine the shard indices for this node
        self.shard_indices = self.sharding_strategy.get_shard(
            len(dataset), shard_id, num_shards
        )

        print(f"Shard indices: {self.shard_indices} for shard {shard_id}")

    def __iter__(self) -> DataLoaderIteratorProtocol[DatasetItem, Batch]:
        """Get an iterator for the dataloader."""
        return DistributedDataLoaderIterator(self)
