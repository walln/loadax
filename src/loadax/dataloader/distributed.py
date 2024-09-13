"""Distributed dataloader that shards batches across the entire cluster."""

import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
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

    def __init__(self, mesh: Mesh, data_shard_axis: str | None = None):
        """Define a distributed sharding strategy for the dataloader.

        A sharding strategy is responsible for determining which data indices
        should be loaded by each worker node. This is done by dividing the
        dataset into equally sized shards and assigning each worker node a
        subset of the shards.

        Args:
            mesh (Mesh): The mesh to use for sharding.
            data_shard_axis (str | None): The axis name for the data sharding.
                This is used to determine how to compute the global batch from
                the node-local batches. If None, then the data sharding axis
                will be required to be supplied when using `distribute_global_batch`.
        """
        self.mesh = mesh
        self.data_axis = data_shard_axis

    def get_shard(self, dataset_size: int, shard_id: int, num_shards: int) -> list[int]:
        """Get the list of data indices for a specific shard."""
        shard_size = dataset_size // num_shards
        start_index = shard_id * shard_size
        end_index = (
            start_index + shard_size if shard_id < num_shards - 1 else dataset_size
        )
        return list(range(start_index, end_index))

    def named_sharding(self, *names: str | None) -> jax.NamedSharding:
        """Get a NamedSharding object for the given axis names.

        This is just a useful convenience method for creating NamedSharding objects,
        just provide the axis names as they exist in the mesh to create the
        NamedSharding object.

        Args:
            *names (str | None): The axis names to use for the NamedSharding object.

        Returns:
            jax.NamedSharding: The NamedSharding object for the given axis names.
        """
        return jax.NamedSharding(self.mesh, PartitionSpec(*names))  # type: ignore # jax PartitionSpec has missing types in current version

    # TODO: walln - add numpy type hints
    def distribute_global_batch(
        self, local_batch: np.ndarray[Any, Any], *, data_axis: str | None = None
    ) -> jnp.ndarray:
        """Distribute a local batch across the entire cluster.

        This method takes a local batch and distributes it across the entire
        cluster using the data sharding strategy. It returns a global batch
        that can be used for training on the entire cluster.

        Args:
            local_batch (np.ndarray): The local batch to distribute.
            data_axis (str | None, optional): The axis name for the data sharding.
                If None, then the data axis must have been provided to the
                sharding_strategy. Defaults to None.

        Returns:
            jax.Array: The global batch.
        """
        # local_batch: the batch of data on the local process
        # data_axis: the axis name for the data sharding

        data_axis = data_axis or self.data_axis
        if not data_axis:
            raise ValueError(
                "data_axis must be provided when using distribute_global_batch"
            )

        # Define the sharding for the global array
        data_sharding = self.named_sharding(data_axis)

        # Get the global batch size and shape
        global_batch_size = local_batch.shape[0] * jax.process_count()
        global_batch_shape = (global_batch_size,) + local_batch.shape[1:]

        # Create the global array from local batches
        def data_callback(index: Any) -> np.ndarray[Any, Any]:
            # index: the global index for the sharded array
            # We need to return the local data corresponding to this index
            # Since each process only has its local data, we check if the index
            # corresponds to our process

            # Calculate the start and end indices for this process
            per_process_batch_size = local_batch.shape[0]
            start = jax.process_index() * per_process_batch_size
            end = start + per_process_batch_size

            # Check if the requested index overlaps with our local data
            index_start = index[0].start or 0
            index_stop = index[0].stop or global_batch_size

            # If the requested index is within our local data range, return
            # the corresponding data
            if start <= index_start < end:
                local_index_start = index_start - start
                local_index_stop = min(index_stop - start, per_process_batch_size)
                return local_batch[local_index_start:local_index_stop]
            else:
                # Return an empty array if the index does not correspond to our data
                return np.zeros((0,) + local_batch.shape[1:], dtype=local_batch.dtype)

        global_batch = jax.make_array_from_callback(
            global_batch_shape, data_sharding, data_callback
        )
        return global_batch


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
        sharding_strategy: JaxShardingStrategy | None = None,
        shard_id: int | None = None,
        num_shards: int | None = None,
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
            shard_id (int | None): The ID of this shard (node).
                If None, then the shard_id will be determined based on the
                current process index and the number of shards.
            num_shards (int | None): The total number of shards (nodes).
                If None, then the number of shards will be determined based on
                the number of processes.
        """
        super().__init__(dataset, batcher, strategy)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.sharding_strategy = sharding_strategy

        if not (shard_id is not None and num_shards is not None) and (
            shard_id or num_shards
        ):
            raise ValueError(
                "Either both shard_id and num_shards must be provided or neither"
            )

        self.num_shards = num_shards or jax.process_count()
        self.shard_id = shard_id or (jax.process_index() % self.num_shards)

        # Determine the shard indices for this node
        self.shard_indices = (
            self.sharding_strategy.get_shard(
                dataset_size=len(dataset),
                shard_id=self.shard_id,
                num_shards=self.num_shards,
            )
            if self.sharding_strategy
            else list(range(len(dataset)))
        )

    def __iter__(self) -> DataLoaderIteratorProtocol[DatasetItem, Batch]:
        """Get an iterator for the dataloader."""
        return DistributedDataLoaderIterator(self)
