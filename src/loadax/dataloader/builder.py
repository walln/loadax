"""The core primitive for creating dataloaders in loadax."""

from typing import Generic, TypeVar

import jax
from jax.sharding import Mesh, PartitionSpec

from loadax.batcher import Batcher
from loadax.dataloader.distributed import DistributedDataLoader, JaxShardingStrategy
from loadax.dataloader.naive import NaiveDataLoader
from loadax.dataset import Dataset
from loadax.strategy import BatchStrategy, FixedBatchStrategy

DatasetItem = TypeVar("DatasetItem", covariant=True)
Batch = TypeVar("Batch", covariant=True)


class DataLoader(Generic[DatasetItem, Batch]):
    """A dataloader is a primitive for efficiently loading data from a dataset.

    A dataloader is effectively a smart iterator that optimizes getting data from the
    dataset. The dataloader is responsible for ensuring the data is batched, ready
    when it is needed, and loaded into mmemory in a way that is efficient. All of this
    should be done without hijacking the main thread and preventing you from doing the
    important parts of your training loop.

    The dataloader is built by chaining methods on the `DataLoader` class. The
    methods are designed to be chained in a fluent style, allowing you to build
    a dataloader with the desired configuration.

    Example:
        >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
        >>> batcher = Batcher(lambda x: x)
        >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
        >>> iterator = iter(dataloader)
        >>> for batch in iterator:
        ...     print(batch)

    Attributes:
        batcher (Batcher): The batcher to use for batching.
        strategy (BatchStrategy): The batch strategy to use for computing batches.
        num_workers (int): The number of workers to use for parallel data loading.
        prefetch_factor (int): The prefetch factor to use for prefetching.
        sharding_strategy (JaxShardingStrategy): The sharding strategy to use for
            distributed data loading.
        shard_id (int): The ID of the shard to load the data from.
        num_shards (int): The number of shards to distribute the data across.
    """

    batcher: Batcher[DatasetItem, Batch]
    strategy: BatchStrategy[DatasetItem] | None = None
    num_workers: int | None = 1
    prefetch_factor: int | None = 2
    sharding_strategy: JaxShardingStrategy | None = None
    shard_id: int | None = None
    num_shards: int | None = None

    def __init__(self, batcher: Batcher[DatasetItem, Batch]):
        """A dataloader is a primitive for efficiently loading data from a dataset.

        A dataloader is effectively a smart iterator that optimizes getting data from
        the dataset. The dataloader is responsible for ensuring the data is batched,
        ready when it is needed, and loaded into mmemory in a way that is efficient.
        All of this should be done without hijacking the main thread and preventing you
        from doing the important parts of your training loop.

        The dataloader is built by chaining methods on the `DataLoader` class. The
        methods are designed to be chained in a fluent style, allowing you to build
        a dataloader with the desired configuration.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> batcher = Batcher(lambda x: x)
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            batcher (Batcher): The batcher to use for batching.
        """
        self.batcher = batcher

    def batch_size(self, batch_size: int) -> "DataLoader[DatasetItem, Batch]":
        """Set the batch size for the dataloader.

        This method sets the batch size for the dataloader. The batch size is the
        number of items to include in a batch. The batch size should be a multiple
        of the number of workers, otherwise the dataloader will not be able to
        prefetch batches efficiently.

        Currently, the dataloader only supports fixed batch sizes. This means that
        with the exception of the last batch, all other batches will have the same
        size. This may change in the future with the introduction of dynamic batch
        sizes.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> batcher = Batcher(lambda x: x)
            >>> dataloader = DataLoader(batcher).batch_size(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            batch_size (int): The batch size to use for the dataloader.

        Returns:
            DataLoader: The dataloader with the batch size set.
        """
        self.strategy = FixedBatchStrategy(batch_size)
        return self

    def workers(self, num_workers: int) -> "DataLoader[DatasetItem, Batch]":
        """Set the number of workers for the dataloader.

        This method sets the number of workers for the dataloader. The number of
        workers determines the number of parallel threads that will be used to
        load data. The number of workers should be a multiple of the batch size,
        otherwise the dataloader will not be able to prefetch batches efficiently.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> batcher = Batcher(lambda x: x)
            >>> dataloader = DataLoader(batcher).batch_size(2).workers(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            num_workers (int): The number of workers to use for the dataloader.

        Returns:
            DataLoader: The dataloader with the number of workers set.
        """
        self.num_workers = num_workers
        return self

    def pretech(self, factor: int) -> "DataLoader[DatasetItem, Batch]":
        """Set the prefetch factor for the dataloader.

        This method sets the prefetch factor for the dataloader. The prefetch
        factor determines the number of batches to prefetch ahead of time. The
        prefetch factor is a multiplier and should be a multiple of the number
        of workers, otherwise the dataloader will not be able to prefetch batches
        efficiently.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> batcher = Batcher(lambda x: x)
            >>> dataloader = DataLoader(batcher).batch_size(2).pretech(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            factor (int): The prefetch factor to use for the dataloader.

        Returns:
            DataLoader: The dataloader with the prefetch factor set.
        """
        self.prefetch_factor = factor if factor > 0 else 1
        return self

    def shard(
        self,
        mesh: Mesh,
        partition_spec: PartitionSpec,
        num_shards: int | None = None,
        shard_id: int | None = None,
    ) -> "DataLoader[DatasetItem, Batch]":
        """Set the mesh and partition spec for the dataloader.

        This will distribute the dataloading across multiple nodes within the same
        distributed network. This is useful for training large models on multiple
        nodes. You can then load the data into multiple devices on each node and
        train the model in parallel.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> batcher = Batcher(lambda x: x)
            >>> dataloader = DataLoader(batcher)
                                .batch_size(2)
                                .shard(mesh, partition_spec)
                                .build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            mesh (Mesh): The mesh to use for sharding.
            partition_spec (PartitionSpec): The partition spec to use for sharding.
            num_shards (int): The number of shards to distribute the data across.
                If not specified, the dataloader will choose automatically using
                jax.process_count().
            shard_id (int): The ID of the shard to load the data from.
                If not specified, the dataloader will choose automatically using
                jax.process_index().

        Returns:
            DataLoader: The dataloader with the mesh and partition spec set.
        """
        self.sharding_strategy = JaxShardingStrategy(mesh, partition_spec)
        self.shard_id = shard_id
        self.num_shards = num_shards
        return self

    def build(
        self, dataset: Dataset[DatasetItem]
    ) -> (
        NaiveDataLoader[DatasetItem, Batch] | DistributedDataLoader[DatasetItem, Batch]
    ):
        """Construct the dataloader from the current configuration.

        This method constructs the dataloader from the current configuration. The
        dataloader is constructed based on the current configuration of the dataloader.
        If the dataloader is not configured to use multiprocessing, it will use the
        naive dataloader. If the dataloader is configured to use multiprocessing, it
        will use the multiprocessing dataloader.

        The default configuration of the dataloader is to use a fixed batch size of
        1 and to use a single worker. This means that there is not parallelism in
        loading the data, but the data will still be prefetched with 2 batches in
        advance on the single worker. This is a good default configuration for most
        use cases, unless you have a fast training loop and potentially many devices.

        Example:
            >>> dataset = InMemoryDataset([1, 2, 3, 4, 5])
            >>> batcher = Batcher(lambda x: x)
            >>> dataloader = DataLoader(batcher).batch_size(2).workers(2).build(dataset)
            >>> iterator = iter(dataloader)
            >>> for batch in iterator:
            ...     print(batch)

        Args:
            dataset (Dataset): The dataset to load data from.

        Returns:
            DataLoader: The dataloader constructed from the current configuration.
        """
        strategy = self.strategy if self.strategy else FixedBatchStrategy(1)

        if self.sharding_strategy:
            return DistributedDataLoader(
                dataset=dataset,
                strategy=strategy,
                batcher=self.batcher,
                num_workers=self.num_workers or 1,
                prefetch_factor=self.prefetch_factor or 1,
                sharding_strategy=self.sharding_strategy,
                shard_id=self.shard_id or jax.process_index(),
                num_shards=self.num_shards or jax.process_count(),
            )

        if self.num_workers or self.prefetch_factor:
            # TODO: Log warning if num_workers is greater than 1 and prefetch_factor is
            # not set to a value greater than 1, and vice versa.

            # Create dummy sharding strategy
            partition_spec = PartitionSpec("dp")  # type: ignore
            sharding_strategy = JaxShardingStrategy(
                Mesh(jax.devices(), ("dp",)), partition_spec
            )
            return DistributedDataLoader(
                dataset=dataset,
                strategy=strategy,
                batcher=self.batcher,
                num_workers=self.num_workers or 1,
                prefetch_factor=self.prefetch_factor or 1,
                sharding_strategy=sharding_strategy,
                shard_id=0,
                num_shards=1,
            )

        return NaiveDataLoader(dataset=dataset, batcher=self.batcher, strategy=strategy)
