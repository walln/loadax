import jax
from jax.sharding import Mesh, PartitionSpec

from loadax import Batcher, DataLoader
from loadax.dataloader.distributed import DistributedDataLoader, JaxShardingStrategy
from loadax.dataset.in_memory import InMemoryDataset
from loadax.strategy import FixedBatchStrategy


def test_distributed_dataloader_single_shard():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = JaxShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = DistributedDataLoader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=0,
        num_shards=1,
    )

    batches = list(dataloader)
    assert len(batches) == 10
    assert all(len(batch) == 10 for batch in batches)
    assert [item for batch in batches for item in batch] == list(range(100))


def test_distributed_dataloader_multiple_shards():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=5)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = JaxShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader0 = DistributedDataLoader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=0,
        num_shards=2,
    )

    dataloader1 = DistributedDataLoader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=1,
        num_shards=2,
    )

    batches0 = list(dataloader0)
    batches1 = list(dataloader1)

    assert len(batches0) == 10
    assert len(batches1) == 10
    assert all(len(batch) == 5 for batch in batches0 + batches1)
    assert [item for batch in batches0 for item in batch] == list(range(0, 50))
    assert [item for batch in batches1 for item in batch] == list(range(50, 100))


def test_distributed_dataloader_uneven_batch():
    dataset = InMemoryDataset(list(range(95)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = JaxShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = DistributedDataLoader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=0,
        num_shards=1,
    )

    batches = list(dataloader)
    assert len(batches) == 10
    assert len(batches[-1]) == 5
    assert [item for batch in batches for item in batch] == list(range(95))


def test_distributed_dataloader_error_handling():
    class ErrorDataset(InMemoryDataset):
        def get(self, index: int) -> int:
            if index == 50:
                raise ValueError("Simulated error")
            return super().get(index)

    dataset = ErrorDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = JaxShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = DistributedDataLoader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=0,
        num_shards=1,
    )

    batches = list(dataloader)
    assert len(batches) == 10
    flattened = [item for batch in batches for item in batch]
    assert len(flattened) == 99
    assert 50 not in flattened


def test_distributed_dataloader_progress():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = JaxShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = DistributedDataLoader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=0,
        num_shards=1,
    )

    iterator = iter(dataloader)
    assert iterator.progress().items_processed == 0
    assert iterator.progress().items_total == 100

    next(iterator)
    assert iterator.progress().items_processed == 10

    list(iterator)
    assert iterator.progress().items_processed == 100


def create_dataloader(
    dataset_size: int, batch_size: int, num_shards: int, shard_id: int
):
    dataset = InMemoryDataset(list(range(dataset_size)))
    batcher = Batcher(lambda x: x)
    mesh = Mesh(jax.devices(), ("dp",))

    return (
        DataLoader(batcher)
        .batch_size(batch_size)
        .prefetch(2)
        .shard(mesh, "dp", num_shards, shard_id)
        .build(dataset)
    )


def test_last_batch_smaller():
    dataloader = create_dataloader(
        dataset_size=95, batch_size=10, num_shards=1, shard_id=0
    )
    batches = list(dataloader)

    assert len(batches) == 10
    assert all(len(batch) == 10 for batch in batches[:-1])
    assert len(batches[-1]) == 5
    assert [item for batch in batches for item in batch] == list(range(95))


def test_exact_multiple_of_batch_size():
    dataloader = create_dataloader(
        dataset_size=100, batch_size=10, num_shards=1, shard_id=0
    )
    batches = list(dataloader)

    assert len(batches) == 10
    assert all(len(batch) == 10 for batch in batches)
    assert [item for batch in batches for item in batch] == list(range(100))


def test_single_item_dataset():
    dataloader = create_dataloader(
        dataset_size=1, batch_size=10, num_shards=1, shard_id=0
    )
    batches = list(dataloader)

    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0] == 0


def test_empty_dataset():
    dataloader = create_dataloader(
        dataset_size=0, batch_size=10, num_shards=1, shard_id=0
    )
    batches = list(dataloader)

    assert len(batches) == 0


def test_uneven_sharding():
    dataloader0 = create_dataloader(
        dataset_size=95, batch_size=10, num_shards=2, shard_id=0
    )
    dataloader1 = create_dataloader(
        dataset_size=95, batch_size=10, num_shards=2, shard_id=1
    )

    batches0 = list(dataloader0)
    batches1 = list(dataloader1)

    assert len(batches0) == 5
    assert len(batches1) == 5
    assert all(len(batch) == 10 for batch in batches0[:-1])
    assert len(batches0[-1]) == 7
    assert all(len(batch) == 10 for batch in batches1[:-1])
    assert len(batches1[-1]) == 8

    items0 = [item for batch in batches0 for item in batch]
    items1 = [item for batch in batches1 for item in batch]
    assert items0 + items1 == list(range(95))


def test_batch_size_larger_than_dataset():
    dataloader = create_dataloader(
        dataset_size=5, batch_size=10, num_shards=1, shard_id=0
    )
    batches = list(dataloader)

    assert len(batches) == 1
    assert len(batches[0]) == 5
    assert batches[0] == list(range(5))


def test_multiple_small_batches():
    dataloader = create_dataloader(
        dataset_size=95, batch_size=3, num_shards=1, shard_id=0
    )
    batches = list(dataloader)

    assert len(batches) == 32
    assert all(len(batch) == 3 for batch in batches[:-1])
    assert len(batches[-1]) == 2
    assert [item for batch in batches for item in batch] == list(range(95))
