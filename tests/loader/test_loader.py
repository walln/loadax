import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from loadax import Batcher
from loadax.dataloader.builder import DataloaderBuilder
from loadax.dataloader.loader import Dataloader
from loadax.dataloader.sharding import DistributedShardingStrategy, NoShardingStrategy
from loadax.dataset.in_memory import InMemoryDataset
from loadax.strategy import FixedBatchStrategy


def test_distributed_dataloader_single_shard():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = Dataloader(
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
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader0 = Dataloader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=0,
        num_shards=2,
    )

    dataloader1 = Dataloader(
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
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = Dataloader(
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
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = Dataloader(
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
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    dataloader = Dataloader(
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
        DataloaderBuilder(batcher)
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
    assert len(batches0[-1]) == 8
    assert all(len(batch) == 10 for batch in batches1[:-1])
    assert len(batches1[-1]) == 7

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


def test_distributed_dataloader_multiple_shards_four():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=5)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    total_items = []
    for shard_id in range(4):
        dataloader = Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
            shard_id=shard_id,
            num_shards=4,
        )
        batches = list(dataloader)
        assert all(
            len(batch) == 5 for batch in batches[:-1]
        )  # Last batch may be smaller
        items = [item for batch in batches for item in batch]
        total_items.extend(items)

    # Ensure all items are covered and no duplicates
    assert sorted(total_items) == list(range(100))
    assert len(total_items) == 100


def test_distributed_dataloader_uneven_shards():
    dataset = InMemoryDataset(list(range(103)))  # 103 is not divisible by 4
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))

    total_items = []
    for shard_id in range(4):
        batcher = Batcher(lambda x: x)
        strategy = FixedBatchStrategy(batch_size=10)
        dataloader = Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=1,
            prefetch_factor=1,
            sharding_strategy=sharding_strategy,
            shard_id=shard_id,
            num_shards=4,
        )
        batches = list(dataloader)
        items = [item for batch in batches for item in batch if item is not None]
        total_items.extend(items)

    # Ensure all items are covered and no duplicates
    assert sorted(total_items) == list(range(103))
    assert len(total_items) == 103


def test_distributed_dataloader_invalid_shard_id():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    with pytest.raises(AssertionError, match="shard_id .* must be in the range"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
            shard_id=5,  # Invalid shard_id, assuming num_shards=4
            num_shards=4,
        )


def test_distributed_dataloader_no_workers():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    with pytest.raises(AssertionError, match="num_workers must be at least 1"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=0,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
        )


def test_distributed_dataloader_negative_workers():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    with pytest.raises(AssertionError, match="num_workers must be at least 1"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=-1,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
        )


def test_distributed_dataloader_no_prefetch():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    with pytest.raises(AssertionError, match="prefetch_factor must be at least 1"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=0,
            sharding_strategy=sharding_strategy,
        )


def test_distributed_dataloader_negative_prefetch():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    with pytest.raises(AssertionError, match="prefetch_factor must be at least 1"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=-1,
            sharding_strategy=sharding_strategy,
        )


def test_distributed_dataloader_variable_item_sizes():
    class VariableSizeDataset(InMemoryDataset):
        def get(self, index):
            return [index] * (index % 5 + 1)

    dataset = VariableSizeDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    dataloader = Dataloader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
    )

    batches = list(dataloader)
    assert len(batches) == 10
    # Verify that items within a batch have variable sizes
    for batch in batches:
        assert all(isinstance(item, list) for item in batch)
        assert len({len(item) for item in batch}) > 1


def test_distribute_global_batch_single_process():
    local_batch = np.arange(10)
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = DistributedShardingStrategy(mesh, data_shard_axis="dp")
    global_batch = sharding_strategy.distribute_global_batch(local_batch)
    # Since we're in single process, the global batch should equal the local batch
    assert np.array_equal(global_batch, local_batch)


def test_distributed_dataloader_invalid_num_shards():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    # num_shards=0 is invalid
    with pytest.raises(AssertionError, match="num_shards must be greater than 0"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
            shard_id=0,
            num_shards=0,
        )

    # num_shards negative
    with pytest.raises(AssertionError, match="num_shards must be greater than 0"):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
            shard_id=0,
            num_shards=-1,
        )


def test_distributed_dataloader_missing_shard_id_or_num_shards():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    # Missing shard_id
    with pytest.raises(
        AssertionError,
        match="Either both shard_id and num_shards must be provided or neither",
    ):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
            num_shards=2,
        )

    # Missing num_shards
    with pytest.raises(
        AssertionError,
        match="Either both shard_id and num_shards must be provided or neither",
    ):
        Dataloader(
            dataset,
            batcher,
            strategy,
            num_workers=2,
            prefetch_factor=2,
            sharding_strategy=sharding_strategy,
            shard_id=0,
        )


def test_distributed_dataloader_dict_items():
    class DictDataset(InMemoryDataset):
        def get(self, index):
            return {"index": index, "value": index * 2}

    dataset = DictDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=10)
    sharding_strategy = NoShardingStrategy()

    dataloader = Dataloader(
        dataset,
        batcher,
        strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
    )

    batches = list(dataloader)
    assert len(batches) == 10
    for batch in batches:
        assert all(isinstance(item, dict) for item in batch)
        for item in batch:
            assert "index" in item
            assert "value" in item


def test_distributed_dataloader_concurrent_iteration():
    import threading

    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=5)
    sharding_strategy = NoShardingStrategy()

    dataloader = Dataloader(
        dataset,
        batcher,
        strategy,
        num_workers=4,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
    )

    results = []

    def iterate_dataloader():
        for batch in dataloader:
            results.extend(batch)

    threads = [threading.Thread(target=iterate_dataloader) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Ensure that data is not duplicated and all items are present
    assert sorted(set(results)) == list(range(100))


def test_distributed_dataloader_finalizer():
    import gc

    dataset = InMemoryDataset(list(range(1000000)))  # Large dataset
    batcher = Batcher(lambda x: x)
    strategy = FixedBatchStrategy(batch_size=1000)
    sharding_strategy = NoShardingStrategy()

    dataloader = Dataloader(
        dataset,
        batcher,
        strategy,
        num_workers=8,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
    )

    iterator = iter(dataloader)
    # Consume some data
    for _ in range(5):
        next(iterator)

    # Delete iterator and force garbage collection
    del iterator
    gc.collect()
