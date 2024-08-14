from loadax.dataset.in_memory import InMemoryDataset
from loadax.batcher import Batcher
from loadax.dataset.range import RangeDataset
from loadax.loader_builder import DataLoaderBuilder


def test_dataloader_batching():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6])
    batcher = Batcher(lambda items: sum(items))
    dataloader = DataLoaderBuilder(batcher).batch_size(2).build(dataset)

    values = []
    for val in dataloader:
        values.append(val)

    assert values == [3, 7, 11]


def test_dataloader_iteration():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6])
    batcher = Batcher(lambda items: sum(items))
    dataloader = DataLoaderBuilder(batcher).batch_size(2).build(dataset)

    values = []
    iterator = iter(dataloader)
    for val in iterator:
        values.append(val)
    assert values == [3, 7, 11]

    values.clear()
    iterator = iter(dataloader)
    for _ in range(3):
        val = next(iterator)
        values.append(val)
    assert values == [3, 7, 11]


def test_multithreaded_dataloader():
    dataset = RangeDataset(0, 1000)
    batcher = Batcher(lambda items: sum(items))
    dataloader = DataLoaderBuilder(batcher).batch_size(10).num_workers(2).build(dataset)

    values = []
    for val in dataloader:
        values.append(val)

    assert len(values) == 100
