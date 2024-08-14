from loadax.loader_builder import DataLoaderBuilder
from loadax.dataset.in_memory import InMemoryDataset
from loadax.batcher import Batcher


def test_single_batch_loader():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batcher = Batcher(lambda x: x)
    loader = DataLoaderBuilder(batcher).batch_size(2).build(dataset)

    batches = []
    for batch in loader:
        batches.append(batch)

    assert len(batches) == 5
    assert batches[0] == [1, 2]
    assert batches[1] == [3, 4]
    assert batches[2] == [5, 6]
    assert batches[3] == [7, 8]
    assert batches[4] == [9, 10]


def test_batching():
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
