from loadax.loader_builder import DataLoaderBuilder
from loadax.dataset import InMemoryDataset
from loadax.batcher import Batcher


def test_prefetch_batch_loader():
    batcher = Batcher(lambda x: x)
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    loader = DataLoaderBuilder(batcher).batch_size(2).pretech(2).build(dataset)
    assert len(loader) == 5
    batches = []
    for batch in loader:
        batches.append(batch)
    assert batches == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_prefetch_batch_loader_initialization():
    batcher = Batcher(lambda x: x)
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    loader = DataLoaderBuilder(batcher).batch_size(2).pretech(2).build(dataset)
    assert loader.prefetch_factor == 2
    assert not loader.data_queue.empty()
    assert loader.data_queue.qsize() == 2


def test_prefetch_batch_loader_iteration():
    batcher = Batcher(lambda x: x)
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    loader = DataLoaderBuilder(batcher).batch_size(2).pretech(2).build(dataset)
    iterator = loader.get_iterator()
    batches = list(iterator)

    assert len(batches) == 10  # 100 items, batch size 10
    for i, batch in enumerate(batches):
        assert batch == [i * 10 + j for j in range(10)]


if __name__ == "__main__":
    test_prefetch_batch_loader_initialization()
    test_prefetch_batch_loader_iteration()
    test_prefetch_batch_loader()
