import time

from loadax import Batcher, DataloaderBuilder, InMemoryDataset


def test_multiprocessing_dataloader():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(2).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    result = []
    for batch in dataloader:
        result.extend(batch)

    assert result == [1, 2, 3, 4, 5]


def test_multiprocessing_dataloader_batch_size():
    dataset = InMemoryDataset(list(range(10)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(3).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    batches = list(dataloader)

    assert len(batches) == 4  # Should create 4 batches (3+3+3+1)
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]


def test_multiprocessing_dataloader_large_dataset():
    dataset = InMemoryDataset(list(range(100)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(10).workers(4).prefetch(2)
    dataloader = builder.build(dataset)

    result = []
    for batch in dataloader:
        result.extend(batch)

    assert result == list(range(100))


def test_multiprocessing_dataloader_empty_dataset():
    dataset = InMemoryDataset([])
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(2).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    result = list(dataloader)

    assert result == []


def test_multiprocessing_dataloader_single_element():
    dataset = InMemoryDataset([1])
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(2).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    result = list(dataloader)

    assert result == [[1]]


def test_multiprocessing_dataloader_non_divisible_batch_size():
    dataset = InMemoryDataset(list(range(7)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(3).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    batches = list(dataloader)

    assert len(batches) == 3  # Should create 3 batches (3+3+1)
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6]


def test_multiprocessing_dataloader_multiple_workers_large_dataset():
    dataset = InMemoryDataset(list(range(1000)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(100).workers(10).prefetch(2)
    dataloader = builder.build(dataset)

    result = []
    for batch in dataloader:
        result.extend(batch)

    assert result == list(range(1000))


def test_multiprocessing_dataloader_worker_handling():
    dataset = InMemoryDataset(list(range(20)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(4).workers(3).prefetch(2)
    dataloader = builder.build(dataset)

    batches = list(dataloader)

    assert len(batches) == 5  # Should create 5 batches (4+4+4+4+4)
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6, 7]
    assert batches[2] == [8, 9, 10, 11]
    assert batches[3] == [12, 13, 14, 15]
    assert batches[4] == [16, 17, 18, 19]


def test_multiprocessing_dataloader_different_prefetch_factors():
    dataset = InMemoryDataset(list(range(20)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(5).workers(2).prefetch(4)
    dataloader = builder.build(dataset)

    batches = list(dataloader)

    assert len(batches) == 4  # Should create 4 batches (5+5+5+5)
    assert batches[0] == [0, 1, 2, 3, 4]
    assert batches[1] == [5, 6, 7, 8, 9]
    assert batches[2] == [10, 11, 12, 13, 14]
    assert batches[3] == [15, 16, 17, 18, 19]


def test_multiprocessing_dataloader_varying_data_types():
    dataset = InMemoryDataset([1, "a", 3.0, True, None])
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(2).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    batches = list(dataloader)

    assert len(batches) == 2  # Should create 3 batches (2+2+1)
    assert batches[0] == [1, "a"]
    assert batches[1] == [3.0, True]


def test_multiprocessing_dataloader_large_number_of_small_batches():
    dataset = InMemoryDataset(list(range(50)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(1).workers(5).prefetch(2)
    dataloader = builder.build(dataset)

    result = []
    for batch in dataloader:
        result.extend(batch)

    assert result == list(range(50))


def test_multiprocessing_dataloader_odd_elements_and_batch_size():
    dataset = InMemoryDataset(list(range(15)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(4).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    batches = list(dataloader)

    assert len(batches) == 4  # Should create 4 batches (4+4+4+3)
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6, 7]
    assert batches[2] == [8, 9, 10, 11]
    assert batches[3] == [12, 13, 14]


class SlowDataset(InMemoryDataset):
    """Slow dataset that simulates slow data retrieval."""

    def get(self, index: int):
        """Get the item at the given index."""
        time.sleep(0.01)  # Simulate slow data retrieval
        return super().get(index)


def test_multiprocessing_dataloader_slow_dataset():
    dataset = SlowDataset(list(range(10)))
    batcher = Batcher(lambda x: x)
    builder = DataloaderBuilder(batcher).batch_size(2).workers(2).prefetch(2)
    dataloader = builder.build(dataset)

    result = []
    for batch in dataloader:
        result.extend(batch)

    assert result == list(range(10))
