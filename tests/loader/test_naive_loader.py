from loadax import Batcher, DataloaderBuilder, InMemoryDataset


def test_single_batch_loader():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    batches = list(loader)

    assert len(batches) == 5
    assert batches == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_batching():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6])
    batcher = Batcher(lambda items: sum(items))
    dataloader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    values = list(dataloader)

    assert values == [3, 7, 11]


def test_dataloader_iteration():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6])
    batcher = Batcher(lambda items: sum(items))
    dataloader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    values = list(dataloader)
    assert values == [3, 7, 11]

    values.clear()
    iterator = iter(dataloader)
    for _ in range(3):
        val = next(iterator)
        values.append(val)
    assert values == [3, 7, 11]


def test_empty_dataset():
    dataset = InMemoryDataset([])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    batches = list(loader)

    assert batches == []


def test_single_item_dataset():
    dataset = InMemoryDataset([1])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    batches = list(loader)

    assert batches == [[1]]


def test_non_divisible_batch_size():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    batches = list(loader)

    assert len(batches) == 3
    assert batches == [[1, 2], [3, 4], [5]]


def test_large_batch_size():
    dataset = InMemoryDataset([1, 2, 3])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(10).build(dataset)

    batches = list(loader)

    assert len(batches) == 1
    assert batches == [[1, 2, 3]]


def test_small_batch_size():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(1).build(dataset)

    batches = list(loader)

    assert len(batches) == 5
    assert batches == [[1], [2], [3], [4], [5]]


def test_dataloader_different_data_types():
    dataset = InMemoryDataset([1, "a", 3.0, {"key": "value"}, True])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    batches = list(loader)

    assert len(batches) == 3
    assert batches == [[1, "a"], [3.0, {"key": "value"}], [True]]


def test_dataloader_custom_batch_function():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6])
    batcher = Batcher(lambda items: [item * 2 for item in items])
    loader = DataloaderBuilder(batcher).workers(0).batch_size(3).build(dataset)

    batches = list(loader)

    assert len(batches) == 2
    assert batches == [[2, 4, 6], [8, 10, 12]]


class CustomDataset(InMemoryDataset):
    """Custom dataset that returns twice the value of the original dataset."""

    def get(self, index: int):
        """Get the item at the given index."""
        if index >= len(self.items):
            return None
        return super().get(index) * 2


def test_dataloader_custom_dataset():
    dataset = CustomDataset([1, 2, 3, 4])
    batcher = Batcher(lambda x: x)
    loader = DataloaderBuilder(batcher).workers(0).batch_size(2).build(dataset)

    batches = list(loader)

    assert len(batches) == 2
    assert batches == [[2, 4], [6, 8]]
