import pytest

from loadax.experimental.dataset.simple import SimpleDataset
from loadax.experimental.loader import Dataloader


@pytest.fixture
def sample_dataset():
    return SimpleDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_dataloader_basic(sample_dataset):
    dataloader = Dataloader(sample_dataset, batch_size=2)
    batches = list(dataloader)
    assert len(batches) == 5
    assert batches == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_dataloader_drop_last(sample_dataset):
    dataloader = Dataloader(sample_dataset, batch_size=3, drop_last=True)
    batches = list(dataloader)
    assert len(batches) == 3
    assert batches == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_dataloader_not_drop_last(sample_dataset):
    dataloader = Dataloader(sample_dataset, batch_size=3, drop_last=False)
    batches = list(dataloader)
    assert len(batches) == 4
    assert batches == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]


@pytest.mark.parametrize(("num_workers", "prefetch_factor"), [(0, 0), (1, 1), (2, 2)])
def test_dataloader_workers_and_prefetch(sample_dataset, num_workers, prefetch_factor):
    dataloader = Dataloader(
        sample_dataset,
        batch_size=2,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    batches = list(dataloader)
    assert len(batches) == 5
    assert batches == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_dataloader_multiple_iterations(sample_dataset):
    dataloader = Dataloader(sample_dataset, batch_size=2)
    for _ in range(3):
        batches = list(dataloader)
        assert len(batches) == 5
        assert batches == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_dataloader_empty_dataset():
    empty_dataset = SimpleDataset([])
    dataloader = Dataloader(empty_dataset, batch_size=2)
    batches = list(dataloader)
    assert len(batches) == 0


def test_dataloader_single_item_dataset():
    single_item_dataset = SimpleDataset([1])
    dataloader = Dataloader(single_item_dataset, batch_size=2)
    batches = list(dataloader)
    assert len(batches) == 1
    assert batches == [[1]]


def test_dataloader_large_batch_size(sample_dataset):
    dataloader = Dataloader(sample_dataset, batch_size=20)
    batches = list(dataloader)
    assert len(batches) == 1
    assert batches == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]


def test_dataloader_with_dict_dataset():
    dict_dataset = SimpleDataset([{"a": i, "b": i * 2} for i in range(1, 6)])
    dataloader = Dataloader(dict_dataset, batch_size=2)
    batches = list(dataloader)
    assert len(batches) == 3
    assert batches == [
        [{"a": 1, "b": 2}, {"a": 2, "b": 4}],
        [{"a": 3, "b": 6}, {"a": 4, "b": 8}],
        [{"a": 5, "b": 10}],
    ]


def test_dataloader_with_tuple_dataset():
    tuple_dataset = SimpleDataset([(i, i * 2) for i in range(1, 6)])
    dataloader = Dataloader(tuple_dataset, batch_size=2)
    batches = list(dataloader)
    assert len(batches) == 3
    assert batches == [[(1, 2), (2, 4)], [(3, 6), (4, 8)], [(5, 10)]]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
def test_dataloader_various_batch_sizes(sample_dataset, batch_size):
    dataloader = Dataloader(sample_dataset, batch_size=batch_size)
    batches = list(dataloader)
    assert len(batches) == (10 + batch_size - 1) // batch_size
    assert [item for batch in batches for item in batch] == list(range(1, 11))


@pytest.mark.parametrize("drop_last", [True, False])
def test_dataloader_len(sample_dataset, drop_last):
    dataloader = Dataloader(sample_dataset, batch_size=3, drop_last=drop_last)
    expected_len = 3 if drop_last else 4
    assert len(dataloader) == expected_len


def test_dataloader_concurrent_iterators(sample_dataset):
    dataloader = Dataloader(sample_dataset, batch_size=2)
    iterator1 = iter(dataloader)
    iterator2 = iter(dataloader)

    next(iterator1)
    batch1 = next(iterator1)
    batch2 = next(iterator2)

    assert batch1 != batch2
