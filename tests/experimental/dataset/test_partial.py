import pytest

from loadax.experimental.dataset.partial_dataset import PartialDataset
from loadax.experimental.dataset.simple import SimpleDataset


@pytest.fixture
def full_dataset():
    return SimpleDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_partial_dataset_initialization(full_dataset):
    partial_dataset = PartialDataset(full_dataset, 2, 7)
    assert isinstance(partial_dataset, PartialDataset)
    assert len(partial_dataset) == 5


def test_partial_dataset_len(full_dataset):
    partial_dataset = PartialDataset(full_dataset, 0, 5)
    assert len(partial_dataset) == 5


def test_partial_dataset_getitem(full_dataset):
    partial_dataset = PartialDataset(full_dataset, 2, 7)
    assert partial_dataset[0] == 3
    assert partial_dataset[2] == 5
    assert partial_dataset[4] == 7

    with pytest.raises(IndexError, match="Index out of range"):
        _ = partial_dataset[5]


def test_partial_dataset_iter(full_dataset):
    partial_dataset = PartialDataset(full_dataset, 2, 7)
    assert list(partial_dataset) == [3, 4, 5, 6, 7]


def test_partial_dataset_invalid_range(full_dataset):
    with pytest.raises(ValueError, match="Invalid start or end index"):
        PartialDataset(full_dataset, -1, 5)

    with pytest.raises(ValueError, match="Invalid start or end index"):
        PartialDataset(full_dataset, 5, 15)

    with pytest.raises(ValueError, match="Invalid start or end index"):
        PartialDataset(full_dataset, 7, 5)


def test_partial_dataset_empty():
    empty_dataset = SimpleDataset([])
    with pytest.raises(ValueError, match="Invalid start or end index"):
        PartialDataset(empty_dataset, 0, 1)


def test_partial_dataset_single_element(full_dataset):
    partial_dataset = PartialDataset(full_dataset, 5, 6)
    assert len(partial_dataset) == 1
    assert list(partial_dataset) == [6]


def test_partial_dataset_protocol_methods(full_dataset):
    partial_dataset = PartialDataset(full_dataset, 2, 7)

    assert hasattr(partial_dataset, "__iter__")
    assert list(iter(partial_dataset)) == [3, 4, 5, 6, 7]

    assert hasattr(partial_dataset, "__len__")
    assert len(partial_dataset) == 5

    assert hasattr(partial_dataset, "__getitem__")
    assert partial_dataset[2] == 5


def test_partial_dataset_split():
    full_dataset = SimpleDataset(list(range(100)))
    partials = PartialDataset.split_dataset(full_dataset, 4)

    assert len(partials) == 4
    assert all(isinstance(p, PartialDataset) for p in partials)
    assert [len(p) for p in partials] == [25, 25, 25, 25]
    assert list(partials[0]) == list(range(0, 25))
    assert list(partials[1]) == list(range(25, 50))
    assert list(partials[2]) == list(range(50, 75))
    assert list(partials[3]) == list(range(75, 100))


def test_partial_dataset_split_uneven():
    full_dataset = SimpleDataset(list(range(10)))
    partials = PartialDataset.split_dataset(full_dataset, 3)

    assert len(partials) == 3
    assert [len(p) for p in partials] == [4, 3, 3]
    assert list(partials[0]) == [0, 1, 2, 3]
    assert list(partials[1]) == [4, 5, 6]
    assert list(partials[2]) == [7, 8, 9]


def test_partial_dataset_split_invalid():
    full_dataset = SimpleDataset(list(range(10)))

    with pytest.raises(ValueError, match="Number of partitions must be at least 1"):
        PartialDataset.split_dataset(full_dataset, 0)

    with pytest.raises(
        ValueError, match="Number of partitions cannot exceed dataset size"
    ):
        PartialDataset.split_dataset(full_dataset, 11)
