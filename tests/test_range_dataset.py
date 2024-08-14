import pytest
from loadax.dataset.range import RangeDataset


def test_range_dataset():
    dataset = RangeDataset(0, 10)
    assert dataset.get(0) == 0
    assert dataset.get(1) == 1
    assert dataset.get(9) == 9
    assert dataset.get(10) is None
    assert len(dataset) == 10


def test_range_dataset_negative():
    dataset = RangeDataset(-10, 10)
    assert dataset.get(-1) == 9
    assert dataset.get(0) == -10
    assert dataset.get(1) == -9
    assert dataset.get(9) == -1
    assert dataset.get(10) == 0
    assert dataset.get(20) is None
    assert dataset.get(19) == 9
    assert len(dataset) == 20


def test_range_dataset_reverse_raises():
    with pytest.raises(ValueError):
        RangeDataset(10, 0)
