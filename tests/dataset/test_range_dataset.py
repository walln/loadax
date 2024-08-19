import pytest

from loadax import RangeDataset


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
    assert dataset.get(19) == 9
    assert dataset.get(20) is None
    assert len(dataset) == 20


def test_range_dataset_reverse_raises():
    with pytest.raises(ValueError, match="start must be less than end"):
        RangeDataset(10, 0)


def test_range_dataset_single_element():
    dataset = RangeDataset(5, 6)
    assert dataset.get(0) == 5
    assert dataset.get(1) is None
    assert len(dataset) == 1


def test_range_dataset_large_range():
    dataset = RangeDataset(0, 1000000)
    assert dataset.get(0) == 0
    assert dataset.get(999999) == 999999
    assert dataset.get(1000000) is None
    assert len(dataset) == 1000000


def test_range_dataset_custom_step():
    dataset = RangeDataset(0, 10, step=2)
    assert dataset.get(0) == 0
    assert dataset.get(1) == 2
    assert dataset.get(4) == 8
    assert dataset.get(5) is None
    assert len(dataset) == 5


def test_range_dataset_edge_cases():
    dataset = RangeDataset(-100, 100, step=3)
    assert dataset.get(0) == -100
    assert dataset.get(65) == 95
    assert dataset.get(67) is None
    assert len(dataset) == 67


def test_range_dataset_step_one():
    dataset = RangeDataset(0, 10, step=1)
    assert dataset.get(0) == 0
    assert dataset.get(9) == 9
    assert dataset.get(10) is None
    assert len(dataset) == 10


def test_range_dataset_invalid_step():
    with pytest.raises(ValueError, match="step must be greater than 0"):
        RangeDataset(0, 10, step=0)

    with pytest.raises(ValueError, match="step must be greater than 0"):
        RangeDataset(0, 10, step=-1)
