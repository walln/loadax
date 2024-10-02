import pytest

from loadax import CombinedDataset, SimpleDataset


@pytest.fixture
def dataset1():
    return SimpleDataset([1, 2, 3])


@pytest.fixture
def dataset2():
    return SimpleDataset([4, 5, 6])


def test_combined_dataset_initialization(dataset1, dataset2):
    combined_dataset = CombinedDataset(dataset1, dataset2)
    assert isinstance(combined_dataset, CombinedDataset)
    assert len(combined_dataset) == len(dataset1) + len(dataset2)


def test_combined_dataset_len(dataset1, dataset2):
    combined_dataset = CombinedDataset(dataset1, dataset2)
    assert len(combined_dataset) == 6


def test_combined_dataset_getitem(dataset1, dataset2):
    combined_dataset = CombinedDataset(dataset1, dataset2)
    assert combined_dataset[0] == 1
    assert combined_dataset[2] == 3
    assert combined_dataset[3] == 4
    assert combined_dataset[5] == 6

    with pytest.raises(IndexError):
        _ = combined_dataset[6]


def test_combined_dataset_iter(dataset1, dataset2):
    combined_dataset = CombinedDataset(dataset1, dataset2)
    assert list(combined_dataset) == [1, 2, 3, 4, 5, 6]


def test_combined_dataset_with_different_data_types():
    dataset1 = SimpleDataset([1, 2, 3])
    dataset2 = SimpleDataset(["a", "b", "c"])
    combined_dataset = CombinedDataset(dataset1, dataset2)
    assert list(combined_dataset) == [1, 2, 3, "a", "b", "c"]


def test_combined_dataset_empty():
    empty_dataset = SimpleDataset([])
    combined_dataset = CombinedDataset(empty_dataset, empty_dataset)
    assert len(combined_dataset) == 0
    assert list(combined_dataset) == []

    with pytest.raises(IndexError):
        _ = combined_dataset[0]


def test_combined_dataset_large():
    large_data1 = list(range(5000))
    large_data2 = list(range(5000, 10000))
    dataset1 = SimpleDataset(large_data1)
    dataset2 = SimpleDataset(large_data2)
    combined_dataset = CombinedDataset(dataset1, dataset2)
    assert len(combined_dataset) == 10000
    assert combined_dataset[0] == 0
    assert combined_dataset[4999] == 4999
    assert combined_dataset[5000] == 5000
    assert combined_dataset[9999] == 9999


def test_combined_dataset_protocol_methods(dataset1, dataset2):
    combined_dataset = CombinedDataset(dataset1, dataset2)

    assert hasattr(combined_dataset, "__iter__")
    assert list(iter(combined_dataset)) == [1, 2, 3, 4, 5, 6]

    assert hasattr(combined_dataset, "__len__")
    assert len(combined_dataset) == 6

    assert hasattr(combined_dataset, "__getitem__")
    assert combined_dataset[2] == 3
    assert combined_dataset[4] == 5


def test_combined_dataset_chaining():
    dataset1 = SimpleDataset([1, 2])
    dataset2 = SimpleDataset([3, 4])
    dataset3 = SimpleDataset([5, 6])

    combined1 = CombinedDataset(dataset1, dataset2)
    combined2 = CombinedDataset(combined1, dataset3)

    assert list(combined2) == [1, 2, 3, 4, 5, 6]


def test_combined_dataset_type_propagation():
    int_dataset = SimpleDataset([1, 2, 3])
    str_dataset = SimpleDataset(["a", "b", "c"])
    float_dataset = SimpleDataset([1.0, 2.0, 3.0])

    combined1 = CombinedDataset(int_dataset, str_dataset)
    combined2 = CombinedDataset(combined1, float_dataset)

    assert isinstance(combined2[0], int)
    assert isinstance(combined2[3], str)
    assert isinstance(combined2[6], float)


def test_combined_dataset_with_empty_dataset():
    non_empty_dataset = SimpleDataset([1, 2, 3])
    empty_dataset = SimpleDataset([])

    combined1 = CombinedDataset(non_empty_dataset, empty_dataset)
    combined2 = CombinedDataset(empty_dataset, non_empty_dataset)

    assert list(combined1) == [1, 2, 3]
    assert list(combined2) == [1, 2, 3]
    assert len(combined1) == len(combined2) == 3
