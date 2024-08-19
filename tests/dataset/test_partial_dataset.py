import pytest

from loadax import InMemoryDataset, PartialDataset


def test_partial_dataset_basic():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    partial_dataset = PartialDataset(dataset, 1, 4)

    assert len(partial_dataset) == 3
    assert partial_dataset.get(0) == 2
    assert partial_dataset.get(1) == 3
    assert partial_dataset.get(2) == 4
    assert partial_dataset.get(3) is None


def test_partial_dataset_out_of_bounds():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    partial_dataset = PartialDataset(dataset, 1, 10)

    assert len(partial_dataset) == 4
    assert partial_dataset.get(0) == 2
    assert partial_dataset.get(3) == 5
    assert partial_dataset.get(4) is None


def test_partial_dataset_negative_indices():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    partial_dataset = PartialDataset(dataset, 1, 4)

    assert partial_dataset.get(-1) == 4
    assert partial_dataset.get(-2) == 3
    assert partial_dataset.get(-3) == 2
    assert partial_dataset.get(-4) is None


def test_partial_dataset_empty():
    dataset = InMemoryDataset([])
    partial_dataset = PartialDataset(dataset, 0, 2)

    assert len(partial_dataset) == 0
    assert partial_dataset.get(0) is None


def test_partial_dataset_split():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    partial_datasets = PartialDataset.split(dataset, 3)

    assert len(partial_datasets) == 3
    assert len(partial_datasets[0]) == 2
    assert len(partial_datasets[1]) == 2
    assert len(partial_datasets[2]) == 1

    assert partial_datasets[0].get(0) == 1
    assert partial_datasets[0].get(1) == 2
    assert partial_datasets[0].get(2) is None

    assert partial_datasets[1].get(0) == 3
    assert partial_datasets[1].get(1) == 4
    assert partial_datasets[1].get(2) is None

    assert partial_datasets[2].get(0) == 5
    assert partial_datasets[2].get(1) is None


def test_partial_dataset_split_exact_parts():
    dataset = InMemoryDataset([1, 2, 3, 4, 5, 6])
    partial_datasets = PartialDataset.split(dataset, 2)

    assert len(partial_datasets) == 2
    assert len(partial_datasets[0]) == 3
    assert len(partial_datasets[1]) == 3

    assert partial_datasets[0].get(0) == 1
    assert partial_datasets[0].get(2) == 3
    assert partial_datasets[0].get(3) is None

    assert partial_datasets[1].get(0) == 4
    assert partial_datasets[1].get(2) == 6
    assert partial_datasets[1].get(3) is None


def test_partial_dataset_split_one_part():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    partial_datasets = PartialDataset.split(dataset, 1)

    assert len(partial_datasets) == 1
    assert len(partial_datasets[0]) == 5

    assert partial_datasets[0].get(0) == 1
    assert partial_datasets[0].get(4) == 5
    assert partial_datasets[0].get(5) is None


def test_partial_dataset_split_more_parts_than_elements():
    dataset = InMemoryDataset([1, 2, 3])
    partial_datasets = PartialDataset.split(dataset, 5)

    assert len(partial_datasets) == 5
    assert len(partial_datasets[0]) == 1
    assert len(partial_datasets[1]) == 1
    assert len(partial_datasets[2]) == 1
    assert len(partial_datasets[3]) == 0
    assert len(partial_datasets[4]) == 0

    assert partial_datasets[0].get(0) == 1
    assert partial_datasets[0].get(1) is None

    assert partial_datasets[1].get(0) == 2
    assert partial_datasets[1].get(1) is None

    assert partial_datasets[2].get(0) == 3
    assert partial_datasets[2].get(1) is None


if __name__ == "__main__":
    pytest.main([__file__])
