from loadax import CombinedDataset, InMemoryDataset


def test_combined_dataset_basic():
    dataset1 = InMemoryDataset([1, 2, 3])
    dataset2 = InMemoryDataset([4, 5, 6])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 6
    assert combined_dataset.get(0) == 1
    assert combined_dataset.get(2) == 3
    assert combined_dataset.get(3) == 4
    assert combined_dataset.get(5) == 6
    assert combined_dataset.get(6) is None


def test_combined_dataset_with_empty_dataset():
    dataset1 = InMemoryDataset([1, 2, 3])
    dataset2 = InMemoryDataset([])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 3
    assert combined_dataset.get(0) == 1
    assert combined_dataset.get(2) == 3
    assert combined_dataset.get(3) is None


def test_combined_dataset_all_empty():
    dataset1 = InMemoryDataset([])
    dataset2 = InMemoryDataset([])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 0
    assert combined_dataset.get(0) is None


def test_combined_dataset_with_complex_types():
    dataset1 = InMemoryDataset([{"a": 1}, {"b": 2}])
    dataset2 = InMemoryDataset([{"c": 3}, {"d": 4}])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 4
    assert combined_dataset.get(0)["a"] == 1
    assert combined_dataset.get(1)["b"] == 2
    assert combined_dataset.get(2)["c"] == 3
    assert combined_dataset.get(3)["d"] == 4
    assert combined_dataset.get(4) is None


def test_combined_dataset_with_single_element_datasets():
    dataset1 = InMemoryDataset([1])
    dataset2 = InMemoryDataset([2])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 2
    assert combined_dataset.get(0) == 1
    assert combined_dataset.get(1) == 2
    assert combined_dataset.get(2) is None


def test_combined_dataset_repr():
    dataset1 = InMemoryDataset([1, 2, 3])
    dataset2 = InMemoryDataset([4, 5, 6])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert (
        repr(combined_dataset) == f"CombinedDataset(datasets=[{dataset1}, {dataset2}])"
    )


def test_combined_dataset_large():
    dataset1 = InMemoryDataset(list(range(1000)))
    dataset2 = InMemoryDataset(list(range(1000, 2000)))
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 2000
    assert combined_dataset.get(0) == 0
    assert combined_dataset.get(999) == 999
    assert combined_dataset.get(1000) == 1000
    assert combined_dataset.get(1999) == 1999
    assert combined_dataset.get(2000) is None


def test_combined_dataset_mixed_types():
    dataset1 = InMemoryDataset([1, 2, 3])
    dataset2 = InMemoryDataset(["a", "b", "c"])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert len(combined_dataset) == 6
    assert combined_dataset.get(0) == 1
    assert combined_dataset.get(3) == "a"
    assert combined_dataset.get(5) == "c"
    assert combined_dataset.get(6) is None


def test_combined_dataset_access_last_element():
    dataset1 = InMemoryDataset([1, 2, 3])
    dataset2 = InMemoryDataset([4, 5, 6])
    combined_dataset = CombinedDataset([dataset1, dataset2])

    assert combined_dataset.get(5) == 6
    assert combined_dataset.get(6) is None


def test_combined_dataset_single_dataset():
    dataset1 = InMemoryDataset([1, 2, 3])
    combined_dataset = CombinedDataset([dataset1])

    assert len(combined_dataset) == 3
    assert combined_dataset.get(0) == 1
    assert combined_dataset.get(2) == 3
    assert combined_dataset.get(3) is None
