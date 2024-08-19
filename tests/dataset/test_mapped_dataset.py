import pytest

from loadax import InMemoryDataset, MappedDataset


def test_mapped_dataset_basic():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])

    def map_fn(x):
        return x * 2

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 5
    assert mapped_dataset.get(0) == 2
    assert mapped_dataset.get(4) == 10
    assert mapped_dataset.get(5) is None
    assert mapped_dataset.get(-1) == 10


def test_mapped_dataset_empty():
    dataset = InMemoryDataset([])

    def map_fn(x):
        return x * 2

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 0
    assert mapped_dataset.get(0) is None


def test_mapped_dataset_string_map():
    dataset = InMemoryDataset([1, 2, 3])

    def map_fn(x):
        return f"item_{x}"

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 3
    assert mapped_dataset.get(0) == "item_1"
    assert mapped_dataset.get(1) == "item_2"
    assert mapped_dataset.get(2) == "item_3"


def test_mapped_dataset_repr():
    dataset = InMemoryDataset([1, 2, 3])

    def map_fn(x):
        return x * 2

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert repr(mapped_dataset) == f"MappedDataset(dataset={dataset}, map_fn={map_fn})"


def test_mapped_dataset_types():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])

    def map_fn(x: int):
        return x * 2

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 5

    item = mapped_dataset.get(0)
    assert item == 2
    assert isinstance(item, int)


def test_mapped_dataset_complex_map():
    dataset = InMemoryDataset([1, 2, 3])

    def map_fn(x):
        return {"original": x, "squared": x**2, "cubed": x**3}

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 3
    assert mapped_dataset.get(0) == {"original": 1, "squared": 1, "cubed": 1}
    assert mapped_dataset.get(1) == {"original": 2, "squared": 4, "cubed": 8}
    assert mapped_dataset.get(2) == {"original": 3, "squared": 9, "cubed": 27}


def test_mapped_dataset_large_dataset():
    dataset = InMemoryDataset(list(range(1000)))

    def map_fn(x):
        return x * 2

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 1000
    assert mapped_dataset.get(0) == 0
    assert mapped_dataset.get(999) == 1998


def test_mapped_dataset_exception_handling():
    dataset = InMemoryDataset([1, 2, 3])

    def map_fn(x):
        if x == 2:
            raise ValueError("Test exception")
        return x * 2

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 3
    assert mapped_dataset.get(0) == 2
    with pytest.raises(ValueError, match="Test exception"):
        mapped_dataset.get(1)
    assert mapped_dataset.get(2) == 6


def test_mapped_dataset_non_unique_mapping():
    dataset = InMemoryDataset([1, 2, 3, 4])

    def map_fn(x):
        return x % 2  # This will map to 0 or 1

    mapped_dataset = MappedDataset(dataset, map_fn)

    assert len(mapped_dataset) == 4
    assert mapped_dataset.get(0) == 1
    assert mapped_dataset.get(1) == 0
    assert mapped_dataset.get(2) == 1
    assert mapped_dataset.get(3) == 0
