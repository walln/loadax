from loadax import InMemoryDataset


def test_in_memory_dataset_basic():
    dataset = InMemoryDataset([1, 2, 3])
    assert dataset.get(0) == 1
    assert dataset.get(1) == 2
    assert dataset.get(2) == 3
    assert dataset.get(3) is None
    assert len(dataset) == 3


def test_in_memory_dataset_out_of_bounds():
    dataset = InMemoryDataset([1, 2, 3])
    assert dataset.get(-1) == 3
    assert dataset.get(-2) == 2
    assert dataset.get(-3) == 1
    assert dataset.get(-4) == 3
    assert dataset.get(4) is None
    assert dataset.get(5) is None


def test_in_memory_dataset_complex_types():
    dataset = InMemoryDataset([{"a": 1}, {"b": 2}])
    assert dataset.get(0)["a"] == 1
    assert dataset.get(1)["b"] == 2
    assert dataset.get(2) is None
    assert len(dataset) == 2


def test_in_memory_dataset_empty():
    dataset = InMemoryDataset([])
    assert dataset.get(0) is None
    assert len(dataset) == 0


def test_in_memory_dataset_single_element():
    dataset = InMemoryDataset([42])
    assert dataset.get(0) == 42
    assert dataset.get(1) is None
    assert len(dataset) == 1


def test_in_memory_dataset_multiple_types():
    dataset = InMemoryDataset([1, "string", 3.14, {"key": "value"}, [5, 6, 7]])
    assert dataset.get(0) == 1
    assert dataset.get(1) == "string"
    assert dataset.get(2) == 3.14
    assert dataset.get(3) == {"key": "value"}
    assert dataset.get(4) == [5, 6, 7]
    assert dataset.get(5) is None
    assert len(dataset) == 5


def test_in_memory_dataset_repr():
    dataset = InMemoryDataset([1, 2, 3])
    assert repr(dataset) == "InMemoryDataset(items=[1, 2]...)"


def test_in_memory_dataset_modifying_retrieved_data():
    # In-place modification of retrieved data, to ensure there are no
    # extraneous copies of theunderlying data
    dataset = InMemoryDataset([{"a": 1}, {"b": 2}])
    item = dataset.get(0)
    item["a"] = 99
    assert dataset.get(0)["a"] == 99  # Ensure original data is modified


def test_in_memory_dataset_large_dataset():
    large_data = list(range(1_000_000))
    dataset = InMemoryDataset(large_data)
    assert len(dataset) == 1000000
    assert dataset.get(0) == 0
    assert dataset.get(999999) == 999999
    assert dataset.get(1000000) is None


def test_in_memory_dataset_with_none_values():
    dataset = InMemoryDataset([1, None, 3])
    assert dataset.get(0) == 1
    assert dataset.get(1) is None
    assert dataset.get(2) == 3
    assert dataset.get(3) is None
    assert len(dataset) == 3
