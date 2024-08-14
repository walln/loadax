from loadax.dataset.in_memory import InMemoryDataset


def test_in_memory_dataset():
    dataset = InMemoryDataset([1, 2, 3])
    assert dataset.get(0) == 1
    assert dataset.get(1) == 2
    assert dataset.get(2) == 3
    assert dataset.get(3) is None
    assert len(dataset) == 3


def test_out_of_bounds():
    dataset = InMemoryDataset([1, 2, 3])
    assert dataset.get(-1) == 3
    assert dataset.get(4) is None
    assert dataset.get(5) is None


def test_complex_types():
    dataset = InMemoryDataset([{"a": 1}, {"b": 2}])
    assert dataset.get(0)["a"] == 1
    assert dataset.get(1)["b"] == 2
