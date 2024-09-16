import jax.random
import pytest

from loadax import InMemoryDataset, ShuffledDataset


def test_shuffled_dataset_basic():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    shuffled_dataset = ShuffledDataset(dataset, key)

    assert len(shuffled_dataset) == 5
    samples = [shuffled_dataset.get(i) for i in range(5)]
    assert sorted(samples) == [1, 2, 3, 4, 5]
    assert shuffled_dataset.get(5) is None


def test_shuffled_dataset_empty():
    dataset = InMemoryDataset([])
    key = jax.random.PRNGKey(0)
    shuffled_dataset = ShuffledDataset(dataset, key)

    assert len(shuffled_dataset) == 0
    assert shuffled_dataset.get(0) is None


def test_shuffled_dataset_single_element():
    dataset = InMemoryDataset([42])
    key = jax.random.PRNGKey(0)
    shuffled_dataset = ShuffledDataset(dataset, key)

    assert len(shuffled_dataset) == 1
    assert shuffled_dataset.get(0) == 42
    assert shuffled_dataset.get(1) is None


def test_shuffled_dataset_repeated_access():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    shuffled_dataset = ShuffledDataset(dataset, key)

    assert len(shuffled_dataset) == 5
    samples = [shuffled_dataset.get(i) for i in range(5)]
    assert sorted(samples) == [1, 2, 3, 4, 5]

    # Access again to ensure consistency
    samples_repeated = [shuffled_dataset.get(i) for i in range(5)]
    assert samples == samples_repeated


def test_shuffled_dataset_complex_types():
    dataset = InMemoryDataset([{"a": 1}, {"b": 2}, {"c": 3}])
    key = jax.random.PRNGKey(0)
    shuffled_dataset = ShuffledDataset(dataset, key)

    assert len(shuffled_dataset) == 3
    samples = [shuffled_dataset.get(i) for i in range(3)]
    assert sorted(samples, key=lambda x: next(iter(x.values()))) == [
        {"a": 1},
        {"b": 2},
        {"c": 3},
    ]


def test_shuffled_dataset_repr():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    shuffled_dataset = ShuffledDataset(dataset, key)

    assert repr(shuffled_dataset) == f"ShuffledDataset(dataset={dataset})"


if __name__ == "__main__":
    pytest.main([__file__])
