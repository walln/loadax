import jax
import pytest

from loadax.dataset.sampled_dataset import SampledDataset
from loadax.dataset.simple import SimpleDataset


@pytest.fixture
def full_dataset():
    return SimpleDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def random_key():
    return jax.random.PRNGKey(0)


def test_sampled_dataset_initialization(full_dataset, random_key):
    sampled_dataset = SampledDataset(full_dataset, 5, random_key)
    assert isinstance(sampled_dataset, SampledDataset)
    assert len(sampled_dataset) == 5


def test_sampled_dataset_len(full_dataset, random_key):
    sampled_dataset = SampledDataset(full_dataset, 3, random_key)
    assert len(sampled_dataset) == 3


def test_sampled_dataset_getitem(full_dataset, random_key):
    sampled_dataset = SampledDataset(full_dataset, 5, random_key)
    assert 1 <= sampled_dataset[0] <= 10
    assert 1 <= sampled_dataset[2] <= 10
    assert 1 <= sampled_dataset[4] <= 10

    with pytest.raises(IndexError, match="Index out of range"):
        _ = sampled_dataset[5]


def test_sampled_dataset_iter(full_dataset, random_key):
    sampled_dataset = SampledDataset(full_dataset, 5, random_key)
    sampled_list = list(sampled_dataset)
    assert len(sampled_list) == 5
    assert all(1 <= x <= 10 for x in sampled_list)
    assert len(set(sampled_list)) == 5  # All elements should be unique


def test_sampled_dataset_invalid_sample_size(full_dataset, random_key):
    with pytest.raises(ValueError, match="Invalid sample size"):
        SampledDataset(full_dataset, -1, random_key)

    with pytest.raises(ValueError, match="Invalid sample size"):
        SampledDataset(full_dataset, 15, random_key)


def test_sampled_dataset_empty():
    empty_dataset = SimpleDataset([])
    with pytest.raises(ValueError, match="Invalid sample size"):
        SampledDataset(empty_dataset, 1, jax.random.PRNGKey(0))


def test_sampled_dataset_full_sample(full_dataset, random_key):
    sampled_dataset = SampledDataset(full_dataset, 10, random_key)
    assert len(sampled_dataset) == 10
    assert set(sampled_dataset) == set(full_dataset)


def test_sampled_dataset_reproducibility(full_dataset):
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(42)

    sampled_dataset1 = SampledDataset(full_dataset, 5, key1)
    sampled_dataset2 = SampledDataset(full_dataset, 5, key2)

    assert list(sampled_dataset1) == list(sampled_dataset2)


def test_sampled_dataset_different_keys(full_dataset):
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    sampled_dataset1 = SampledDataset(full_dataset, 5, key1)
    sampled_dataset2 = SampledDataset(full_dataset, 5, key2)

    assert list(sampled_dataset1) != list(sampled_dataset2)


def test_sampled_dataset_protocol_methods(full_dataset, random_key):
    sampled_dataset = SampledDataset(full_dataset, 5, random_key)

    assert hasattr(sampled_dataset, "__iter__")
    assert len(list(iter(sampled_dataset))) == 5

    assert hasattr(sampled_dataset, "__len__")
    assert len(sampled_dataset) == 5

    assert hasattr(sampled_dataset, "__getitem__")
    assert 1 <= sampled_dataset[2] <= 10
