import jax
import pytest

from loadax import SimpleDataset


@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]


def test_simple_dataset_initialization(sample_data):
    dataset = SimpleDataset(sample_data)
    assert isinstance(dataset, SimpleDataset)
    assert len(dataset) == len(sample_data)


def test_simple_dataset_len(sample_data):
    dataset = SimpleDataset(sample_data)
    assert len(dataset) == 5


def test_simple_dataset_getitem(sample_data):
    dataset = SimpleDataset(sample_data)
    assert dataset[0] == 1
    assert dataset[2] == 3
    assert dataset[-1] == 5

    with pytest.raises(IndexError):
        _ = dataset[10]


def test_simple_dataset_iter(sample_data):
    dataset = SimpleDataset(sample_data)
    assert list(dataset) == sample_data


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1],
        ["a", "b", "c"],
        [{"key": "value"}, {"key": "another_value"}],
    ],
)
def test_simple_dataset_with_different_data_types(data):
    dataset = SimpleDataset(data)
    assert len(dataset) == len(data)
    assert list(dataset) == data


def test_simple_dataset_empty():
    dataset = SimpleDataset([])
    assert len(dataset) == 0
    assert list(dataset) == []

    with pytest.raises(IndexError):
        _ = dataset[0]


def test_simple_dataset_large():
    large_data = list(range(10000))
    dataset = SimpleDataset(large_data)
    assert len(dataset) == 10000
    assert dataset[9999] == 9999


def test_simple_dataset_protocol_methods(sample_data):
    dataset = SimpleDataset(sample_data)

    assert hasattr(dataset, "__iter__")
    assert list(iter(dataset)) == sample_data

    assert hasattr(dataset, "__len__")
    assert len(dataset) == len(sample_data)

    assert hasattr(dataset, "__getitem__")
    assert dataset[2] == sample_data[2]


def test_simple_dataset_split_dataset_by_node(sample_data):
    dataset = SimpleDataset(sample_data)
    shard = dataset.split_dataset_by_node(world_size=2, rank=0)
    assert len(shard) == 3
    assert list(shard) == [1, 2, 3]

    shard = dataset.split_dataset_by_node(world_size=2, rank=1)
    assert len(shard) == 2
    assert list(shard) == [4, 5]


def test_simple_dataset_shuffle(sample_data):
    dataset = SimpleDataset(sample_data)
    shuffled_dataset = dataset.shuffle(jax.random.PRNGKey(0))
    assert list(shuffled_dataset) != sample_data
    assert sorted(shuffled_dataset) == sample_data
