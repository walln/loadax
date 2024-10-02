import jax.numpy as jnp
import pytest

from loadax.dataset.dataset import MappedDataset
from loadax.dataset.simple import SimpleDataset


@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def transform_function():
    return lambda x: x * 2


def test_mapped_dataset_initialization(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_dataset = MappedDataset(base_dataset, transform_function)
    assert isinstance(mapped_dataset, MappedDataset)
    assert len(mapped_dataset) == len(sample_data)


def test_mapped_dataset_len(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_dataset = MappedDataset(base_dataset, transform_function)
    assert len(mapped_dataset) == 5


def test_mapped_dataset_getitem(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_dataset = MappedDataset(base_dataset, transform_function)
    assert mapped_dataset[0] == 2
    assert mapped_dataset[2] == 6
    assert mapped_dataset[-1] == 10

    with pytest.raises(IndexError):
        _ = mapped_dataset[10]


def test_mapped_dataset_iter(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_dataset = MappedDataset(base_dataset, transform_function)
    assert list(mapped_dataset) == [2, 4, 6, 8, 10]


@pytest.mark.parametrize(
    ("data", "transform"),
    [
        ([], lambda x: x),
        ([1], lambda x: x + 1),
        (["a", "b", "c"], lambda x: x.upper()),
        ([{"key": "value"}, {"key": "another_value"}], lambda x: x["key"]),
    ],
)
def test_mapped_dataset_with_different_data_types(data, transform):
    base_dataset = SimpleDataset(data)
    mapped_dataset = MappedDataset(base_dataset, transform)
    assert len(mapped_dataset) == len(data)
    assert list(mapped_dataset) == [transform(item) for item in data]


def test_mapped_dataset_empty():
    base_dataset = SimpleDataset([])
    mapped_dataset = MappedDataset(base_dataset, lambda x: x)
    assert len(mapped_dataset) == 0
    assert list(mapped_dataset) == []

    with pytest.raises(IndexError):
        _ = mapped_dataset[0]


def test_mapped_dataset_large():
    large_data = list(range(10000))
    base_dataset = SimpleDataset(large_data)
    mapped_dataset = MappedDataset(base_dataset, lambda x: x * 2)
    assert len(mapped_dataset) == 10000
    assert mapped_dataset[9999] == 19998


def test_mapped_dataset_protocol_methods(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_dataset = MappedDataset(base_dataset, transform_function)

    assert hasattr(mapped_dataset, "__iter__")
    assert list(iter(mapped_dataset)) == [2, 4, 6, 8, 10]

    assert hasattr(mapped_dataset, "__len__")
    assert len(mapped_dataset) == len(sample_data)

    assert hasattr(mapped_dataset, "__getitem__")
    assert mapped_dataset[2] == 6


def test_mapped_dataset_with_jax_transform():
    base_dataset = SimpleDataset([jnp.array([1, 2]), jnp.array([3, 4])])
    mapped_dataset = MappedDataset(base_dataset, lambda x: jnp.sum(x))

    assert jnp.array_equal(mapped_dataset[0], jnp.array(3))
    assert jnp.array_equal(mapped_dataset[1], jnp.array(7))


def test_mapped_dataset_chaining():
    base_dataset = SimpleDataset([1, 2, 3, 4, 5])
    first_map = MappedDataset(base_dataset, lambda x: x * 2)
    second_map = MappedDataset(first_map, lambda x: x + 1)

    assert list(second_map) == [3, 5, 7, 9, 11]


def test_mapped_dataset_type_propagation():
    base_dataset = SimpleDataset([1, 2, 3, 4, 5])

    # Transform integers to strings
    str_mapped_dataset = MappedDataset(base_dataset, lambda x: str(x))
    assert all(isinstance(item, str) for item in str_mapped_dataset)

    # Transform integers to floats
    float_mapped_dataset = MappedDataset(base_dataset, lambda x: float(x))
    assert all(isinstance(item, float) for item in float_mapped_dataset)

    # Transform to complex data structure
    complex_mapped_dataset = MappedDataset(
        base_dataset, lambda x: {"value": x, "squared": x**2}
    )
    assert all(isinstance(item, dict) for item in complex_mapped_dataset)
    assert all("value" in item and "squared" in item for item in complex_mapped_dataset)


def test_simple_dataset_map():
    base_dataset = SimpleDataset([1, 2, 3, 4, 5])

    def transform(x):
        return x * 2

    # Test basic mapping
    mapped_dataset = base_dataset.map(transform)
    assert isinstance(mapped_dataset, MappedDataset)
    assert list(mapped_dataset) == [2, 4, 6, 8, 10]

    # Test length preservation
    assert len(mapped_dataset) == len(base_dataset)

    # Test indexing
    assert mapped_dataset[2] == 6

    # Test with more complex transform
    def complex_transform(x):
        return {"original": x, "doubled": x * 2}

    complex_mapped_dataset = base_dataset.map(complex_transform)
    assert complex_mapped_dataset[0] == {"original": 1, "doubled": 2}

    # Test with empty dataset
    empty_dataset = SimpleDataset([])
    empty_mapped = empty_dataset.map(transform)
    assert len(empty_mapped) == 0
    assert list(empty_mapped) == []

    # Test type propagation
    def str_transform(x):
        return str(x)

    str_mapped = base_dataset.map(str_transform)
    assert all(isinstance(item, str) for item in str_mapped)

    # Test chaining of map operations
    chained_dataset = base_dataset.map(lambda x: x * 2).map(lambda x: x + 1)
    assert list(chained_dataset) == [3, 5, 7, 9, 11]
