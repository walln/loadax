import jax.numpy as jnp
import pytest

from loadax.dataset.dataset import MappedBatchDataset
from loadax.dataset.simple import SimpleDataset


@pytest.fixture
def sample_data():
    return list(range(100))


@pytest.fixture
def transform_function():
    return lambda batch: [x * 2 for x in batch]


def test_mapped_batch_dataset_initialization(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_batch_dataset = MappedBatchDataset(
        base_dataset, transform_function, batch_size=10
    )
    assert isinstance(mapped_batch_dataset, MappedBatchDataset)
    assert len(mapped_batch_dataset) == 10  # 100 items / 10 batch size


def test_mapped_batch_dataset_len(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_batch_dataset = MappedBatchDataset(
        base_dataset, transform_function, batch_size=10
    )
    assert len(mapped_batch_dataset) == 10

    # Test with non-divisible batch size
    mapped_batch_dataset = MappedBatchDataset(
        base_dataset, transform_function, batch_size=30
    )
    assert len(mapped_batch_dataset) == 4  # Ceil(100 / 30)


def test_mapped_batch_dataset_getitem(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_batch_dataset = MappedBatchDataset(
        base_dataset, transform_function, batch_size=10
    )
    assert mapped_batch_dataset[0] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert mapped_batch_dataset[5] == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]

    with pytest.raises(IndexError):
        _ = mapped_batch_dataset[10]


def test_mapped_batch_dataset_iter(sample_data, transform_function):
    base_dataset = SimpleDataset(sample_data)
    mapped_batch_dataset = MappedBatchDataset(
        base_dataset, transform_function, batch_size=10
    )
    batches = list(mapped_batch_dataset)
    assert len(batches) == 10
    assert batches[0] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert batches[-1] == [180, 182, 184, 186, 188, 190, 192, 194, 196, 198]


def test_mapped_batch_dataset_with_different_data_types():
    base_dataset = SimpleDataset(["a", "b", "c", "d", "e", "f"])

    def transform(batch):
        return [x.upper() for x in batch]

    mapped_batch_dataset = MappedBatchDataset(base_dataset, transform, batch_size=2)
    assert list(mapped_batch_dataset) == [["A", "B"], ["C", "D"], ["E", "F"]]


def test_mapped_batch_dataset_empty():
    base_dataset = SimpleDataset([])
    mapped_batch_dataset = MappedBatchDataset(base_dataset, lambda x: x, batch_size=10)
    assert len(mapped_batch_dataset) == 0
    assert list(mapped_batch_dataset) == []

    with pytest.raises(IndexError):
        _ = mapped_batch_dataset[0]


def test_mapped_batch_dataset_with_jax_transform():
    base_dataset = SimpleDataset(
        [jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6]), jnp.array([7, 8])]
    )

    def transform(batch):
        return [jnp.sum(x) for x in batch]

    mapped_batch_dataset = MappedBatchDataset(base_dataset, transform, batch_size=2)

    assert jnp.array_equal(mapped_batch_dataset[0], jnp.array([3, 7]))
    assert jnp.array_equal(mapped_batch_dataset[1], jnp.array([11, 15]))


def test_mapped_batch_dataset_with_non_divisible_batch():
    base_dataset = SimpleDataset(list(range(10)))

    def transform(batch):
        return [x * 2 for x in batch]

    mapped_batch_dataset = MappedBatchDataset(base_dataset, transform, batch_size=3)

    batches = list(mapped_batch_dataset)
    assert len(batches) == 4
    assert batches == [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18]]


def test_mapped_batch_dataset_large():
    large_data = list(range(10000))
    base_dataset = SimpleDataset(large_data)
    mapped_batch_dataset = MappedBatchDataset(
        base_dataset, lambda batch: [x * 2 for x in batch], batch_size=100
    )
    assert len(mapped_batch_dataset) == 100
    assert mapped_batch_dataset[99][-1] == 19998


def test_mapped_batch_dataset_type_propagation():
    base_dataset = SimpleDataset([1, 2, 3, 4, 5, 6])

    # Transform integers to strings
    def str_transform(batch):
        return [str(x) for x in batch]

    str_mapped_dataset = MappedBatchDataset(base_dataset, str_transform, batch_size=2)
    assert all(isinstance(item, str) for batch in str_mapped_dataset for item in batch)

    # Transform integers to floats
    def float_transform(batch):
        return [float(x) for x in batch]

    float_mapped_dataset = MappedBatchDataset(
        base_dataset, float_transform, batch_size=2
    )
    assert all(
        isinstance(item, float) for batch in float_mapped_dataset for item in batch
    )

    # Transform to complex data structure
    def complex_transform(batch):
        return [{"value": x, "squared": x**2} for x in batch]

    complex_mapped_dataset = MappedBatchDataset(
        base_dataset, complex_transform, batch_size=2
    )
    assert all(
        isinstance(item, dict) for batch in complex_mapped_dataset for item in batch
    )
    assert all(
        "value" in item and "squared" in item
        for batch in complex_mapped_dataset
        for item in batch
    )


def test_simple_dataset_batch_map():
    base_dataset = SimpleDataset(list(range(10)))

    def transform(batch):
        return [x * 2 for x in batch]

    # Test with divisible batch size
    mapped_dataset = base_dataset.map_batch(transform, batch_size=2)
    assert isinstance(mapped_dataset, MappedBatchDataset)
    assert len(mapped_dataset) == 5
    assert list(mapped_dataset) == [[0, 2], [4, 6], [8, 10], [12, 14], [16, 18]]

    # Test with non-divisible batch size
    mapped_dataset = base_dataset.map_batch(transform, batch_size=3)
    assert len(mapped_dataset) == 4
    assert list(mapped_dataset) == [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18]]

    # Test with batch size larger than dataset
    mapped_dataset = base_dataset.map_batch(transform, batch_size=15)
    assert len(mapped_dataset) == 1
    assert list(mapped_dataset) == [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]]

    # Test with empty dataset
    empty_dataset = SimpleDataset([])
    mapped_empty = empty_dataset.map_batch(transform, batch_size=2)
    assert len(mapped_empty) == 0
    assert list(mapped_empty) == []

    # Test type propagation
    def str_transform(batch):
        return [str(x) for x in batch]

    str_mapped = base_dataset.map_batch(str_transform, batch_size=2)
    assert all(isinstance(item, str) for batch in str_mapped for item in batch)
