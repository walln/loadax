import jax.random

from loadax import (
    InMemoryDataset,
    SampledDatasetWithoutReplacement,
    SampledDatasetWithReplacement,
)


def test_sampled_dataset_without_replacement():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3, key)

    samples = [sampled_dataset.get(i) for i in range(3)]
    assert len(sampled_dataset) == 3
    assert len(samples) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in samples)
    assert sampled_dataset.get(3) is None


def test_sampled_dataset_without_replacement_no_duplicates():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithoutReplacement(dataset, 5, key)

    samples = [sampled_dataset.get(i) for i in range(5)]
    assert len(sampled_dataset) == 5
    assert len(samples) == 5
    assert sorted(samples) == [1, 2, 3, 4, 5]
    assert sampled_dataset.get(5) is None


def test_sampled_dataset_with_replacement():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)

    samples = [sampled_dataset.get(i) for i in range(3)]
    assert len(sampled_dataset) == 3
    assert len(samples) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in samples)
    assert sampled_dataset.get(3) is None


def test_sampled_dataset_with_replacement_all_indices():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithReplacement(dataset, 10, key)

    samples = [sampled_dataset.get(i) for i in range(10)]
    assert len(sampled_dataset) == 10
    assert len(samples) == 10
    assert all(item in [1, 2, 3, 4, 5] for item in samples)
    assert sampled_dataset.get(10) is None


def test_sampled_dataset_without_replacement_edge_case():
    dataset = InMemoryDataset([1, 2])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithoutReplacement(dataset, 2, key)

    samples = [sampled_dataset.get(i) for i in range(2)]
    assert len(sampled_dataset) == 2
    assert len(samples) == 2
    assert sorted(samples) == [1, 2]
    assert sampled_dataset.get(2) is None


def test_sampled_dataset_with_replacement_edge_case():
    dataset = InMemoryDataset([1, 2])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithReplacement(dataset, 2, key)

    samples = [sampled_dataset.get(i) for i in range(2)]
    assert len(sampled_dataset) == 2
    assert len(samples) == 2
    assert all(item in [1, 2] for item in samples)
    assert sampled_dataset.get(2) is None


def test_sampled_dataset_without_replacement_empty():
    dataset = InMemoryDataset([])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithoutReplacement(dataset, 3, key)

    assert len(sampled_dataset) == 3
    assert sampled_dataset.get(0) is None
    assert sampled_dataset.get(1) is None
    assert sampled_dataset.get(2) is None


def test_sampled_dataset_with_replacement_empty():
    dataset = InMemoryDataset([])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)

    assert len(sampled_dataset) == 3
    assert sampled_dataset.get(0) is None
    assert sampled_dataset.get(1) is None
    assert sampled_dataset.get(2) is None


def test_sampled_dataset_without_replacement_single_element():
    dataset = InMemoryDataset([42])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithoutReplacement(dataset, 1, key)

    assert len(sampled_dataset) == 1
    assert sampled_dataset.get(0) == 42
    assert sampled_dataset.get(1) is None


def test_sampled_dataset_with_replacement_single_element():
    dataset = InMemoryDataset([42])
    key = jax.random.PRNGKey(0)
    sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)

    assert len(sampled_dataset) == 3
    assert sampled_dataset.get(0) == 42
    assert sampled_dataset.get(1) == 42
    assert sampled_dataset.get(2) == 42
    assert sampled_dataset.get(3) is None


def test_sampled_dataset_without_replacement_repeated_sampling():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    sampled_dataset1 = SampledDatasetWithoutReplacement(dataset, 3, key)
    sampled_dataset2 = SampledDatasetWithoutReplacement(dataset, 3, key)

    samples1 = [sampled_dataset1.get(i) for i in range(3)]
    samples2 = [sampled_dataset2.get(i) for i in range(3)]

    assert len(sampled_dataset1) == 3
    assert len(sampled_dataset2) == 3
    assert len(samples1) == 3
    assert len(samples2) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in samples1)
    assert all(item in [1, 2, 3, 4, 5] for item in samples2)
    assert set(samples1) == set(samples2)


def test_sampled_dataset_without_replacement_repeated_sampling_shared_key():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    sampled_dataset1 = SampledDatasetWithoutReplacement(dataset, 3, key)
    sampled_dataset2 = SampledDatasetWithoutReplacement(dataset, 3, subkey)

    samples1 = [sampled_dataset1.get(i) for i in range(3)]
    samples2 = [sampled_dataset2.get(i) for i in range(3)]

    assert len(sampled_dataset1) == 3
    assert len(sampled_dataset2) == 3
    assert len(samples1) == 3
    assert len(samples2) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in samples1)
    assert all(item in [1, 2, 3, 4, 5] for item in samples2)
    assert set(samples1) != set(samples2)


def test_sampled_dataset_with_replacement_repeated_sampling():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    sampled_dataset1 = SampledDatasetWithReplacement(dataset, 3, key)
    sampled_dataset2 = SampledDatasetWithReplacement(dataset, 3, subkey)

    samples1 = [sampled_dataset1.get(i) for i in range(3)]
    samples2 = [sampled_dataset2.get(i) for i in range(3)]

    assert len(sampled_dataset1) == 3
    assert len(sampled_dataset2) == 3
    assert len(samples1) == 3
    assert len(samples2) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in samples1)
    assert all(item in [1, 2, 3, 4, 5] for item in samples2)
    assert samples1 != samples2


def test_sampled_dataset_with_replacement_repeated_sampling_shared_key():
    dataset = InMemoryDataset([1, 2, 3, 4, 5])
    key = jax.random.PRNGKey(1)
    sampled_dataset1 = SampledDatasetWithReplacement(dataset, 3, key)
    sampled_dataset2 = SampledDatasetWithReplacement(dataset, 3, key)

    samples1 = [sampled_dataset1.get(i) for i in range(3)]
    samples2 = [sampled_dataset2.get(i) for i in range(3)]

    assert len(sampled_dataset1) == 3
    assert len(sampled_dataset2) == 3
    assert len(samples1) == 3
    assert len(samples2) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in samples1)
    assert all(item in [1, 2, 3, 4, 5] for item in samples2)
    assert samples1 == samples2
