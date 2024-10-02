import pytest
from datasets import Dataset as HFDataset

from loadax.experimental.dataset.huggingface import HuggingFaceDataset


@pytest.fixture
def sample_hf_dataset():
    return HFDataset.from_dict({"text": ["Hello", "World", "Test"], "label": [0, 1, 0]})


def test_huggingface_dataset_initialization(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)
    assert isinstance(dataset, HuggingFaceDataset)
    assert len(dataset) == len(sample_hf_dataset)


def test_huggingface_dataset_len(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)
    assert len(dataset) == 3


def test_huggingface_dataset_getitem(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)
    assert dataset[0] == {"text": "Hello", "label": 0}
    assert dataset[1] == {"text": "World", "label": 1}
    assert dataset[2] == {"text": "Test", "label": 0}

    with pytest.raises(IndexError):
        _ = dataset[3]


def test_huggingface_dataset_iter(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)
    assert list(dataset) == [
        {"text": "Hello", "label": 0},
        {"text": "World", "label": 1},
        {"text": "Test", "label": 0},
    ]


def test_huggingface_dataset_split_dataset_by_node(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)
    shard = dataset.split_dataset_by_node(world_size=2, rank=0)
    assert len(shard) == 2
    assert list(shard) == [
        {"text": "Hello", "label": 0},
        {"text": "World", "label": 1},
    ]

    shard = dataset.split_dataset_by_node(world_size=2, rank=1)
    assert len(shard) == 1
    assert list(shard) == [{"text": "Test", "label": 0}]


@pytest.mark.parametrize(
    ("path", "name", "split"),
    [
        # ("glue", "ax", "test"),
        # disabled because even though this is a small split the memmap makes
        # running tests slow
        ("imdb", None, "test"),
    ],
)
def test_huggingface_dataset_from_hub(path, name, split):
    dataset = HuggingFaceDataset.from_hub(path=path, name=name, split=split)
    assert isinstance(dataset, HuggingFaceDataset)
    assert len(dataset) > 0


def test_huggingface_dataset_protocol_methods(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)

    assert hasattr(dataset, "__iter__")
    assert list(iter(dataset)) == [
        {"text": "Hello", "label": 0},
        {"text": "World", "label": 1},
        {"text": "Test", "label": 0},
    ]

    assert hasattr(dataset, "__len__")
    assert len(dataset) == 3

    assert hasattr(dataset, "__getitem__")
    assert dataset[1] == {"text": "World", "label": 1}


def test_huggingface_dataset_empty():
    empty_hf_dataset = HFDataset.from_dict({"text": [], "label": []})
    dataset = HuggingFaceDataset(empty_hf_dataset)
    assert len(dataset) == 0
    assert list(dataset) == []

    with pytest.raises(IndexError):
        _ = dataset[0]


@pytest.mark.parametrize(
    ("dataset_size", "world_size", "rank", "expected_shard_size"),
    [
        (10000, 2, 0, 5000),  # Even split
        (10000, 2, 1, 5000),  # Even split, second shard
        (10001, 2, 0, 5001),  # Uneven split, first shard gets extra
        (10001, 2, 1, 5000),  # Uneven split, second shard
        (10000, 3, 0, 3334),  # Uneven split with 3 shards
        (10000, 3, 1, 3333),  # Uneven split with 3 shards
        (10000, 3, 2, 3333),  # Uneven split with 3 shards
    ],
)
def test_huggingface_dataset_shardable(
    dataset_size, world_size, rank, expected_shard_size
):
    # Create a dataset of the specified size
    large_dataset = HFDataset.from_dict(
        {
            "text": [f"Example {i}" for i in range(dataset_size)],
            "label": [i % 3 for i in range(dataset_size)],
        }
    )

    dataset = HuggingFaceDataset(large_dataset).split_dataset_by_node(
        world_size=world_size, rank=rank
    )
    assert len(dataset) == expected_shard_size

    # Check the content of the shard
    div = dataset_size // world_size
    mod = dataset_size % world_size
    start = div * rank + min(rank, mod)
    end = start + div + (1 if rank < mod else 0)
    expected_shard = [
        {"text": f"Example {i}", "label": i % 3} for i in range(start, end)
    ]
    assert len(expected_shard) == expected_shard_size
    assert list(dataset) == expected_shard


def test_huggingface_dataset_dataset(sample_hf_dataset):
    dataset = HuggingFaceDataset(sample_hf_dataset)
    assert isinstance(dataset.dataset, HFDataset)
    assert len(dataset.dataset) == len(sample_hf_dataset)
    assert list(dataset.dataset) == [
        {"text": "Hello", "label": 0},
        {"text": "World", "label": 1},
        {"text": "Test", "label": 0},
    ]
