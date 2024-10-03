# Huggingface Integration

Loadax provides integration with Huggingface Datasets. This integration allows you to load datasets from the Huggingface hub and use them with loadax.

## Loading a Dataset

To load a dataset from the Huggingface hub, you can use the `from_hub` method of the `HuggingFaceDataset` class. This method takes in the path to the dataset on the Huggingface hub, the name of the dataset, and the split of the dataset. The dataset is lazily loaded from the Huggingface cache.

```python
from loadax.dataset.huggingface import HuggingFaceDataset

dataset = HuggingFaceDataset.from_hub("stanfordnlp/imdb", split="train")
```

Alternatively, you can construct a `HuggingFaceDataset` object directly from a Huggingface Dataset object. This is useful if you want to do some preprocessing and store the results in the Huggingface dataset cache.

```python
from loadax.dataset.huggingface import HuggingFaceDataset
import datasets as hf_datasets

train_data = hf_datasets.load_dataset("stanfordnlp/imdb", split="train")
train_dataset = HuggingFaceDataset(train_data)
```

## Sharding a Dataset

Huggingface datasets natively support sharding, no need to wrap them in a `ShardedDataset`. Instead you can use the `split_dataset_by_node` method to get a shard of the dataset for a given node. This method takes in the world size and the rank of the node and returns a shard of the dataset. The shards are contiguous and consistent for a given `world_size`.

```python
from loadax.dataset.huggingface import HuggingFaceDataset

dataset = HuggingFaceDataset.from_hub("stanfordnlp/imdb", split="train")
shard = dataset.split_dataset_by_node(world_size=2, rank=0)
```

::: loadax.dataset.huggingface.HuggingFaceDataset
        selection:
            members: false
