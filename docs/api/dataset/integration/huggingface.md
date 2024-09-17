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

Currently, leveraging the Huggingface native sharding functionality is not supported. This is a work in progress,
loadax will support it in the future, but for now avoid `dataset.shard` and use the `ShardingStrategy` interface
or allow `DataloaderBuilder` to automatically shard the dataset. This is because huggingface's sharding algorithm is
not guaranteed to be consistent with the sharding strategy that loadax leverages to optimize for JAX's device
placement.

::: loadax.dataset.huggingface.HuggingFaceDataset
        selection:
            members: false
