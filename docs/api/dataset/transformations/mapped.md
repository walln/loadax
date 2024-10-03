# MappedDataset

A mapped dataset is a transformation that applies a functional transformation to the underlying dataset. The transformation is applied lazily, the underlying dataset is preserved. Because loadax performs dataloading in the background, this means it is acceptable to perform lightweight data augmentation or transformations on the dataset.

If you have some complicated transformations you may still want to perform them ahead of time.

```python title="Creating a mapped dataset"
from loadax import MappedDataset, SimpleDataset

def transform(x):
    return x * 2

dataset = MappedDataset(SimpleDataset([1, 2, 3, 4, 5]), transform)

for i in range(len(dataset)):
    print(dataset.get(i))

#> 2
#> 4
#> 6
#> 8
#> 10
```

::: loadax.dataset.dataset.MappedDataset
    selection:
      members: false


# MappedBatchDataset

A mapped batch dataset is a transformation that applies a functional transformation to the underlying dataset. The transformation is applied lazily, the underlying dataset is preserved. Because loadax performs dataloading in the background, this means it is acceptable to perform lightweight data augmentation or transformations on the dataset. 

Similar to the `MappedDataset`, but the transformation is applied to batches of items instead of individual items. This is useful for performing batch-level transformations such as data augmentation or working with more expensive transformations that can be vectorized.

```python title="Creating a mapped batch dataset"
from loadax import MappedBatchDataset, SimpleDataset

def transform(batch):
    return [item * 2 for item in batch]

dataset = MappedBatchDataset(SimpleDataset([1, 2, 3, 4, 5]), transform)

for i in range(len(dataset)):
    print(dataset.get(i))

#> 2
#> 4
#> 6
#> 8
#> 10
```

::: loadax.dataset.dataset.MappedBatchDataset
    selection:
      members: false
