# MappedDataset

A mapped dataset is a transformation that applies a functional transformation to the underlying dataset. The transformation is applied lazily, the underlying dataset is preserved. Because loadax performs dataloading in the background, this means it is acceptable to perform lightweight data augmentation or transformations on the dataset.

If you have some complicated transformations you may still want to perform them ahead of time.

```python title="Creating a mapped dataset"
from loadax import MappedDataset, InMemoryDataset

def transform(x):
    return x * 2

dataset = MappedDataset(InMemoryDataset([1, 2, 3, 4, 5]), transform)

for i in range(len(dataset)):
    print(dataset.get(i))

#> 2
#> 4
#> 6
#> 8
#> 10
```

::: loadax.dataset.transform.MappedDataset
    selection:
      members: false
