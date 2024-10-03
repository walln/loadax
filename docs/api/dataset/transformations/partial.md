# PartialDataset

A partial dataset is a simple dataset that returns a subset of the underlying dataset. This is useful for testing and debugging.

```python title="Creating a partial dataset"
from loadax import PartialDataset, SimpleDataset

dataset = SimpleDataset([1, 2, 3, 4, 5])
partial_dataset = PartialDataset(dataset, 2, 4)

for i in range(len(partial_dataset)):
    print(partial_dataset.get(i))

#> 2
#> 3
#> 4
```

::: loadax.dataset.partial_dataset.PartialDataset
    selection:
      members: false
