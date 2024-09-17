# CombinedDataset

A combined dataset is a simple dataset that combines multiple underlying datasets. The combined dataset will return the items from the first underlying dataset, then the items from the second underlying dataset, and so on.

```python title="Creating a combined dataset"
from loadax import CombinedDataset, InMemoryDataset

dataset1 = InMemoryDataset([1, 2, 3, 4, 5])
dataset2 = InMemoryDataset([6, 7, 8, 9, 10])
combined_dataset = CombinedDataset(dataset1, dataset2)

for i in range(len(combined_dataset)):
    print(combined_dataset.get(i))

#> 1
#> 2
#> 3
#> 4
#> 5
#> 6
#> 7
#> 8
#> 9
#> 10
```

::: loadax.dataset.transform.CombinedDataset
    selection:
      members: false
