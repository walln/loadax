# RangeDataset

The RangeDataset is a simple dataset that returns a range of integers. This is useful for testing and debugging.

```python title="Creating a range dataset"
from loadax import RangeDataset

dataset = RangeDataset(start=0, end=10, step=2)

for i in range(len(dataset)):
    print(dataset.get(i))

#> 0
#> 2
#> 4
#> 6
#> 8
```

::: loadax.dataset.range.RangeDataset
    selection:
      members: false
