# Simple

The Simple is a simple dataset that stores all underlying items in a list in memory. This is a simple
dataset and is only useful for small datasets and debugging.

```python title="Creating an in-memory dataset"
from loadax import Simple

dataset = Simple([1, 2, 3, 4, 5])

for i in range(len(dataset)):
    print(dataset.get(i))

#> 1
#> 2
#> 3
#> 4
#> 5
```

::: loadax.dataset.simple.SimpleDataset
    selection:
      members: false