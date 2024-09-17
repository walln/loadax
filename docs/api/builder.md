# DataloaderBuilder

Most of the time, you just want to create a dataloader, set a few options, and be done with it. The `DataloaderBuilder` is a convenient way to do this. It is a fluent builder that allows you to chain together methods to create a dataloader instead of configuring each option individually.

```python title="Creating a dataloader"
from loadax import DataloaderBuilder, InMemoryDataset, Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)
dataloader = DataloaderBuilder(batcher).batch_size(2).build(dataset)

for batch in dataloader:
    print(batch)

#> [1, 2]
#> [3, 4]
#> [5]
```

::: loadax.DataloaderBuilder
    selection:
      members: false
