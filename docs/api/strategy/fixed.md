# Fixed Batch Strategy

The FixedBatchStrategy is a simple batch strategy that batches data into fixed size batches. This ensures all
batches are the same size, except for the last batch which may be smaller.

```python title="Creating a fixed batch strategy"
from loadax import Batcher, InMemoryDataset, FixedBatchStrategy

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)
batch_strategy = FixedBatchStrategy(batch_size=2)

for batch in batch_strategy.batch(dataset):
    print(batch)

#> [1, 2]
#> [3, 4]
#> [5]
```

::: loadax.strategy.FixedBatchStrategy
    selection:
      members: false
