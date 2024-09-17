# Batch Strategy

A batch strategy is a simple interface that defines how to collate batches from a dataset. The batch strategy is
responsible for determining how to batch data from a stream of elements. Currently, loadax only supports the
`FixedBatchStrategy`, which batches data into fixed size batches. However, in the future, loadax will support
other batch strategies such as variable batch sizes, and even variable batch sizes that are determined by the
data itself.

```python title="Creating a batch strategy"
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

::: loadax.strategy.BatchStrategy
    selection:
      members: false
