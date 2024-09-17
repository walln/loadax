# Dataloader

The Dataloader is the main interface for loading data into your training loop. The Dataloader is responsible for
defining how to efficiently load data from a dataset and allocate it to the appropriate devices, batches, and
all of the other features that make up proper data loading.

The Dataloader works by spawning background workers to prefetch data into a cache, and then filling batches from
the cache as they become available. The use of background workers allows the dataloader to be highly efficient
and not block the main thread, which is important for training loops. Loadax takes care of the parllelization
details for you, so your dataloading is fast, reliable, and simple. The background cache will load out of order,
as utilizes mutlithreading to load data in parallel, however the actual batches **will** be in order. This is because
loadax ensures deterministic ordering of batches, and the background workers will load batches in the order that
they are requested.

```python title="Creating a dataloader"
from loadax import Dataloader, InMemoryDataset, Batcher
from loadax.strategy import FixedBatchStrategy
from loadax.dataloader.sharding import NoShardingStrategy

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)
dataloader = Dataloader(
    dataset=dataset,
    batcher=batcher,
    strategy=FixedBatchStrategy(batch_size=2),
    num_workers=2,
    prefetch_factor=2,
    sharding_strategy=NoShardingStrategy(),
)
for batch in dataloader:
    print(batch)

#> [1, 2]
#> [3, 4]
#> [5]
```

::: loadax.dataloader.Dataloader
    selection:
      members: false
::: loadax.dataloader.loader.DataloaderIterator
    selection:
      members: false
