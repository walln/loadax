# Sharding Strategy

A sharding strategy is a simple interface that defines how to shard data across multiple devices and potentially
multiple hosts. In the case of single-node training, this is not necessary, but in the case of distributed training,
the ShardingStrategy informs loadax of how to partition the dataset across multiple hosts. Because loadax can use this
information to pre-compute which data needs to be loaded on each host, it can then only load the data that is needed
on each host, which can significantly improve data loading performance and reduce network traffic.

```python title="Creating a sharding strategy"
from loadax import InMemoryDataset, NoShardingStrategy

dataset = InMemoryDataset([1, 2, 3, 4, 5])
sharding_strategy = NoShardingStrategy()

for shard in sharding_strategy.get_shard_indices(dataset_size=5, shard_id=0, num_shards=1):
    print(shard)

#> range(0, 5)
```

::: loadax.dataloader.sharding.ShardingStrategy
    selection:
      members: false

::: loadax.dataloader.sharding.NoShardingStrategy
    selection:
      members: false

::: loadax.dataloader.sharding.DistributedShardingStrategy
    selection:
      members: false
