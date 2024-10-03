# ShardedDataset

A dataset that implements the `Shardable` protocol. This allows you to partition your dataset across multiple hosts.

```python title="Creating a sharded dataset"
from loadax import SimpleDataset, ShardedDataset

dataset = SimpleDataset([1, 2, 3, 4, 5])
sharded_dataset = ShardedDataset(dataset, num_shards=2, shard_id=0)
```

The shardable protocol requires you to implement the `split_dataset_by_node` method. This method should take in the `world_size` and the `rank` of the current host and return a shard of the dataset for that host.

Loadax's provided datasets will implement this method for you.

::: loadax.dataset.sharded_dataset.ShardedDataset
    selection:
      members: false

::: loadax.dataset.sharded_dataset.Shardable
    selection:
      members: false
