# Multi-Host Training

Loadax provides a simple interface for defining your dataloading strategy for distributed training. This means that you can easily train your models on multiple hosts, and load data in parallel across multiple hosts. Loadax also provides a few common sharding configurations that can be used out of the box, but you can also create your own sharding configurations using JAX's `Mesh` and `NamedSharding` primitives.

Loadax's DistributedDataloader will automatically determine which elements to load on each shard within the network ensuring that the data is evenly distributed, and each node only gets the data it needs. This requires no manual configuration, replication, or network topology changes.

```python title="Creating a distributed dataloader"
from loadax import DataloaderBuilder, InMemoryDataset, Batcher
from loadax.dataloader.loader import Dataloader
from loadax.sharding_utilities import fsdp_sharding

mesh, axis_names = fsdp_sharding()

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)

mesh = Mesh(...)
dataloader = DataloaderBuilder(batcher)
    .batch_size(2)
    .shard(mesh, axis_names[0])
    .build(dataset)
```
