# Multi-Host Training

Loadax provides a simple interface for defining your dataloading strategy for distributed training. This means that you can easily train your models on multiple hosts, and load data in parallel across multiple hosts. Loadax also provides a few common sharding configurations that can be used out of the box, but you can also create your own sharding configurations using JAX's `Mesh` and `NamedSharding` primitives.

Loadax's `Dataloader` will automatically determine which elements to load on each host within the network ensuring that the data is evenly distributed, and each host only gets the data it needs. This requires no manual configuration, replication, or network topology changes.

```python title="Creating a distributed dataloader"
from loadax import Dataloader, SimpleDataset, ShardedDataset
from loadax.sharding.presets import make_fsdp_mesh_config # or make your own

config = make_fsdp_mesh_config(axis_names=("data", "model"), batch_axis_name="data")
mesh = config.create_device_mesh()

# You can use jax.process_index() to get the rank of the current host and jax.process_count() to get the total number of hosts.
dataset = SimpleDataset([1, 2, 3, 4, 5]).split_dataset_by_node(world_size=2, rank=0)
dataloader = Dataloader(dataset, batch_size=2)

with mesh:
    for batch in dataloader:
        print(batch)
```

See documentation for sharding presets to learn about common configurations such as FSDP and DDP. Or checkout the examples to learn more.
