# Sharding Presets

Loadax provides a few common sharding configurations that can be used out of the box. These presets are strong starting points for your sharding configuration, but you can also create your own sharding configurations using JAX's `Mesh` and `NamedSharding` primitives.

## FSDP Sharding

The FSDP sharding preset is a simple configuration that shards the model across multiple hosts and devices, and the data across multiple hosts. In FSDP training, the model parameters are split across multiple devices and each model shard recieves unique data. This configuration is useful for training models that are too large to fit on a single device.

```python title="Creating an FSDP sharding preset"
from loadax import DataloaderBuilder, InMemoryDataset, Batcher
from loadax.sharding_utilities import fsdp_sharding

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)

mesh, axis_names = fsdp_sharding()

dataloader = DataloaderBuilder(batcher)
    .batch_size(2)
    .shard(mesh, axis_names[0])
    .build(dataset)
```

::: loadax.sharding_utilities.fsdp_sharding
    selection:
      members: false

## DDP Sharding

The DDP sharding preset is a simple configuration that replicates the model across multiple devices and each replica recieves unique data. This configuration is ideal for training smaller models when you have multiple devices available.

```python title="DDP Sharding"
from loadax import DataloaderBuilder, InMemoryDataset, Batcher
from loadax.sharding_utilities import ddp_sharding

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)

mesh, axis_names = ddp_sharding()

dataloader = DataloaderBuilder(batcher)
    .batch_size(2)
    .shard(mesh, axis_names[0])
    .build(dataset)
```

::: loadax.sharding_utilities.ddp_sharding
    selection:
      members: false
