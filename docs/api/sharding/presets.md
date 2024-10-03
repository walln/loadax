# Sharding Presets

Loadax provides a few common sharding configurations that can be used out of the box. These presets are strong starting points for your sharding configuration, but you can also create your own sharding configurations using JAX's `Mesh` and `NamedSharding` primitives.

## FSDP Sharding

The FSDP sharding preset is a simple configuration that shards the model across multiple hosts and devices, and the data across multiple hosts. In FSDP training, the model parameters are split across multiple devices and each model shard recieves unique data. This configuration is useful for training models that are too large to fit on a single device.

```python title="Creating an FSDP sharding preset"
from loadax import Dataloader, SimpleDataset
from loadax.sharding.placement import host_to_global_device_array
from loadax.sharding.presets import make_fsdp_mesh_config

dataset = SimpleDataset([jax.array([i]) for i in range(100)])

mesh_config = make_fsdp_mesh_config(
        mesh_axis_names=("data", "model"), batch_axis_names="data"
)
mesh = mesh_config.create_device_mesh()

dataloader = Dataloader(dataset, batch_size=8)

# Create your model, optimizer, metrics, and a train_step function
# sharding your model parameters. See your framework's documentation
# for how to configure FSDP or see examples/fsdp.py for an example 
# using flax's NNX api!
...

with mesh:
    for local_batch in dataloader:
        # Stack the batch of arrays into a single array
        local_batch = jnp.stack(local_batch)

        # Convert the local batch to a global device array
        global_batch = host_to_global_device_array(local_batch)

        # Use jax.lax.with_sharding_constraint to specify the sharding of the input
        sharded_batch = jax.lax.with_sharding_constraint(
            global_batch, jax.sharding.PartitionSpec(mesh_rules.data)
        )

        # let jax.jit handle the movement of data across devices
        loss = train_step(model, optimizer, metrics, sharded_batch)
```

::: loadax.sharding.presets.fsdp.make_fsdp_mesh_config
    selection:
      members: false

## DDP Sharding

The DDP sharding preset is a simple configuration that replicates the model across multiple devices and each replica recieves unique data. This configuration is ideal for training smaller models when you have multiple devices available.

```python title="DDP Sharding"
from loadax import Dataloader, SimpleDataset
from loadax.sharding.placement import host_to_global_device_array
from loadax.sharding.presets import make_ddp_mesh_config

dataset = SimpleDataset([jax.array([i]) for i in range(100)])

mesh_config = make_ddp_mesh_config(
        mesh_axis_names=("data",), batch_axis_names="data"
)
mesh = mesh_config.create_device_mesh()

dataloader = Dataloader(dataset, batch_size=8)

# Create your model, optimizer, metrics, and a train_step function
# letting jax.pmap handle replicate the model and sharding the data.
...

with mesh:
    for local_batch in dataloader:
        # Stack the batch of arrays into a single array
        local_batch = jnp.stack(local_batch)

        # Convert the local batch to a global device array
        global_batch = host_to_global_device_array(local_batch)

        # Use jax.lax.with_sharding_constraint to specify the sharding of the input
        sharded_batch = jax.lax.with_sharding_constraint(
            global_batch, jax.sharding.PartitionSpec(mesh_rules.data)
        )

        # Use pmap to replicate the computation across all devices
        loss = pmap_train_step(model, optimizer, metrics, sharded_batch)


```

::: loadax.sharding.presets.ddp.make_ddp_mesh_config
    selection:
      members: false
