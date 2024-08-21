"""This is a simple example of how you can compose a distributed FSDP training loop with Loadax.

This example is not meant to be a complete training loop, but rather a demonstration of how you can leverage
Jax's powerful parallelization primitives in combination with Loadax to achieve distributed training.

Loadax does not lock you into any particular sharding strategy, but instead allows you to define your own
sharding strategy and optimize for your architecture, network topology, device placement, etc. In fact there is no reason that you cannot create new training paradigms ontop of Loadax, such as asynchronous training, or training on
heterogeneous devices.

This example is a simple demonstration of the FSDP training strategy, which shards the model across the devices
and supplies each device with a local shard of the global data. In a distributed setting, this means that both the
data and model are sharded across nodes/devices, such that the model may not be purely replicated across the devices.

If you want to learn more about FSDP, the UvA DL Notebooks have a great tutorial here: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html I would recommend the entire tutorials as a great resource for understanding some of the concepts behind distributed training.

"""

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loadax.dataloader.distributed import DistributedDataLoader, JaxShardingStrategy
from loadax import InMemoryDataset, Batcher
from loadax.strategy import FixedBatchStrategy

def test_distributed_dataloader_with_parameter_sharding():
    # Simulate a simple dataset with 10 items
    dataset = InMemoryDataset(items=[jnp.array([i]) for i in range(10)])
    batcher = Batcher(lambda items: jnp.stack(items))
    batch_strategy = FixedBatchStrategy(batch_size=1)

    # Create a global mesh across the devices
    devices = jax.devices()
    # In real FSDP training, you want to reshape devices based on your network topology
    # But for demonstration purposes we'll just add a 'model' axis for parameter sharding
    mesh = Mesh(devices, ('data', 'model'))

    # Sharding specification: parameters are sharded across the 'model' axis, and data across the 'data' axis
    param_sharding_spec = PartitionSpec('model')
    data_sharding_spec = PartitionSpec('data')

    # This is the shard_id, which is used to determine which jax process this loader instance is running on
    shard_id = 0
    num_shards = len(devices)
    sharding_strategy = JaxShardingStrategy(mesh, data_sharding_spec)

    # Create the DistributedDataLoader
    dataloader = DistributedDataLoader(
        dataset=dataset,
        batcher=batcher,
        strategy=batch_strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    # Define a simple model function that works with sharded parameters
    def simple_model(params, x):
        return params * x  # Example of a simple linear model

    # Initialize the model parameters and shard them across devices
    params = jnp.array([1.0])
    sharded_params = jax.device_put(jnp.array_split(params, num_shards), NamedSharding(mesh, param_sharding_spec))

    # Define a function to compute the loss
    def loss_fn(params, x):
        predictions = simple_model(params, x)
        return jnp.mean((predictions - x) ** 2)

    # Define a function to compute the gradients using pjit
    def compute_gradients(params, batch):
        grads = jax.grad(loss_fn)(params, batch)
        return grads

    # Use pjit to handle both gradient computation and parameter updates with sharded parameters
    for batch in dataloader:
        # You still have total control over array placement, so you can decide how to parallelize your
        # intra-node batch across the local devices, when to synchronize across nodes, etc. This is a really
        # powerful set of primitives and you do not need to use pjit, this is just an example of how you can
        # leverage Jax's powerful parallelization primitives in combination with Loadax to achieve distributed
        # training.
        local_sharded_batch = jax.device_put(jnp.array(batch), NamedSharding(mesh, data_sharding_spec))

        # Compute gradients with pjit, allowing for parameter sharding across devices
        grads = pjit(compute_gradients, in_axis_resources=(param_sharding_spec, data_sharding_spec),
                     out_axis_resources=param_sharding_spec)(sharded_params, local_sharded_batch)

        # Update the parameters using the mean gradients across the replicas
        sharded_params -= pjit(lambda p, g: 0.1 * g, in_axis_resources=(param_sharding_spec, param_sharding_spec),
                               out_axis_resources=param_sharding_spec)(sharded_params, grads)

        # Optionally, you can also perform a forward pass on the local shard
        sharded_result = pjit(simple_model, in_axis_resources=(param_sharding_spec, data_sharding_spec),
                              out_axis_resources=data_sharding_spec)(sharded_params, local_sharded_batch)
        
        # Validate that the results are correct
        for original, result in zip(batch, sharded_result):
            assert jnp.all(result == original * 2)

    # Aggregate the update parameters across the replicas
    final_params = jax.tree_map(lambda x: x[0], sharded_params)
    print(f"Final params: {final_params}")

test_distributed_dataloader_with_parameter_sharding()
