"""This is a simple example of how you can compose a distributed training loop with Loadax.

This example is not meant to be a complete training loop, but rather a demonstration of how you can leverage
Jax's powerful parallelization primitives in combination with Loadax to achieve distributed training.

Loadax does not lock you into any particular sharding strategy, but instead allows you to define your own
sharding strategy and optimize for your architecture, network topology, device placement, etc. In fact there 
is no reason that you cannot create new training paradigms ontop of Loadax or fully customize your training topology.

This example is a simple demonstration of the DataParallel training strategy, which replicates the model across
the devices and supplies each device with a local shard of the global data. In a distributed setting, this means
that each node will have a copy of the model and a sharded section of the data, the node then performs the 
training on the local shard as you normally would on a single node, and then the gradients are aggregated 
across the nodes to update the model parameters.
"""

import jax
import jax.numpy as jnp
from loadax.dataloader.loader import Dataloader
from loadax.dataloader.sharding import DistributedShardingStrategy
from loadax import InMemoryDataset, Batcher
from loadax.strategy import FixedBatchStrategy
from loadax.sharding_utilities import ddp_sharding

# If using multiple hosts, you will need to initialize the jax distributed runtime
# jax.distributed.initialize()


def test_distributed_dataloader_on_logical_devices():
    # Simulate a simple dataset with 128 items
    dataset = InMemoryDataset(items=[jnp.array([i]) for i in range(128)])
    batcher = Batcher(lambda items: jnp.stack(items))
    batch_strategy = FixedBatchStrategy(batch_size=1)

    # Create a global mesh across the logical devices, loadax provides a preset for DDP
    # this preset will ensure each node gets a partition of the data that is precomputed
    # to match the sharding that will be used to supply the local devices with the data.
    # You can also create your own sharding configurations.
    mesh, axis_names = ddp_sharding()

    # Create a sharding strategy that loadax can use to determine how to partition the dataset 
    # and orchestrate the distributed data loading
    sharding_strategy = DistributedShardingStrategy(mesh, 'data')

    # Create the Dataloader
    dataloader = Dataloader(
        dataset=dataset,
        batcher=batcher,
        strategy=batch_strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
    )

    # Define a simple model function
    def simple_model(x):
        return x * 2
    
    # Placeholder for computing your loss, depends on your model/objective
    def loss_fn(params, x):
        predictions = simple_model(x)
        return jnp.mean((predictions - x) ** 2)
    
    # Intialize our model parameters
    params = jnp.array([1.0])

    # We are using a data parallel strategy, so we need to replicate the model across the devices
    # check your NN framework's documentation for how to do this
    replicated_params = jnp.broadcast_to(params, (len(jax.devices()),) + params.shape)

    # Define a function to compute the gradients
    def compute_gradients(params, batch):
        grads = jax.grad(loss_fn)(params, batch)
        return grads

    for batch in dataloader:
        # Convert the batch to a local shard this will change based on what your batch looks like
        # in our case batch is just a jax array, but your could have pytrees, classes, etc.
        local_sharded_batch = sharding_strategy.distribute_global_batch(batch)

        # Compute the gradients (You can use pjit automatic parallelization here instead if you prefer)
        # we'll use pmap here for demonstration purposes as we can then easily parallelize the per-node
        # batch across all local devices such that the global batch size is replicated across all devices
        # globally. For example a global batch of 32 on 4 nodes would split the global batch into 4 shards
        # across the 4 distributed nodes, and then each batch of 8 elements is pmapped across the 8 local devices
        grads = jax.pmap(compute_gradients, axis_name='data')(replicated_params, local_sharded_batch)

        # Update the parameters using the mean gradients across the replicas (You can use any optimizer here)
        replicated_params -= jax.pmap(lambda p, g: 0.1 * jax.lax.pmean(g, 'data'), axis_name='data')(replicated_params, grads)


        # You may also want to just perform the forward pass on the local shard:
        # Apply the simple model using pjit/pmap (depending on your use case)
        sharded_result = jax.jit(simple_model)(local_sharded_batch)
        # Validate that the results are correct
        for original, result in zip(batch, sharded_result):
            assert jnp.all(result == original * 2)

    # Aggregate the update parameters across the replicas
    final_params = jax.tree.map(lambda x: x[0], replicated_params)
    print(f"Final params: {final_params}")

test_distributed_dataloader_on_logical_devices()
