"""Example of how to use the distributed dataloader."""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loadax.dataloader.distributed import DistributedDataLoader, JaxShardingStrategy
from loadax import InMemoryDataset, Batcher
from loadax.strategy import FixedBatchStrategy


def test_distributed_dataloader_on_logical_devices():
    # Simulate a simple dataset with 10 items
    dataset = InMemoryDataset(data=[jnp.array([i]) for i in range(10)])
    batcher = Batcher(lambda items: jnp.stack(items))
    batch_strategy = FixedBatchStrategy(batch_size=2)

    # Create logical devices by slicing the actual physical device(s)
    logical_device_count = 4  # Simulate 4 logical devices
    logical_devices = jax.devices()[:logical_device_count]  # Use the first few devices

    # Create a global mesh across these logical devices
    mesh = Mesh(logical_devices, ('data',))
    sharding_spec = PartitionSpec('data')

    # Simulate sharding strategy with logical devices
    shard_id = 0  # Simulate this as shard 0
    num_shards = logical_device_count  # Assume the number of shards equals logical devices
    sharding_strategy = JaxShardingStrategy(mesh, sharding_spec)

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

    # Define a simple model function
    def simple_model(x):
        return x * 2

    # Run through the dataloader and apply the model using pjit
    for batch in dataloader:
        local_sharded_batch = jax.device_put(jnp.array(batch), NamedSharding(mesh, sharding_spec))
        
        # Apply the simple model using pjit
        sharded_result = jax.jit(simple_model)(local_sharded_batch)
        
        # Validate that the results are correct
        for original, result in zip(batch, sharded_result):
            assert jnp.all(result == original * 2)

# Run the test
test_distributed_dataloader_on_logical_devices()
