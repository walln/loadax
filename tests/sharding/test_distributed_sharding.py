import jax
from jax.sharding import Mesh, PartitionSpec

from loadax.dataloader.sharding import DistributedShardingStrategy


def test_distributed_sharding_uneven_shards():
    dataset_size = 103
    mesh = Mesh(jax.devices(), ("dp",))
    sharding_strategy = DistributedShardingStrategy(mesh, PartitionSpec("dp"))
    num_shards = 4

    all_indices = []

    for shard_id in range(num_shards):
        shard_indices = sharding_strategy.get_shard_indices(
            dataset_size, shard_id, num_shards
        )

        for index in shard_indices:
            assert index not in all_indices
            all_indices.append(index)
