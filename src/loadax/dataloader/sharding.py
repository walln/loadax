"""Sharding utilities for dataloaders."""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec


class ShardingStrategy(ABC):
    """Abstract base class for sharding strategies."""

    mesh: Mesh | None = None
    data_axis: str | None = None

    @abstractmethod
    def get_shard_indices(
        self, dataset_size: int, shard_id: int, num_shards: int
    ) -> range:
        """Get the data indices for a specific shard.

        Args:
            dataset_size: The size of the dataset.
            shard_id: The ID of the shard.
            num_shards: The total number of shards.

        Returns:
            range: The data indices for the shard.
        """
        pass

    @abstractmethod
    def needs_sharding(self) -> bool:
        """Check if the dataloader needs to be sharded."""
        pass

    @abstractmethod
    def distribute_global_batch(
        self, local_batch: jnp.ndarray, *, data_axis: str | None = None
    ) -> jnp.ndarray:
        """Distribute a local batch across the entire cluster.

        Take a local batch and stitch it together into a global batch with all processes
        contributing their local batch to the global batch. This is useful for
        distributed training where the data is not replicated across the entire cluster.

        Args:
            local_batch: The local batch to distribute.
            data_axis: The axis name for the data sharding.
                If None, then the data axis must have been provided to the
                sharding_strategy. Defaults to None.

        Returns:
            np.ndarray: The global batch.
        """
        pass

    def named_sharding(self, *names: str | None) -> jax.NamedSharding:
        """Get a NamedSharding object for the given axis names.

        This is just a useful convenience method for creating NamedSharding objects,
        just provide the axis names as they exist in the mesh to create the
        NamedSharding object.

        Args:
            *names (str | None): The axis names to use for the NamedSharding object.

        Returns:
            jax.NamedSharding: The NamedSharding object for the given axis names.
        """
        return jax.NamedSharding(self.mesh, PartitionSpec(*names))  # type: ignore # jax PartitionSpec has missing types in current version


class NoShardingStrategy(ShardingStrategy):
    """A sharding strategy that does not shard the data.

    This is the default sharding strategy used on single node training. This also
    is useful if you are doing ensemble training where each node should get the exact
    same batch of data.
    """

    def get_shard_indices(
        self, dataset_size: int, shard_id: int, num_shards: int
    ) -> range:
        """Get the data indices for a specific shard."""
        return range(dataset_size)

    def needs_sharding(self) -> bool:
        """Check if the dataloader needs to be sharded."""
        return False

    def distribute_global_batch(
        self, local_batch: jnp.ndarray, *, data_axis: str | None = None
    ) -> jnp.ndarray:
        """Distribute a local batch across the entire cluster.

        Take a local batch and stitch it together into a global batch with all processes
        contributing their local batch to the global batch. This is useful for
        distributed training where the data is not replicated across the entire cluster.

        Args:
            local_batch: The local batch to distribute.
            data_axis: The axis name for the data sharding.
                If None, then the data axis must have been provided to the
                sharding_strategy. Defaults to None.

        Returns:
            np.ndarray: The global batch.
        """
        return local_batch


class DistributedShardingStrategy(ShardingStrategy):
    """Define a distributed sharding strategy for the dataloader.

    This sharding strategy is useful for distributed training where each node
    should only load a partition of the dataset. This is done by dividing the
    dataset into equally sized shards and assigning each worker node a
    subset of the shards.

    The most common way to use this sharding strategy is to have each jax process
    load a `local_batch` which is a partition of the batch at any given step,
    because in most distributed training setups, the data is not replicated across
    the entire cluster. This means that we can pre-compute which process needs which
    data and then we can just load the data for that process. Upon loading the
    `local_batch`, we can then use the `distribute_global_batch` method to distribute
    the `local_batch` across the entire cluster. This name is atually a bit misleading,
    because the `local_batch`is not actually distributed across the entire cluster,
    but rather is just stitched into a `global_batch` that is distributed across the
    entire cluster. This enables you to write your jax code in a way that is agnostic
    to the distribution of the data.
    """

    def __init__(self, mesh: Mesh, data_shard_axis: str | None = None):
        """Create a new distributed sharding strategy.

        Args:
            mesh (Mesh): The mesh to use for sharding.
            data_shard_axis (str | None): The axis name for the data sharding.
                This is used to determine how to compute the global batch from
                the node-local batches. If None, then the data sharding axis
                will be required to be supplied when using `distribute_global_batch`.
        """
        self.mesh = mesh
        self.data_axis = data_shard_axis

    def get_shard_indices(
        self, dataset_size: int, shard_id: int, num_shards: int
    ) -> range:
        """Get the data indices for a specific shard.

        Args:
            dataset_size: The size of the dataset.
            shard_id: The ID of the shard.
            num_shards: The total number of shards.

        Returns:
            range: The data indices for the shard.
        """
        if num_shards <= 1:
            return range(dataset_size)

        base_shard_size = dataset_size // num_shards
        remainder = dataset_size % num_shards

        # distribute the remainder evenly across the first 'remainder' shards
        if shard_id < remainder:
            shard_size = base_shard_size + 1
            start_index = shard_id * shard_size
        else:
            shard_size = base_shard_size
            start_index = (
                remainder * (base_shard_size + 1) + (shard_id - remainder) * shard_size
            )

        end_index = start_index + shard_size
        return range(start_index, end_index)

    def needs_sharding(self) -> bool:
        """Check if the dataloader needs to be sharded."""
        return True

    def distribute_global_batch(
        self, local_batch: jnp.ndarray, *, data_axis: str | None = None
    ) -> jnp.ndarray:
        """Distribute a local batch across the entire cluster.

        Take a local batch and stitch it together into a global batch with all processes
        contributing their local batch to the global batch. This is useful for
        distributed training where the data is not replicated across the entire cluster.

        Args:
            local_batch: The local batch to distribute.
            data_axis: The axis name for the data sharding.
                If None, then the data axis must have been provided to the
                sharding_strategy. Defaults to None.

        Returns:
            np.ndarray: The global batch.
        """
        # local_batch: the batch of data on the local process
        # data_axis: the axis name for the data sharding

        data_axis = data_axis or self.data_axis
        if not data_axis:
            raise ValueError(
                "data_axis must be provided when using distribute_global_batch"
            )

        if jax.process_count() == 1:
            return jnp.asarray(local_batch)

        data_sharding = self.named_sharding(data_axis)
        global_batch_size = local_batch.shape[0] * jax.process_count()
        global_batch_shape = (global_batch_size,) + local_batch.shape[1:]

        def data_callback(index: Any) -> jnp.ndarray:
            # index: the global index for the sharded array
            # We need to return the local data corresponding to this index
            # Since each process only has its local data, we check if the index
            # corresponds to our process

            # Calculate the start and end indices for this process
            per_process_batch_size = local_batch.shape[0]
            start = jax.process_index() * per_process_batch_size
            end = start + per_process_batch_size

            # Check if the requested index overlaps with our local data
            index_start = index[0].start or 0
            index_stop = index[0].stop or global_batch_size

            # If the requested index is within our local data range, return
            # the corresponding data
            if start <= index_start < end:
                local_index_start = index_start - start
                local_index_stop = min(index_stop - start, per_process_batch_size)
                return local_batch[local_index_start:local_index_stop]
            else:
                # Return an empty array if the index does not correspond to our data
                return jnp.zeros((0,) + local_batch.shape[1:], dtype=local_batch.dtype)

        global_batch = jax.make_array_from_callback(
            global_batch_shape, data_sharding, data_callback
        )
        return global_batch
