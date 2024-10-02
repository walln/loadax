from typing import Any

import jax
import numpy as np
from jax._src.mesh import thread_resources
from jax.sharding import PartitionSpec

from loadax.sharding.partition_spec import (
    DataPartitionType,
    data_partition_type_to_spec,
)
from loadax.sharding.tree_utils import (
    Nested,
    complete_partition_spec_tree,
    flatten_items,
    shapes,
    tree_paths,
)


def host_to_global_device_array(
    host_arrays: Nested[jax.Array],
    *,
    partition: DataPartitionType = DataPartitionType.FULL,
) -> Nested[jax.Array]:
    """Converts the given host device arrays to global device arrays.

    Must be called within the context of a Mesh.

    We cannot use `multihost_utils.host_local_array_to_global_array` since the local
    mesh may not be contiguous. According to yashkatariya@google.com,
    "using `jax.make_array_from_single_device_arrays` is the right solution."

    Args:
        host_arrays: a nested tree of device arrays in host memory. Usually these
            present the per-host portion of the global input batch.
        partition: how the global array should be partitioned.

    Returns:
        A nested tree with the same structure as `host_arrays`, but global device
        arrays at the leaves. Each global device array is partitioned
        according to `partition`.

    Raises:
        NotImplementedError: if the given `partition` type is not supported.
    """
    mesh = thread_resources.env.physical_mesh
    partition_spec = data_partition_type_to_spec(partition)

    local_devices = mesh.local_devices

    def put_to_devices_fully_partitioned(x: jax.Array) -> list[jax.Array]:
        len_local_devices = len(local_devices)
        if x.shape[0] % len_local_devices != 0:
            raise ValueError(
                f"({x.shape}) cannot be sharded across {len_local_devices} devices."
            )
        # np.reshape is faster than np.split, jnp.reshape, and jnp.split.
        xs = np.reshape(
            x, (len_local_devices, x.shape[0] // len_local_devices, *x.shape[1:])
        )
        return [
            jax.device_put(x_i, device)
            for x_i, device in zip(xs, local_devices, strict=False)
        ]

    def put_to_devices_replicated(x: jax.Array) -> list[jax.Array]:
        # Replicate `x` to every local device.
        return [jax.device_put(x, device) for device in local_devices]

    if partition == DataPartitionType.FULL:
        put_to_devices = put_to_devices_fully_partitioned
    elif partition == DataPartitionType.REPLICATED:
        put_to_devices = put_to_devices_replicated
    else:
        raise NotImplementedError(f"Unsupported partition: {partition}")

    device_arrays = jax.tree_util.tree_map(put_to_devices, host_arrays)
    partition_specs = complete_partition_spec_tree(
        jax.tree_util.tree_structure(host_arrays),
        partition_spec,
    )

    def make_gda(
        x: jax.Array, device_buffers: list[jax.Array], partition_spec: PartitionSpec
    ) -> jax.Array:
        if partition == DataPartitionType.FULL:
            global_batch_size = x.shape[0] * jax.process_count()
        elif partition == DataPartitionType.REPLICATED:
            global_batch_size = x.shape[0]
        else:
            raise NotImplementedError(f"Unsupported partition: {partition}")
        global_shape = (global_batch_size, *list(x.shape[1:]))
        return jax.make_array_from_single_device_arrays(
            shape=global_shape,
            sharding=jax.sharding.NamedSharding(mesh, partition_spec),
            arrays=device_buffers,
        )

    return jax.tree_util.tree_map(make_gda, host_arrays, device_arrays, partition_specs)  # type: ignore


def global_to_host_array(
    global_arrays: Nested[jax.Array],
    *,
    partition: DataPartitionType = DataPartitionType.FULL,
) -> Nested[jax.Array]:
    """Extracts host addressable rows from each Array in `global_arrays`.

    Args:
        global_arrays: A Nested[jax.Array].
            Each leaf Array must have shape [global_batch_size, ...] with identical
            global_batch_size across arrays.
            The arrays must be partitioned in the same way and can be partitioned
            only along the batch axis.
        partition: How the global array should be partitioned.

    Returns:
        A Nested[jax.Array] with the same structure as `global_array`. Each leaf
        Array will have shape [host_batch_size, ...] where `host_batch_size` will be
        equal to `global_batch_size` if the global Arrays are replicated or
        `global_batch_size // process_count` if the global Arrays are partitioned
        across hosts.
    """

    def sort_global_shards(global_shards: list[jax.Shard]) -> list[jax.Shard]:
        # We should sort jax.Array.global_shards by using this function to guarantee
        # round-trip equality of host_to_global_device_array and global_to_host_array.
        # Shards are sorted in-place.
        global_shards.sort(key=lambda shard: shard.index)
        return global_shards

    global_array_items = flatten_items(global_arrays)
    if not global_array_items:
        return global_arrays  # no leaf jax.Array.
    first_path, first_value = global_array_items[0]
    sorted_first_value_shards = sort_global_shards(list(first_value.global_shards))
    first_value_shard_is_local = [
        shard.data is not None for shard in sorted_first_value_shards
    ]
    batch_size = first_value.shape[0]

    def get_local_array(path: str, value: jax.Array) -> jax.Array:
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Value batch size mismatch: {batch_size} @ {first_path} vs. "
                f"{value.shape[0]} @ {path} of {shapes(global_arrays)}"
            )
        sorted_value_shards = sort_global_shards(list(value.global_shards))
        value_shard_is_local = [shard.data is not None for shard in sorted_value_shards]
        if value_shard_is_local != first_value_shard_is_local:
            raise ValueError(
                f"Value shard mismatch: {first_value_shard_is_local} @ {first_path} "
                f"vs. {value_shard_is_local} @ {path}"
            )
        local_data = [
            shard.data for shard in sorted_value_shards if shard.data is not None
        ]
        if not local_data:
            raise ValueError(f"No local shard found: {sorted_value_shards}.")
        if partition == DataPartitionType.FULL:
            # return ndarray its faster than jnp.concatenate
            return np.concatenate(local_data, axis=0)  # type: ignore
        elif partition == DataPartitionType.REPLICATED:
            return local_data[0]  # type: ignore
        else:
            raise NotImplementedError(f"Unsupported partition: {partition}")

    # TODO: jtu types are bad
    return jax.tree_util.tree_map(  # type: ignore
        get_local_array, tree_paths(global_arrays), global_arrays
    )


def with_sharding_constraint(x: jax.Array, shardings: Any) -> jax.Array:
    """Syntax sugar for `jax.lax.with_sharding_constraint`.

    Used from within the context of a Mesh, this will produce a no-op if the Mesh
    is empty or has only one device.
    """
    mesh = thread_resources.env.physical_mesh
    if mesh.empty or mesh.size == 1:
        return x
    # TODO: jax types are bad
    return jax.lax.with_sharding_constraint(x, shardings)  # type: ignore
