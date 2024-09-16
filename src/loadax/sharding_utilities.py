"""Common utilities for defining sharding configurations."""

from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh

ShardingPreset = tuple[Mesh, tuple[str, ...]]


def fsdp_sharding() -> ShardingPreset:
    """Default sharding configuration for FSDP.

    FSDP is a distributed training technique where both the model's parameters and the
    data are sharded across multipled devices and potentially multiple nodes. This
    configuration ensures that each node manages a subset of the data and holds a
    partition of the model's parameters. By sharding the model, FSDP enables training
    of models that are too large to fit on a single device, optimize memory usage and
    computational efficiency.

    Loadax's default FSDP sharding configuration leverages JAX's parallelization
    primitives to minimize network traffic and optimize data loading. The model's
    parameters are distributed across devices within each data shard, allowing for
    scalable training across different architectures and network topologies. The data
    is partitioned such that each node can precompute what data it needs to load,
    reducing network traffic and improving data loading efficiency.

    In the case of single-node training, this configuration is still valid and only
    utilizes a single data shard. The data is still able to be used across all local
    devices, this is just to mean that the sharding splits are not pre-computed.

    This configuration is useful for FSDP training, but you can also create your own
    sharding configurations. See the documentation for more information.

    Returns:
        mesh (Mesh): The mesh to use for sharding.
        axis_names (tuple[str]): The axis names to use for sharding. The first axis
            name is used for the data sharding, and the second axis name is used for
            the model sharding.
    """
    num_total_devices = len(jax.devices())
    num_data_shards = jax.process_count()
    num_model_shards = num_total_devices // num_data_shards

    assert (
        num_total_devices % num_data_shards == 0
    ), "Number of devices must be divisible by number of data shards"

    devices: np.ndarray[Any, Any] = np.array(jax.devices()).reshape(
        (num_data_shards, num_model_shards)
    )
    axis_names = ("data", "model")
    mesh = Mesh(devices, axis_names)

    return mesh, axis_names


def ddp_sharding() -> ShardingPreset:
    """Default sharding configuration for DDP.

    DDP is a distributed training technique where the model's parameters are fully
    replicated across all devices, and the data is divided into shards, each processed
    by a separate node. This setup allows each device to handle unique subsets of the
    data while maintaining synchronized copies of the entire model. DDP is effective
    for scenarios where model size fits within the memory constraints of individual
    devices but benefits from data parallelism to accelerate training.

    Loadax's default DDP sharding configuration ensures that data is efficiently
    partitioned across nodes, and each device within a node holds a complete copy of the
    model's parameters. This configuration leverages JAX's parallelization capabilities
    to synchronize parameter updates across all replicas, ensuring consistency and
    facilitating scalable training.

    In the case of single-node training, this configuration is equivalent to DP training
    and does works perfectly fine for single-node training.

    This configuration is useful for DDP training, but you can also create your own
    sharding configurations. See the documentation for more information.
    """
    num_total_devices = len(jax.devices())
    num_data_shards = jax.process_count()

    assert (
        num_total_devices % num_data_shards == 0
    ), "Number of devices must be divisible by number of data shards"

    devices: np.ndarray[Any, Any] = np.array(jax.devices()).reshape((num_data_shards,))
    axis_names = ("data",)
    mesh = Mesh(devices, axis_names)

    return mesh, axis_names
