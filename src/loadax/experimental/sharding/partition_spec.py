from enum import Enum

from jax._src.mesh import thread_resources
from jax.sharding import PartitionSpec


class DataPartitionType(Enum):
    """Partition strategy for a given batch."""

    FULL = "full"
    """Data is fully partitioned across all devices."""
    REPLICATED = "replicated"
    """Data is fully replicated across all devices."""


def input_partition_spec() -> PartitionSpec:
    """Returns partition spec for the input batch.

    We partition the inputs along all axes. For example, if the mesh has shape (64, 4)
    and axis  names of ("data", "model"), the partition spec will be
    (("data", "model"), None...) so that the batch axis of every global tensor will be
    partitioned 256 (= 64 * 4) ways.

    Must be called within the context of a Mesh.
    """
    mesh = thread_resources.env.physical_mesh
    return PartitionSpec(
        mesh.axis_names,
    )


def data_partition_type_to_spec(partition: DataPartitionType) -> PartitionSpec:
    """Returns a PartitionSpec for the given partition type."""
    if partition == DataPartitionType.FULL:
        return input_partition_spec()
    elif partition == DataPartitionType.REPLICATED:
        return None
    else:
        raise NotImplementedError(f"Unsupported partition: {partition}")
