import dataclasses
from collections.abc import Sequence

import jax

from loadax.experimental.typing_utils import REQUIRED, Required

MeshShape = Sequence[int]


# Adapted from https://github.com/apple/axlearn/blob/ef391f7f9640f77f693cae77558bb846a6464ad3/axlearn/common/utils.py#L64
@dataclasses.dataclass
class HybridMeshShape:
    """A mesh shape for hybrid (i.e., ICI and DCN) parallelism.

    For example, with mesh axes (data, model):
    - Pure fsdp on a v4-8:
        HybridMeshShape(ici_mesh_shape=(1, 4), dcn_mesh_shape=(1, 1))
    - Two-way data parallelism over 2 H100 nodes, and fsdp within-node:
        HybridMeshShape(ici_mesh_shape=(1, 8), dcn_mesh_shape=(2, 1))
    """

    ici_mesh_shape: MeshShape
    """The mesh shape for the ICI (inner-core) parallelism. This represents
    the host-local sharding."""
    dcn_mesh_shape: MeshShape
    """The mesh shape for the DCN (data-parallel) parallelism. This represents
    the inter-host sharding."""

    def __post_init__(self) -> None:
        if len(self.ici_mesh_shape) != len(self.dcn_mesh_shape):
            raise ValueError(
                f"""The mesh shapes must have the same number of axes, but
                found {len(self.ici_mesh_shape)} and {len(self.dcn_mesh_shape)}."""
            )

    def __len__(self):
        assert len(self.ici_mesh_shape) == len(self.dcn_mesh_shape)
        return len(self.ici_mesh_shape)


# Adapted from https://github.com/apple/axlearn/blob/ef391f7f9640f77f693cae77558bb846a6464ad3/axlearn/common/trainer.py#L45
@dataclasses.dataclass
class MeshConfig:
    """Sharding Mesh Configuration."""

    mesh_shape: Required[MeshShape | HybridMeshShape] = REQUIRED
    """ If specified as a MeshShape, must have the same length as mesh_axis_names.
    Implicitly, this treats the mesh shape as the ICI mesh shape; we default to a DCN
    mesh shape that partitions the first non-singleton axis across granules (e.g. TPU
    slices or GPU nodes). If all axes are singletons, this implies a single-granule
    environment and therefore an all-1's DCN mesh shape.

    As an example on 2 H100 nodes, for mesh axes (pipeline, data, model) and a MeshShape
    of (1, 2, 8), we break the "data" axis across DCN -- this produces a DCN mesh shape
    (1, 2, 1) and an ICI mesh shape (1, 1, 8), i.e. 2-way data-parallelism across DCN,
    and 8-way model parallelism within-node (e.g. NVLink). If instead the MeshShape is
    provided as (2, 1, 8), we break along the "pipeline" axis, producing a DCN mesh
    shape of (2, 1, 1) and ICI mesh shape (1, 1, 8) for 2-way pipeline-parallelism
    across DCN and 8-way model parallelism within-node.

    If specified as a HybridMeshShape, each member must have the same length as
    mesh_axis_names.

    Use `mesh_rules` to set different mesh shapes depending on the hardware platform.
    """

    mesh_axis_names: Required[Sequence[str]] = REQUIRED
    """The mesh axis names. The names can be referenced in ParameterSpec.mesh_axes."""
    batch_axis_names: str | Sequence[str] = "data"
    """Subset of mesh axis names over which leaves of the input batch are sharded."""
    mesh_rules: Sequence[tuple[str, MeshShape | None]] | None = None
    """An optional list of (regex, MeshShape) pairs to override the default mesh
    configuration.

    This is useful when we want to use different mesh shapes depending on the
    device types (e.g., 'tpu-v4-128' vs. 'gpu-p4de.24xlarge-32').

    Given a `mesh_selector` string (usually representing the device type and set by
    user's launch script), the first rule that with a regex that matches the selector
    will determine the mesh shape.

    If no rule matches, the default mesh configuration will be used.
    """

    def create_device_mesh(
        self, devices: list[jax.Device] | None = None
    ) -> jax.sharding.Mesh:
        """Creates a Mesh object for the given devices.

        Args:
            devices (list[jax.Device] | None, optional): A list of devices to create a
                Mesh object for. If None, the default devices will be used. Defaults to
                None.

        Returns:
            jax.sharding.Mesh: A Mesh object for the given devices.
        """
        from loadax.experimental.sharding.mesh_utils import create_device_mesh

        return jax.sharding.Mesh(
            devices or create_device_mesh(mesh_shape=self.mesh_shape),
            self.mesh_axis_names,
        )
