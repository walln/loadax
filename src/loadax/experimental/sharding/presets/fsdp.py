import re
from collections.abc import Sequence

import jax

from loadax.experimental.sharding.mesh_shape import (
    HybridMeshShape,
    MeshConfig,
    MeshShape,
)


def make_fsdp_mesh_config(
    mesh_axis_names: Sequence[str],
    batch_axis_names: str | Sequence[str] = "data",
    mesh_rules: list[tuple[str, MeshShape | HybridMeshShape | None]] | None = None,
    mesh_selector: str | None = None,
) -> MeshConfig:
    """Creates a MeshConfig configured for Fully Sharded Data Parallel (FSDP) training.

    - Detects whether the execution is on a single node or multiple nodes.
    - Determines the number of devices per node.
    - Configures mesh shapes accordingly.
    - Applies mesh rules based on the mesh_selector.

    Args:
        mesh_axis_names (Sequence[str]): The names of the mesh axes.
        batch_axis_names (str | Sequence[str], optional): Subset of mesh axis names
            over which leaves of the input batch are sharded. Defaults to "data".
        mesh_rules (List[Tuple[str, MeshShape | HybridMeshShape]] | None, optional):
            Optional list of (regex, MeshShape) pairs to override the default mesh
            configuration based on the mesh_selector. Defaults to None.
        mesh_selector (str, optional): A string representing the hardware type or
            configuration, used to select the appropriate mesh rule. If None, no rules
            are applied.

    Returns:
        MeshConfig: The configured mesh configuration for FSDP.
    """
    # Initialize default mesh_shape as None
    default_mesh_shape: MeshShape | HybridMeshShape | None = None

    # Apply mesh rules if provided
    if mesh_rules and mesh_selector:
        for pattern, shape in mesh_rules:
            if re.match(pattern, mesh_selector):
                if shape is None:
                    raise ValueError(
                        f"Mesh shape for pattern '{pattern}' cannot be None."
                    )
                default_mesh_shape = shape
                print(f"Mesh rule matched: pattern='{pattern}', applying shape={shape}")
                break

    # If no mesh_rule matched or no rules provided, infer mesh_shape
    if default_mesh_shape is None:
        # Total number of nodes participating in the computation
        num_nodes = jax.process_count()

        # Number of devices (e.g., GPUs) available on the current node
        # we assume all nodes are homogeneous (jax assumes this as well)
        devices_per_node = len(jax.local_devices())

        if num_nodes < 1:
            raise ValueError(f"Invalid number of nodes: {num_nodes}. Must be >= 1.")

        if devices_per_node < 1:
            raise ValueError(f"""Invalid number of devices per node: {devices_per_node}.
            Must be >= 1.""")

        # Configure DCN mesh shape
        if num_nodes == 1:
            # Single-node setup
            dcn_mesh_shape = tuple([1] * len(mesh_axis_names))
        else:
            # Multi-node setup: assuming data or pipeline parallelism across nodes
            # Identify the first axis to partition (first non-singleton)
            # Here, we assume that the first axis is the one to be partitioned
            # Modify this logic based on your specific parallelism strategy
            dcn_mesh_shape = list([1] * len(mesh_axis_names))
            # For simplicity, let's partition along the first axis
            dcn_mesh_shape[0] = num_nodes

        # Configure ICI (Intra-Component Interconnect) mesh shape
        ici_mesh_shape = list([1] * len(mesh_axis_names))
        # Assume model parallelism is on the last axis
        ici_mesh_shape[-1] = devices_per_node

        hybrid_mesh_shape = HybridMeshShape(
            ici_mesh_shape=tuple(ici_mesh_shape), dcn_mesh_shape=tuple(dcn_mesh_shape)
        )

        default_mesh_shape = hybrid_mesh_shape

    # Instantiate MeshConfig
    mesh_config = MeshConfig(
        mesh_shape=default_mesh_shape,
        mesh_axis_names=mesh_axis_names,
        batch_axis_names=batch_axis_names,
        mesh_rules=mesh_rules,
    )

    return mesh_config
