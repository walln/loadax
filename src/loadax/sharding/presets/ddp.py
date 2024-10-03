import re
from collections.abc import Sequence

import jax

from loadax.sharding.mesh_shape import (
    HybridMeshShape,
    MeshConfig,
    MeshShape,
)


def make_ddp_mesh_config(
    mesh_axis_names: Sequence[str] = ("data",),
    batch_axis_names: str | Sequence[str] = "data",
    mesh_rules: list[tuple[str, MeshShape | HybridMeshShape | None]] | None = None,
    mesh_selector: str | None = None,
) -> MeshConfig:
    """Creates a MeshConfig configured for Data Parallel (DP) training.

    - Detects whether the execution is on a single node or multiple nodes.
    - Determines the total number of devices across all nodes.
    - Configures mesh shapes for data parallelism.
    - Applies mesh rules based on the mesh_selector.

    Args:
        mesh_axis_names (Sequence[str], optional): The names of the mesh axes.
            Defaults to ("data",).
        batch_axis_names (str | Sequence[str], optional): Subset of mesh axis names
            over which leaves of the input batch are sharded. Defaults to "data".
        mesh_rules (List[Tuple[str, MeshShape | HybridMeshShape]] | None, optional):
            Optional list of (regex, MeshShape) pairs to override the default mesh
            configuration based on the mesh_selector. Defaults to None.
        mesh_selector (str, optional): A string representing the hardware type or
            configuration, used to select the appropriate mesh rule. If None, no rules
            are applied.

    Returns:
        MeshConfig: The configured mesh configuration for DP.
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

        # For DP, we use a single axis for data parallelism
        ici_mesh_shape = (devices_per_node,)
        dcn_mesh_shape = (num_nodes,)

        hybrid_mesh_shape = HybridMeshShape(
            ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape
        )

        default_mesh_shape = hybrid_mesh_shape

    # Instantiate MeshConfig
    mesh_config = MeshConfig(
        mesh_shape=default_mesh_shape,
        mesh_axis_names=mesh_axis_names,
        batch_axis_names=batch_axis_names,
        mesh_rules=mesh_rules,  # type: ignore
    )

    return mesh_config
