# Mesh

Loadax has a specialized mesh definition called a `HybridMeshShape`. This enables you to define your mesh with two contexts, the global topology (inter-node) and the local topology (intra-node). This makes it easier to reason about the placement of data across the network.

To use loadax's mesh abstractions, you need to define a `MeshConfig`. This config tells loadax how to split up the work across the mesh. A `MeshConfig` must specify a `mesh_shape`, which is a `HybridMeshShape` and some annotations about the mesh axes. 

Typically you should not need to define a `MeshConfig` yourself. Instead you can rely on Loadax's automatic mesh discovery to find a good mesh shape for your cluster for a given parallelization strategy. See the [Presets](./presets.md) section for more details.

::: loadax.sharding.mesh_shape.HybridMeshShape

::: loadax.sharding.mesh_shape.MeshConfig

