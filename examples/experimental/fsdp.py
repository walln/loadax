import jax
import jax.numpy as jnp
import numpy as np

from loadax.experimental.dataset.simple import SimpleDataset
from loadax.experimental.sharding.placement import host_to_global_device_array
from loadax.experimental.sharding.presets import make_fsdp_mesh_config

if __name__ == "__main__":
    dataset = SimpleDataset(list(range(128)))

    mesh_config = make_fsdp_mesh_config(
        mesh_axis_names=("data", "model"), batch_axis_names="data"
    )
    mesh = mesh_config.create_device_mesh()

    # TODO: use dataloader and batcher

    with mesh:
        input_batch = np.arange(16)
        global_input_batch = host_to_global_device_array(input_batch)
        jax.debug.visualize_array_sharding(jnp.array(input_batch))
        jax.debug.visualize_array_sharding(global_input_batch)

        # print(mesh.mesh_axes)
