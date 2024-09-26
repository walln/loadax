import os

import jax.numpy as jnp
import numpy as np

from loadax.experimental.sharding import host_to_global_device_array

# simulate 8 xla devices via env var
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

from loadax.experimental.sharding.presets import make_fsdp_mesh_config

jax.config.update("jax_platform_name", "cpu")


if __name__ == "__main__":
    mesh_config = make_fsdp_mesh_config(
        mesh_axis_names=("data", "model"),
        batch_axis_names="data",
    )

    print(mesh_config.mesh_shape)
    print(mesh_config.mesh_axis_names)
    print(mesh_config.batch_axis_names)
    print(mesh_config.mesh_rules)

    mesh = mesh_config.create_device_mesh()
    print("Realized mesh:")
    print(mesh.devices)
    print(mesh.shape)

    # Given a local batch of data that is a range within the global batch, create the
    # logical sharding for the data.
    input_batch = np.arange(16)

    with mesh:
        print(input_batch)
        global_input_batch = host_to_global_device_array(input_batch)
        jax.debug.visualize_array_sharding(jnp.array(input_batch))
        jax.debug.visualize_array_sharding(global_input_batch)
