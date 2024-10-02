import os

import jax
import jax.numpy as jnp

from loadax.experimental.dataset.simple import SimpleDataset
from loadax.experimental.loader import Dataloader
from loadax.experimental.sharding.placement import host_to_global_device_array
from loadax.experimental.sharding.presets import make_fsdp_mesh_config

# simulate 8 xla devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


if __name__ == "__main__":
    dataset = SimpleDataset(list(range(128)))
    dataset = dataset.shuffle(jax.random.PRNGKey(0))

    mesh_config = make_fsdp_mesh_config(
        mesh_axis_names=("data", "model"), batch_axis_names="data"
    )
    mesh = mesh_config.create_device_mesh()

    dataloader = Dataloader(dataset, batch_size=16)

    with mesh:
        input_batch = next(iter(dataloader))
        input_batch = jnp.array(input_batch)
        global_input_batch = host_to_global_device_array(input_batch)
        jax.debug.visualize_array_sharding(jnp.array(input_batch))
        jax.debug.visualize_array_sharding(global_input_batch)

        # TODO: port over fsdp model setup
