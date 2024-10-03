"""This is a simple example of how to do DDP training with Loadax.

This example demonstrates the DataParallel (DP) training strategy, which replicates
the model across all devices and supplies each device with a local shard of the
global data. In a distributed setting, this means that each node will have a copy
of the model and a sharded section of the data.

Loadax allows you to define your own sharding strategy and optimize for your
architecture, network topology, device placement, etc. This example shows how to
use Loadax's powerful primitives to achieve distributed training with the DP
strategy.
"""

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from loadax.dataloader.loader import Dataloader
from loadax.dataset.simple import SimpleDataset
from loadax.sharding.placement import host_to_global_device_array
from loadax.sharding.presets import make_ddp_mesh_config

# Simulate 8 XLA devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


@dataclass(unsafe_hash=True)
class MeshRules:
    """MeshRules are logical sharding rules for the physical mesh."""

    data: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        """Compute the sharding rules for the given keys."""
        return tuple(getattr(self, key) for key in keys)


mesh_rules = MeshRules(data="data")


class Model(nnx.Module):
    """Simple model for demonstration purposes."""

    def __init__(self, rngs: nnx.Rngs):
        """Initialize the model."""
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (1, 1)))

    def __call__(self, x):
        """Forward pass of the model."""
        return x * self.w


@nnx.jit
def create_model():
    """Create the model and optimizer."""
    model = Model(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1))
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    return model, optimizer, metrics


@nnx.jit
def train_step(
    model: Model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: jnp.ndarray
):
    """Perform a single training step."""

    def loss_fn(model):
        y_pred = model(batch)
        return jnp.mean((y_pred - batch) ** 2)

    loss, grad = nnx.value_and_grad(loss_fn)(model)

    optimizer.update(grad)
    metrics.update(loss=loss)

    return loss


if __name__ == "__main__":
    dataset = SimpleDataset([jnp.array([1.0]) for i in range(128)])
    dataset = dataset.shuffle(jax.random.PRNGKey(0))

    mesh_config = make_ddp_mesh_config(
        mesh_axis_names=("data",), batch_axis_names="data"
    )
    mesh = mesh_config.create_device_mesh()

    dataloader = Dataloader(dataset, batch_size=8)

    with mesh:
        model, optimizer, metrics = create_model()

        # Configure pmap for distributed training:
        # - axis_name="data" specifies the axis for parallelization
        # - in_axes=(None, None, None, 0) ensures the first 3 args
        #   (model, optimizer, metrics) are not sharded but replicated across devices,
        #   while the batch is sharded
        # - donate_argnums=(1, 2) allows in-place updates for optimizer and metrics
        pmap_train_step = nnx.pmap(
            train_step,
            axis_name="data",
            in_axes=(None, None, None, 0),
            out_axes=None,
            donate_argnums=(0, 1, 2),
        )

        for local_batch in dataloader:
            # Stack the batch of arrays into a single array
            local_batch = jnp.stack(local_batch)

            # Convert the local batch to a global device array
            global_batch = host_to_global_device_array(local_batch)

            # Use jax.lax.with_sharding_constraint to specify the sharding of the input
            sharded_batch = jax.lax.with_sharding_constraint(
                global_batch, jax.sharding.PartitionSpec(mesh_rules.data)
            )

            # Use pmap to replicate the computation across all devices
            loss = pmap_train_step(model, optimizer, metrics, sharded_batch)

            # Compute and print the metrics
            for metric, value in metrics.compute().items():
                print(f"train_{metric}: {value}")

    # Unreplicate the model parameters for final output
    final_params = jax.device_get(jax.tree.map(lambda x: x[0], model.w))
    print(f"Final params: {final_params}")
