"""This is a simple example of how to do distributed FSDP training loop with Loadax.

This example is not meant to be a complete training loop, but rather a demonstration of
how you can leverage Jax's powerful parallelization primitives in combination with
Loadax to achieve distributed training.

Loadax does not lock you into any particular sharding strategy, but instead allows you
to define your own sharding strategy and optimize for your architecture, network
topology, device placement, etc. In fact there is no reason that you cannot create
new training paradigms ontop of Loadax or fully customize your training topology.

This example is a simple demonstration of the FSDP training strategy, which shards the
model across the devices and supplies each device with a local shard of the global
data. In a distributed setting, this means that both the data and model are sharded
across nodes/devices, such that the model may not be purely replicated across the
devices.

If you want to learn more about FSDP, the UvA DL Notebooks have a great tutorial
here: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
I would recommend the entire tutorials as a great resource for understanding some of the
concepts behind distributed training.

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
from loadax.sharding.presets import make_fsdp_mesh_config

# simulate 8 xla devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


# MeshRules is an example of you how you can define your own sharding rules.
# In this case we are doing logical partitioning based on data, and the layer type
# you could just use the pysical axis names if you wanted to and just remove this
# abstraction.
@dataclass(unsafe_hash=True)
class MeshRules:
    """MeshRules are logic sharding rules for the physical mesh."""

    mlp: str | None = None
    data: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        """Compute the sharding rules for the given keys."""
        return tuple(getattr(self, key) for key in keys)


mesh_rules = MeshRules(
    mlp="model",
    data="data",
)


class Model(nnx.Module):
    """Simple MLP model for demonstration purposes."""

    # Creating a model for FSDP is really simple, using NNX variable metadata
    # you just need to annotate parameters with the sharding rules. Here we use
    # a simple function to go logical partitioning.

    def __init__(self, d_in: int, d_model: int, d_out: int, rngs: nnx.Rngs):
        """Initialize the model."""
        self.w1 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (d_in, d_out)),
            sharding=mesh_rules("mlp"),
        )
        self.b1 = nnx.Param(jnp.zeros((d_model,)), sharding=mesh_rules("mlp"))
        self.w2 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (d_model, d_out)),
            sharding=mesh_rules("mlp"),
        )

    def __call__(self, x):
        """Forward pass of the model."""
        return nnx.relu(x @ self.w1 + self.b1) @ self.w2


@nnx.jit
def create_model():
    """Create the model and optimizer with sharded parameters and opt state."""
    # Notice how all we need to change for FSDP is to shard the parameters
    # and optimizer state. Its a single line of code!
    model = Model(1, 32, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1))
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    state = nnx.state(optimizer)
    sharded_state = jax.lax.with_sharding_constraint(
        state, nnx.get_named_sharding(state, mesh)
    )
    nnx.update(optimizer, sharded_state)

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


if __name__ == "__main__":
    dataset = SimpleDataset([jnp.array([i]) for i in range(128)])
    dataset = dataset.shuffle(jax.random.PRNGKey(0))

    mesh_config = make_fsdp_mesh_config(
        mesh_axis_names=("data", "model"), batch_axis_names="data"
    )
    mesh = mesh_config.create_device_mesh()

    dataloader = Dataloader(dataset, batch_size=16)

    model, optimizer, metrics = create_model()

    with mesh:
        for local_batch in dataloader:
            # You still have total control over array placement, so you can decide how
            # to parallelize your intra-node batch across the local devices, when to
            # synchronize across nodes, etc.
            # This is a really powerful set of primitives and you do not need to use
            # pjit, this is just an example of how you can leverage Jax's powerful
            # parallelization primitives in combination with Loadax to achieve
            # distributed training.

            # First stack the batch of arrays into a single array.
            local_batch = jnp.stack(local_batch)

            # Then convert the local batch to a global device array
            # if you are training on multiple nodes, this lets all nodes
            # synchronize the batch across all devices without actually
            # moving the data between the nodes. This can dramatically
            # reduce network traffic and the disk space needed for large
            # datasets.
            global_batch = host_to_global_device_array(local_batch)

            # Use jax.lax.with_sharding_constraint to specify the sharding of the input
            # this is for GSPMD partitioning, wiich allows you to not have to configure
            # your model for a specific batch size.
            sharded_batch = jax.lax.with_sharding_constraint(
                global_batch, jax.sharding.PartitionSpec(mesh_rules.data)
            )

            # Perform the training step just as you would in a normal training loop
            train_step(model, optimizer, metrics, sharded_batch)

            # Compute the metrics and print them
            for metric, value in metrics.compute().items():
                print(f"train_{metric}: {value}")
