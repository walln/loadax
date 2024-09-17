"""This is a simple example of how you can compose a distributed FSDP training loop with Loadax.

This example is not meant to be a complete training loop, but rather a demonstration of how you can leverage
Jax's powerful parallelization primitives in combination with Loadax to achieve distributed training.

Loadax does not lock you into any particular sharding strategy, but instead allows you to define your own
sharding strategy and optimize for your architecture, network topology, device placement, etc. In fact there is 
no reason that you cannot create new training paradigms ontop of Loadax or fully customize your training topology.

This example is a simple demonstration of the FSDP training strategy, which shards the model across the devices
and supplies each device with a local shard of the global data. In a distributed setting, this means that both the
data and model are sharded across nodes/devices, such that the model may not be purely replicated across the devices.

If you want to learn more about FSDP, the UvA DL Notebooks have a great tutorial 
here: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html 
I would recommend the entire tutorials as a great resource for understanding some of the concepts 
behind distributed training.

"""

from dataclasses import dataclass
import jax
import jax.experimental
import jax.experimental.mesh_utils
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
from loadax.dataloader.loader import Dataloader
from loadax.dataloader.sharding import DistributedShardingStrategy
from loadax import InMemoryDataset, Batcher
from loadax.strategy import FixedBatchStrategy
import os
import flax.nnx as nnx
import optax
from loadax.sharding_utilities import fsdp_sharding

# Use xla flags to simulate a distributed environment
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

# If using multiple hosts, you will need to initialize the jax distributed runtime
# jax.distributed.initialize()

# You can create your own sharding configurations or use the presets provided by loadax
# See the documentation for more information
mesh, axis_names = fsdp_sharding()

dataset = InMemoryDataset(items=[jnp.array([i]) for i in range(128)])
batcher = Batcher(lambda items: jnp.stack(items))
batch_strategy = FixedBatchStrategy(batch_size=16)

# The data shard axis is used to determine how to construct the global batch from the local batches
# this is optional, but the most convenient way to do FSDP training. To understand why, see the
# documentation for the DistributedShardingStrategy
sharding_strategy = DistributedShardingStrategy(mesh, data_shard_axis='data')

@dataclass(unsafe_hash=True)
class MeshRules:
    mlp: str | None = None
    data: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        return tuple(getattr(self, key) for key in keys)

mesh_rules = MeshRules(
    mlp='model',
    data='data',
)

class Model(nnx.Module):
    def __init__(self, d_in: int, d_model: int, d_out: int, rngs: nnx.Rngs):
        self.w1 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (d_in, d_out)),
            sharding=mesh_rules('mlp')
        )
        self.b1 = nnx.Param(
            jnp.zeros((d_model,)),
            sharding=mesh_rules('mlp')
        )
        self.w2 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (d_model, d_out)),
            sharding=mesh_rules('mlp')
        )

    def __call__(self, x):
        return nnx.relu(x @ self.w1 + self.b1) @ self.w2
    
@nnx.jit    
def create_model():
    model = Model(1, 32, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss')
    )

    state = nnx.state(optimizer)
    sharded_state = jax.lax.with_sharding_constraint(state, nnx.get_named_sharding(state, mesh))
    nnx.update(optimizer, sharded_state)

    return model, optimizer, metrics

@nnx.jit
def train_step(model: Model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: jnp.ndarray):
    def loss_fn(model):
        y_pred = model(batch)
        return jnp.mean((y_pred - batch) ** 2)
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grad)
    metrics.update(loss=loss)

def test_distributed_dataloader_with_parameter_sharding():
    dataloader = Dataloader(
        dataset=dataset,
        batcher=batcher,
        strategy=batch_strategy,
        num_workers=2,
        prefetch_factor=2,
        sharding_strategy=sharding_strategy,
    )

    model, optimizer, metrics = create_model()

    for local_batch in dataloader:
        # You still have total control over array placement, so you can decide how to parallelize your
        # intra-node batch across the local devices, when to synchronize across nodes, etc. This is a really
        # powerful set of primitives and you do not need to use pjit, this is just an example of how you can
        # leverage Jax's powerful parallelization primitives in combination with Loadax to achieve distributed
        # training.         
        global_batch = sharding_strategy.distribute_global_batch(local_batch)
        train_step(model, optimizer, metrics, global_batch)

        for metric, value in metrics.compute().items():
            print(f"train_{metric}: {value}")

test_distributed_dataloader_with_parameter_sharding()
