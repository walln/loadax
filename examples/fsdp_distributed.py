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
from typing import Callable
import jax
import jax.experimental
import jax.experimental.mesh_utils
import jax.experimental.multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from loadax.dataloader.distributed import DistributedDataLoader, JaxShardingStrategy
from loadax import InMemoryDataset, Batcher
from loadax.strategy import FixedBatchStrategy
import os
import flax.nnx as nnx
import optax

# Use xla flags to simulate a distributed environment
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

num_total_devices = len(jax.devices())
num_data_shards = jax.process_count()
num_model_shards = num_total_devices // num_data_shards

assert num_total_devices % num_data_shards == 0, "Number of devices must be divisible by number of data shards"

devices = np.array(jax.devices()).reshape((num_data_shards, num_model_shards))
mesh = Mesh(devices, ('data', 'model'))

dataset = InMemoryDataset(items=[jnp.array([i]) for i in range(128)])
batcher = Batcher(lambda items: jnp.stack(items))
batch_strategy = FixedBatchStrategy(batch_size=16)

# This is the shard_id, which is used to determine which jax process this loader instance is running on
sharding_strategy = JaxShardingStrategy(mesh, data_shard_axis='data')

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
    print(f"Devices: {devices}")
    print(f"Mesh: {mesh}")
    print(f"Sharding strategy: {sharding_strategy}")

    # Create the DistributedDataLoader
    dataloader = DistributedDataLoader(
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
