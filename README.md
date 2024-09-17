# Loadax

Loadax is a dataloading library designed for the JAX ecosystem. It provides utilities for feeding data into your training loop without having to worry about batching, shuffling, and other preprocessing steps. Loadax also supports offloading data loading to the background, and prefetching a cache to improve performance, and jax-native distributed data loading.

[!Important] Loadax is currently in early development, and the rest of this README is a working draft.

## Installation

```bash
pip install loadax
```

## Usage

### Data Loading

Loadax provides a simple interface for loading data into your training loop. Here is an example of loading data from a list of items:

```python
from loadax import Dataloader, InMemoryDataset, Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
batcher = Batcher(lambda x: x)
loader = Dataloader(batcher).batch_size(2).build(dataset)

for batch in loader:
    print(batch)

# Output:
# [1, 2]
# [3, 4]
# [5, 6]
# [7, 8]
# [9, 10]
```

A dataloader is a definition of how to load data from a dataset. It itself is stateless enabling you to define mutliple dataloaders for the same dataset, and even multipple iterators for the same dataloader.

```python
dataloader = Dataloader(batcher).batch_size(2).build(dataset)

iter_a = iter(dataloader)
iter_b = iter(dataloader)

val = next(iter_a)
print(val)
# Output: 1

val = next(iter_b)
print(val)
# Output: 1
```

In the above examples we create an object called a batcher. A batcher is an interface that defines how to collate your data into batches. This is useful for when you want to alter the way your data is batched such as stacking into a single array.

### Data Prefetching

When training models, it is essential to ensure that you are not blocking the training loop and especially your accelerator(s), with IO bound tasks. Loadax provides a simple interface for prefetching data into a cache using background worker(s).

```python
from loadax import Dataloader, InMemoryDataset, Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
batcher = Batcher(lambda x: x)
loader = Dataloader(batcher).batch_size(2).prefetch(3).build(dataset)

for batch in loader:
    print(batch)

# Output:
# [1, 2]
# [3, 4]
# [5, 6]
# [7, 8]
# [9, 10]
```

In the above example we create a dataloader with a prefetch factor of 3. This means that the loader will prefetch 3 batches ahead of the current index. The future batches are kept in a cache, depending on your configuration can be eagerly loaded into device memory or kept in host memory.

### Using Multiple Workers

In the same way that the dataloader can be used to prefetch data, it can also offload the dataloading into multiple background workers. Lets take a look at an example of why you may want to do this.

In the following example we have a dataset that is slow to load an individual item due to some pre-processing. Ignore the details of the MappedDataset as we will get to that later, for now just know that it lazily transforms the data from the source dataset.

```python
from loadax import Dataloader, RangeDataset, MappedDataset, Batcher

def slow_fn(x):
    time.sleep(0.1)
    return x * 2

dataset = MappedDataset(RangeDataset(0, 10), slow_fn)
batcher = Batcher(lambda x: x)
loader = Dataloader(batcher).batch_size(2).workers(2).build(dataset)

for batch in loader:
    print(batch)

# Output:
# [0, 2]
# [4, 6]
# [8, 10]
# [12, 14]
# [16, 18]
```

In the above example we create a dataloader with 2 workers. This means that the loader will create 2 workers to load the data. The data is loaded in parallel, alowing the background workers to do the slow processing and then the data is batched and ready for consumption.

A important note is that the implementation of the background workers currently leverages the `concurrent.futures` library, because multiprocessing does not work well with JAX. This means each node is using a single python process and depending on your python version and how IO bound your datset loading is you may rarely see GIL contention.

### Distributed Data Loading

Loadax also supports distributed data loading. This means that you can easily shard your dataset across multiple nodes/jax processes and load data in parallel. Loadax will automatically determine which elements to load on each shard within the network ensuring that the data is evenly distributed, and each node only gets the data it needs.

With the inter-node distribution handled for you, it is now trivial to build advanced distributed training loops with paradigms such as model and data parallelism.

```python
from loadax import Dataloader, InMemoryDataset, Batcher
from loadax.sharding_utilities import fsdp_sharding
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp

# Standard jax distributed setup
...

# Create your own mesh or use the presets provided by loadax
mesh, axis_names = fsdp_sharding()

dataset_size = 128
batch_size = 8

# Create a host-specific (jax process) dataloader
dataset = InMemoryDataset(list(range(dataset_size)))
batcher = Batcher(lambda x: jnp.stack(x))

dataloader = (
    Dataloader(batcher)
        .batch_size(batch_size)
        .workers(2)
        .prefetch(2)
        .shard(mesh, data_axis_name=axis_names[0])
        .build(dataset)
    )

def create_model():
    # Create your model and shard however you like (see fsdp example for zero style fsdp)
    ...

model, optimizer = create_model()

def train_step(model, optimizer, batch):
    # Your loss calculation logic...
    return loss

for local_batch in dataloader:
    # Distribute the host-local-batch to create a sharded-global-batch
    # jax will not need to do any data movement since loadax can ensure the sharding is correct
    # for each host, unless you specify otherwise this also does not require contiguous device layout
    global_batch = dataloader.sharding_strategy.distribute_global_batch(local_batch)

    loss = train_step(model, optimizer, global_batch)
```

The sharding primitives that Loadax provides are powerful as they declare the way data is distributed up front. This enables loadax to be deterministic as is decides which elements to load on each process, and even which elements to load into each specific batch. This guaranteed determinism enables you to focus on other things rather than ensuring that your dataloading is correct and can be reproduced.

### Type Hinting

Another benefit of Loadax is that the underlying shape of your data is passed through all the way into your training loop. This means you can use type hints to ensure that your data is the correct shape.

```python
from loadax import Dataloader, RangeDataset, Batcher

# RangeDataset has a DatasetItem type of Int, this is a generic argument that can be supplied to any dataset
# type. We can look more into this when we get to datasets.
dataset = RangeDataset(0, 10)

# this function is inferred to return an int
def my_fn(x: list[int]) -> int:
    return sum(x)

batcher = Batcher(my_fn)
loader = Dataloader(batcher).batch_size(2).build(dataset)

for batch in loader:
    print(batch)

# Output:
# [1, 2]
# [3, 4]
# [5, 6]
# [7, 8]
# [9, 10]
```

Because you define the Batcher (or use a predefined one for common operations), the type of the batch can be inferred all the way from the dataset definition.

### Datasets

Loadax provides a simple interface for defining your dataset. As long as you can perform indexed access on your data, you can use Loadax to load your data. See the [Dataset Protocol](https://github.com/walln/loadax/blob/main/src/loadax/dataset/protocol.py) for more details.

Additionally, Loadax provides a few common datasets that can be used out of the box. These include:

- InMemoryDataset
- RangeDataset

```python
dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dataset = RangeDataset(0, 10)
```

Daasets can also be transformed using a variety of transformations. Transformations are lazily applied to the dataset, meaning that they are only applied when the data is actually accessed. Because your dataloader likely is prefetching and using background workers, this should not block your training loop. This also means that you can use jax to jit compile your transformation function.

```python
from loadax import MappedDataset, RangeDataset, ShuffledDataset

def slow_fn(x):
    time.sleep(0.1)
    return x * 2

base_dataset = ShuffledDataset(RangeDataset(0, 10))
dataset = MappedDataset(base_dataset, slow_fn)
```

When iterating through `dataset`, the the slow_fn will be applied lazily to the underlying dataset, which in itself is lazily shuffling the range dataset. This Composable pattern allows you to build complex dataloading pipelines.

### More Features

This was just a quick tour of what Loadax has to offer. For more information, please see the [documentation](https://walln.github.io/loadax/).

#### Dataset Integrations

Loadax has a few common dataset source on the roadmap, including:

- PolarsDataset
- SQLiteDataset
- HuggingFaceDataset

Feel free to open an issue if you have a use case that you would like to see included.
