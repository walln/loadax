# Loadax

Loadax is a dataloading library designed for the JAX ecosystem. It provides utilities for feeding data into your training loop without having to worry about batching, shuffling, and other preprocessing steps. Loadax also supports offloading data loading to the background, and prefetching a cache to improve performance. Going forward Loadax will implement suppport for the following dataloading strategies:

- Multi-Host Data Loading
- Data Parallelism
- Data Parallelism with Sharding
- Data Parallelism with Sharding and Prefetching

## Installation

```bash
pip install loadax
```

## Usage

### Data Loading

Loadax provides a simple interface for loading data into your training loop. Here is an example of loading data from a list of items:

```python
from loadax.dataloader import DataLoader
from loadax.dataset import InMemoryDataset
from loadax.batcher import Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
batcher = Batcher(lambda x: x)
loader = DataLoader(batcher).batch_size(2).build(dataset)

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
dataloader = DataLoader(batcher).batch_size(2).build(dataset)

fast_iterator = iter(dataloader)
slow_iterator = iter(dataloader)

val = next(fast_iterator)
print(val)
# Output: 1

val = next(slow_iterator)
print(val)
# Output: 1
```

In the above examples we create an object called a batcher. A batcher is an interface that defines how to collate your data into batches. This is useful for when you want to alter the way your data is batched such as stacking into a single array.

### Data Prefetching

When training models, it is essential to ensure that you are not blocking the training loop and especially your accelerator(s), with IO bound tasks. Loadax provides a simple interface for prefetching data into a cache with a background process.

```python
from loadax.dataloader import DataLoader
from loadax.dataset import InMemoryDataset
from loadax.batcher import Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
batcher = Batcher(lambda x: x)
loader = DataLoader(batcher).batch_size(2).prefetch(3).build(dataset)

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

### Multi-Processing Data Loading

In the same way that the dataloader can be used to prefetch data, it can also offload the dataloading into multiple processes. Lets take a look at an example of why you may want to do this.

In the following example we have a dataset that is slow to load an individual item due to some pre-processing. Ignore the details of the MappedDataset as we will get to that later, for now just know that it lazily transforms the data from the source dataset.

```python
from loadax.dataloader import DataLoader
from loadax.dataset import RangeDataset
from loadax.dataset.transform import MappedDataset
from loadax.batcher import Batcher

def slow_fn(x):
    time.sleep(0.1)
    return x * 2

dataset = MappedDataset(RangeDataset(0, 1000000), slow_fn)
batcher = Batcher(lambda x: x)
loader = DataLoader(batcher).batch_size(2).workers(2).build(dataset)

for batch in loader:
    print(batch)

# Output:
# [0, 2]
# [4, 6]
# [8, 10]
# [12, 14]
# [16, 18]
# [20, 22]
# [24, 26]
# [28, 30]
# [32, 34]
# [36, 38]
```

In the above example we create a dataloader with 2 workers. This means that the loader will create 2 processes to load the data. The data is loaded in parallel, alowing the background workers to do the slow processing and then the data is batched and ready for consumption.

### Type Hints

Another benefit of Loadax is that the underlying shape of your data is passed through all the way into your training loop. This means you can use type hints to ensure that your data is the correct shape.

```python
from loadax.dataloader import DataLoader
from loadax.dataset import RangeDataset
from loadax.batcher import Batcher

# RangeDataset has a DatasetItem type of Int, this is a generic argument that can be supplied to any dataset
# type. We can look more into this when we get to datasets.
dataset = RangeDataset(0, 10)

# this function is inferred to return an int
def my_fn(x: list[int]) -> int:
    return sum(x)

batcher = Batcher(my_fn)
loader = DataLoader(batcher).batch_size(2).build(dataset)

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
