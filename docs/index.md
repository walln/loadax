# Loadax

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/walln/loadax/ci.yml)
![PyPI - Status](https://img.shields.io/pypi/status/loadax)
![PyPI - Downloads](https://img.shields.io/pypi/dm/loadax)
![GitHub License](https://img.shields.io/github/license/walln/loadax)

Loadax is a dataloading library designed for the JAX ecosystem. It provides utilities for feeding data into your training loop without having to worry about batching, shuffling, and other preprocessing steps. Loadax also handles
background prefetching to improve performance, and distriubted data loading to train on multiple devices and even multiple hosts.

```py title="Loadax Example"
from loadax import DataloaderBuilder, InMemoryDataset, Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)
loader = DataloaderBuilder(batcher).batch_size(2).build(dataset)

for batch in loader:
    print(batch)

#> [1, 2]
#> [3, 4]
#> [5]
```

## Installation

```bash
uv add loadax
```
