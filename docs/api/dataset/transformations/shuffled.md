# ShuffledDataset

A shuffled dataset is a simple dataset that shuffles the items in the underlying dataset. The shuffling is performed lazily and does not actually shuffle the underlying storage. If your underlying storage does not perform well with random access, you may want to consider performing the shuffling in advance. However, for almost all use cases, this should not be necessary.

The shuffling procedure is deterministic and leverages JAX's random number generation.

```python title="Creating a shuffled dataset"
from loadax import ShuffledDataset, InMemoryDataset
import jax

dataset = InMemoryDataset([1, 2, 3, 4, 5])
key = jax.random.PRNGKey(0)
shuffled_dataset = ShuffledDataset(dataset, key)
```

::: loadax.dataset.transform.ShuffledDataset
    selection:
      members: false
