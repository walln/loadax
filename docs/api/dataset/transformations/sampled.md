# SampledDataset

A sampled dataset is a simple dataset that returns a subset of the underlying dataset. The sampling is performed lazily and does not actually sample the underlying storage. If your underlying storage does not perform well with random access, you may want to consider performing the sampling in advance. However, for almost all use cases, this should not be necessary.

The sampling procedure is deterministic and leverages JAX's random number generation.

```python title="Creating a sampled dataset"
from loadax import SampledDatasetWithReplacement, InMemoryDataset
import jax

dataset = InMemoryDataset([1, 2, 3, 4, 5])
key = jax.random.PRNGKey(0)
sampled_dataset = SampledDatasetWithReplacement(dataset, 3, key)
```

::: loadax.dataset.transform.SampledDatasetWithoutReplacement
    selection:
      members: false

::: loadax.dataset.transform.SampledDatasetWithReplacement
    selection:
      members: false
