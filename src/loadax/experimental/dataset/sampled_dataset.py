from collections.abc import Iterator
from typing import Generic

import jax
import jax.numpy as jnp

from loadax.experimental.dataset.dataset import Dataset, Example


class SampledDataset(Dataset[Example], Generic[Example]):
    """A dataset that represents a random sample of another dataset.

    This dataset type allows you to create a new dataset that contains
    a random subset of elements from an existing dataset, specified by
    a sample size and a random key.
    """

    def __init__(self, dataset: Dataset[Example], sample_size: int, key: jax.Array):
        """Initialize the SampledDataset.

        Args:
            dataset: The original dataset to create a sampled view of.
            sample_size: The number of samples to include in the new dataset.
            key: The random key to use for sampling.
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.key = key

        if sample_size < 0 or sample_size > len(dataset):
            raise ValueError("Invalid sample size")

        self.indices = jax.random.choice(
            key, jnp.arange(len(dataset)), shape=(sample_size,), replace=False
        )

    def __iter__(self) -> Iterator[Example]:
        """Return an iterator over the sampled dataset."""
        return (self.dataset[int(i)] for i in self.indices)

    def __len__(self) -> int:
        """Return the number of samples in the sampled dataset."""
        return self.sample_size

    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index from the sampled dataset.

        Args:
            index: The index of the example to retrieve.

        Returns:
            The data example at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        return self.dataset[int(self.indices[index])]
