from collections.abc import Iterator
from typing import Generic

import jax

from loadax.dataset.dataset import Dataset, Example
from loadax.dataset.sharded_dataset import (
    Shardable,
    compute_shard_boundaries,
)
from loadax.dataset.shuffled_dataset import Shuffleable


class SimpleDataset(
    Shardable[Example], Shuffleable[Example], Dataset[Example], Generic[Example]
):
    """A dataset that wraps a list of examples.

    Args:
        data (List[Example]): The list of data examples.
    """

    def __init__(self, data: list[Example]):
        """Initialize a simple dataset in-memory from a list.

        Args:
            data (List[Example]): The list of data examples.
        """
        self.data = data

    def __iter__(self) -> Iterator[Example]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index.

        Args:
            index (int): The index of the example to retrieve.

        Returns:
            Example: The data example at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.data[index]

    def split_dataset_by_node(self, world_size: int, rank: int) -> Dataset[Example]:
        """Split the dataset into shards.

        Args:
            world_size (int): The number of nodes.
            rank (int): The rank of the current node.

        Returns:
            Dataset[Example]: The shard of the dataset for the current node.
        """
        start, end = compute_shard_boundaries(
            num_shards=world_size,
            shard_id=rank,
            dataset_size=len(self),
            drop_remainder=False,
        )
        return SimpleDataset(self.data[start:end])

    def shuffle(self, seed: jax.Array) -> "Dataset[Example]":
        """Shuffle the dataset.

        Args:
            seed: The seed to use for the shuffle. This is a jax
            PRNGKey as all randomization in loadax is implemented using jax.random.

        Returns:
            The shuffled dataset.
        """
        indices = jax.random.permutation(seed, len(self))
        return SimpleDataset([self.data[i] for i in indices])
