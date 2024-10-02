from abc import abstractmethod
from collections.abc import Iterator
from typing import Generic, Protocol, runtime_checkable

from loadax.experimental.dataset.dataset import Dataset, Example


def compute_shard_boundaries(
    num_shards: int,
    shard_id: int,
    dataset_size: int,
    *,
    drop_remainder: bool = True,
) -> tuple[int, int]:
    """Compute the start and end indices for a specific shard.

    This function deterministically computes the boundaries of a shard given the total
    number of shards, the shard ID, and the size of the dataset.

    Args:
        num_shards (int): The total number of shards.
        shard_id (int): The ID of the current shard (0-based).
        dataset_size (int): The total size of the dataset.
        drop_remainder (bool, optional): If True, drops the last incomplete shard
            ensuring all shards have equal size. Defaults to True.

    Returns:
        Tuple[int, int]: A tuple containing the start (inclusive) and end (exclusive)
            indices for the shard.

    Raises:
        ValueError: If shard_id is out of range or if drop_remainder is True
            and dataset_size is less than num_shards.
    """
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"Invalid shard_id: {shard_id}. Must be in [0, {num_shards}).")

    if drop_remainder and dataset_size < num_shards:
        raise ValueError(
            f"Invalid dataset_size: {dataset_size}. Must be >= num_shards "
            f"({num_shards}) when drop_remainder is True."
        )

    if drop_remainder:
        shard_size = dataset_size // num_shards
        start = shard_size * shard_id
        end = start + shard_size
    else:
        # Distribute the remainder across the first 'remainder' shards
        base_size, remainder = divmod(dataset_size, num_shards)
        if shard_id < remainder:
            start = (base_size + 1) * shard_id
            end = start + base_size + 1
        else:
            start = (base_size + 1) * remainder + base_size * (shard_id - remainder)
            end = start + base_size

    # Ensure the end does not exceed dataset_size
    end = min(end, dataset_size)

    return start, end


@runtime_checkable
class Shardable(Protocol, Generic[Example]):
    """A shardable dataset must implement the Shardable protocol.

    Each dataset has to implement sharding by itself because the underlying
    storage may have unique constraints to consider when creating the sharding
    boundaries.
    """

    @abstractmethod
    def split_dataset_by_node(self, world_size: int, rank: int) -> Dataset[Example]:
        """Split the dataset into shards.

        If possible the shards should be of equal size and non-overlapping
        and continguous.

        Args:
            world_size (int): The number of nodes.
            rank (int): The rank of the current node.

        Returns:
            Dataset[Example]: The shard of the dataset for the current node.
        """
        pass


class ShardedDataset(Dataset[Example], Generic[Example]):
    """Divides the dataset into non-overlapping contiguous shards."""

    def __init__(
        self,
        dataset: Dataset[Example],
        num_shards: int,
        shard_id: int,
        *,
        drop_remainder: bool = True,
    ):
        """Initialize a ShardedDataset to shard the given dataset.

        Args:
            dataset (Dataset[E]): The underlying dataset to shard.
            num_shards (int): Total number of shards.
            shard_id (int): The ID of the current shard (0-based).
            drop_remainder (bool, optional): Whether to drop the last incomplete shard.
                Defaults to True.

        Raises:
            TypeError: If `dataset` is not an instance of Dataset.
            ValueError: If `num_shards` is not a positive integer.
            ValueError: If `shard_id` is not in the range [0, num_shards).
            ValueError: If `drop_remainder` is True and `dataset_size` < `num_shards`.
        """
        if not isinstance(dataset, Shardable):
            raise TypeError("dataset must implement the Shardable protocol.")
        if not isinstance(num_shards, int) or num_shards <= 0:
            raise ValueError("num_shards must be a positive integer.")
        if not isinstance(shard_id, int) or not (0 <= shard_id < num_shards):
            raise ValueError(f"shard_id must be an integer in [0, {num_shards}).")

        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.drop_remainder = drop_remainder
        self.dataset_size = len(self.dataset)

        if self.drop_remainder and self.dataset_size < self.num_shards:
            raise ValueError(
                f"dataset_size ({self.dataset_size}) must be >= num_shards "
                f"({self.num_shards}) when drop_remainder is True."
            )

        self.start, self.end = compute_shard_boundaries(
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            dataset_size=self.dataset_size,
            drop_remainder=self.drop_remainder,
        )

        self._length = max(0, self.end - self.start)

    def __iter__(self) -> Iterator[Example]:
        """Iterate over the examples in the shard."""
        for idx in range(self.start, self.end):
            yield self.dataset[idx]

    def __len__(self) -> int:
        """Return the number of examples in the shard."""
        return self._length

    def __getitem__(self, index: int) -> Example:
        """Retrieve an example by its index within the shard.

        Args:
            index (int): The index of the example to retrieve.

        Returns:
            E: The data example at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(
                f"Index {index} out of range for shard with length {len(self)}."
            )

        actual_index = self.start + index
        return self.dataset[actual_index]

    def shard_boundaries(self) -> tuple[int, int]:
        """Return the start and end boundaries of the shard.

        Returns:
            Tuple[int, int]: The (start, end) indices of the shard.
        """
        return self.start, self.end
