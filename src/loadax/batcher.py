"""Batcher is a protocol for batching data."""

from collections.abc import Callable
from typing import Generic, TypeVar

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


class Batcher(Generic[DatasetItem, Batch]):
    """Batcher is a protocol for batching data.

    A batcher is responsible for taking a list of items and returning a batch. The
    batcher is responsible for handling any necessary transformations, such as
    stacking, padding, and other preprocessing steps. The batcher is also responsible
    for handling any necessary device placement, such as moving the data to the correct
    device. (For now, this is subject to change as Sharding is implemented.)

    An additional benefit of the batcher is that the data type hints can be inferred
    from the supplied batching function. This means that you can define batchers will
    then supply type hints in your training loop.

    Example:
        >>> batcher = Batcher(lambda x: x)
        >>> batch = batcher.batch([1, 2, 3, 4, 5])
        >>> print(batch)

    Attributes:
        batch_fn (Callable[[list[DatasetItem]], Batch]): The function to use for
            batching.
        device (str | None): The device to place the data on.
    """

    def __init__(
        self, batch_fn: Callable[[list[DatasetItem]], Batch], device: str | None = None
    ):
        """Batcher is a protocol for batching data.

        A batcher is responsible for taking a list of items and returning a batch. The
        batcher is responsible for handling any necessary transformations, such as
        stacking, padding, and other preprocessing steps. The batcher is also
        responsible for handling any necessary device placement, such as moving the
        data to the correct device. (For now, this is subject to change as Sharding
        is implemented.)

        An additional benefit of the batcher is that the data type hints can be inferred
        from the supplied batching function. This means that you can define batchers
        will then supply type hints in your training loop.

        Example:
            >>> batcher = Batcher(lambda x: x)
            >>> batch = batcher.batch([1, 2, 3, 4, 5])
            >>> print(batch)

        Args:
            batch_fn (Callable[[list[DatasetItem]], Batch]): The function to use for
                batching.
            device (str | None): The device to place the data on.
        """
        self.batch_fn = batch_fn
        self.device = device

    def batch(self, data: list[DatasetItem]) -> Batch:
        """Batch the given data.

        This method hands off control of the loaded data into your batching function.

        Example:
            >>> batcher = Batcher(lambda x: x)
            >>> batch = batcher.batch([1, 2, 3, 4, 5])
            >>> print(batch)

        Args:
            data (list[DatasetItem]): The data to batch.

        Returns:
            Batch: The batched data.
        """
        return self.batch_fn(data)
