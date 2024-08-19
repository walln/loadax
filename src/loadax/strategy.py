"""Batching strategies for fulling batches with loaded data."""

from collections import deque
from typing import Generic, Protocol, TypeVar

DatasetItem = TypeVar("DatasetItem")


class BatchStrategy(Protocol, Generic[DatasetItem]):
    """The method for determining how much data goes into a batch.

    This protocol defines the interface for determining how much data goes into a batch
    as it is loaded from the dataset.

    Attributes:
        batch_size: The size of the batch.
    """

    batch_size: int

    def add(self, item: DatasetItem) -> None:
        """Add an item to the batch.

        Args:
            item: The item to add to the batch.
        """
        ...

    def batch(self, *, force: bool) -> list[DatasetItem] | None:
        """Get the batch.

        This method method returns the batch if it is full, or None if the batch is not
        full.

        Args:
            force: Whether to force the batch to be returned even if it is not full.

        Returns:
            The batch, or None if the batch is not full.
        """
        ...

    def clone(self) -> "BatchStrategy[DatasetItem]":
        """Clone the batch strategy.

        This is likely not necessary but exists in advanced use cases such as sharing
        a batch strategy between multiple processes.

        Returns:
            A clone of the batch strategy.
        """
        ...


class FixedBatchStrategy(BatchStrategy[DatasetItem]):
    """A batch strategy that batches items into a fixed size batch.

    This batch strategy batches items into a fixed size batch. It will always return
    the same batch size, and will return None if the batch is not full. This is
    almost always what you want.

    Attributes:
        batch_size: The size of the batch.
        items: The items in the batch.
    """

    def __init__(self, batch_size: int):
        """A batch strategy that batches items into a fixed size batch.

        This batch strategy batches items into a fixed size batch. It will always return
        the same batch size, and will return None if the batch is not full. This is
        almost always what you want.

        Args:
            batch_size: The size of the batch.
        """
        self.batch_size = batch_size
        self.items: deque[DatasetItem] = deque()

    def add(self, item: DatasetItem) -> None:
        """Add an item to the batch.

        Args:
            item: The item to add to the batch.
        """
        self.items.append(item)

    def batch(self, *, force: bool) -> list[DatasetItem] | None:
        """Get the batch.

        This method method returns the batch if it is full, or None if the batch is not
        full.

        Args:
            force: Whether to force the batch to be returned even if it is not full.

        Returns:
            The batch, or None if the batch is not full.
        """
        if len(self.items) >= self.batch_size:
            return [self.items.popleft() for _ in range(self.batch_size)]
        elif force and self.items:
            return list(self.items)
        return None

    def clone(self) -> "FixedBatchStrategy[DatasetItem]":
        """Clone the batch strategy.

        This is likely not necessary but exists in advanced use cases such as sharing
        a batch strategy between multiple processes.

        Returns:
            A clone of the batch strategy.
        """
        new_strategy: FixedBatchStrategy[DatasetItem] = FixedBatchStrategy(
            self.batch_size
        )
        new_strategy.items = self.items.copy()
        return new_strategy
