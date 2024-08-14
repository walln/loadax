from typing import Generic, Protocol
from typing import TypeVar

T = TypeVar("T")


class BatchStrategy(Protocol, Generic[T]):
    batch_size: int

    def __init__(self):
        raise NotImplementedError

    def add(self, item: T):
        raise NotImplementedError

    def batch(self, force: bool) -> list[T] | None:
        raise NotImplementedError

    def clone(self) -> "BatchStrategy[T]":
        raise NotImplementedError


class FixedBatchStrategy(BatchStrategy[T]):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.items: list[T] = []

    def add(self, item: T):
        self.items.append(item)

    def batch(self, force: bool) -> list[T] | None:
        if len(self.items) >= self.batch_size:
            items = self.items[: self.batch_size]
            self.items = self.items[self.batch_size :]
            return items
        elif force and self.items:
            items = self.items
            self.items = []
            return items

    def clone(self) -> "FixedBatchStrategy[T]":
        new_strategy = FixedBatchStrategy(self.batch_size)
        new_strategy.items = self.items.copy()
        return new_strategy
