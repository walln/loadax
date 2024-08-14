from typing import Generic, Protocol
from typing import TypeVar

T = TypeVar("T")


class BatchStrategy(Protocol, Generic[T]):
    def __init__(self):
        raise NotImplementedError

    def add(self, item: T):
        raise NotImplementedError

    def batch(self, force: bool) -> list[T] | None:
        raise NotImplementedError


class FixedBatchStrategy(BatchStrategy[T]):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.items: list[T] = []

    def add(self, item: T):
        self.items.append(item)

    def batch(self, force: bool) -> list[T] | None:
        if len(self.items) < self.batch_size and not force:
            return None

        items = self.items.copy()
        self.items.clear()

        if not items:
            return None

        return items
