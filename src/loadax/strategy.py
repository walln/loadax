from typing import Generic, Protocol
from typing import TypeVar

DatasetItem = TypeVar("DatasetItem")


class BatchStrategy(Protocol, Generic[DatasetItem]):
    batch_size: int

    def __init__(self):
        raise NotImplementedError

    def add(self, item: DatasetItem):
        raise NotImplementedError

    def batch(self, force: bool) -> list[DatasetItem] | None:
        raise NotImplementedError

    def clone(self) -> "BatchStrategy[DatasetItem]":
        raise NotImplementedError


class FixedBatchStrategy(BatchStrategy[DatasetItem]):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.items: list[DatasetItem] = []

    def add(self, item: DatasetItem):
        self.items.append(item)

    def batch(self, force: bool) -> list[DatasetItem] | None:
        if len(self.items) >= self.batch_size:
            items = self.items[: self.batch_size]
            self.items = self.items[self.batch_size :]
            return items
        elif force and self.items:
            items = self.items
            self.items = []
            return items

    def clone(self) -> "FixedBatchStrategy[DatasetItem]":
        new_strategy = FixedBatchStrategy(self.batch_size)
        new_strategy.items = self.items.copy()
        return new_strategy
