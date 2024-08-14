from typing import Protocol, Generic
from typing import TypeVar

T = TypeVar("I")


class Progress:
    items_processed: int
    items_total: int

    def __init__(self, items_processed: int, items_total: int):
        self.items_processed = items_processed
        self.items_total = items_total

    def __repr__(self):
        return f"Progress(items_processed={self.items_processed}, items_total={self.items_total})"


class DataLoaderIterator(Protocol, Generic[T]):
    def progress(self) -> Progress: ...


class DataLoader(Protocol, Generic[T]):
    def __iter__(self) -> DataLoaderIterator: ...
    def num_items(self) -> int: ...
