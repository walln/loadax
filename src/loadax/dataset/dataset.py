from typing import Generic, Protocol
from typing import TypeVar

T = TypeVar("T")


class DatasetIterator(Protocol, Generic[T]):
    def __init__(self, dataset: "Dataset[T]"):
        self.current = 0
        self.dataset = dataset

    def __next__(self) -> T | None:
        item = self.dataset.get(self.current)
        self.current += 1
        return item


class Dataset(Protocol, Generic[T]):
    def __init__(self):
        raise NotImplementedError

    def get(self, index: int) -> T | None:
        raise NotImplementedError

    def is_empty(self) -> bool:
        return len(self) == 0

    def __iter__(self) -> DatasetIterator[T]:
        return DatasetIterator(self)

    def __len__(self) -> int:
        raise NotImplementedError
