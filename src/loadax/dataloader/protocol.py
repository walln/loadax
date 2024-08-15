from typing import Protocol, Generic
from typing import TypeVar

from loadax.dataloader.progress import Progress

DatasetItem = TypeVar("DatasetItem")
Batch = TypeVar("Batch")


class DataLoaderIteratorProtocol(Protocol, Generic[Batch]):
    def progress(self) -> Progress: ...

    def __next__(self) -> Batch: ...


class DataLoaderProtocol(Protocol, Generic[DatasetItem, Batch]):
    def __iter__(self) -> DataLoaderIteratorProtocol[Batch]: ...
    def num_items(self) -> int: ...
