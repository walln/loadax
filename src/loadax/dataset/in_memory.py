from loadax.dataset.dataset import Dataset
from typing import Generic, TypeVar

T = TypeVar("T")


class InMemoryDataset(Dataset, Generic[T]):
    def __init__(self, items: list[T]):
        self.items = items

    def get(self, index: int) -> T | None:
        if index >= len(self.items):
            return None

        if index < 0:
            index = len(self.items) + index
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"InMemoryDataset(items={self.items[:2]}...)"

    def from_dataset(dataset: Dataset[T]) -> "InMemoryDataset[T]":
        items = [item for item in dataset]
        return InMemoryDataset(items)

    # TODO: implement from different file types, ex: from_csv, from_json, from_parquet
