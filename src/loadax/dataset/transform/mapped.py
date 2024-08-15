from loadax.dataset.protocol import Dataset
from typing import Callable, TypeVar

DatasetItem = TypeVar("DatasetItem")
MappedItem = TypeVar("MappedItem")


class MappedDataset(Dataset[MappedItem]):
    def __init__(
        self, dataset: Dataset[DatasetItem], map_fn: Callable[[DatasetItem], MappedItem]
    ):
        self.dataset = dataset
        self.map_fn = map_fn

    def get(self, index: int) -> MappedItem | None:
        item = self.dataset.get(index)
        if item is None:
            return None
        return self.map_fn(item)

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return f"MappedDataset(dataset={self.dataset}, map_fn={self.map_fn})"
