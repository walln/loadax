from loadax.dataset.protocol import Dataset
from typing import TypeVar
import random

DatasetItem = TypeVar("DatasetItem")


class ShuffledDataset(Dataset[DatasetItem]):
    def __init__(self, dataset: Dataset[DatasetItem]):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.dataset)

    def get(self, index: int) -> DatasetItem | None:
        if index < 0 or index >= len(self.indices):
            return None
        return self.dataset.get(self.indices[index])

    def __repr__(self) -> str:
        return f"ShuffledDataset(dataset={self.dataset})"
