from loadax.dataset.protocol import Dataset
from typing import TypeVar

DatasetItem = TypeVar("DatasetItem")


class CombinedDataset(Dataset[DatasetItem]):
    def __init__(self, datasets: list[Dataset[DatasetItem]]):
        self.datasets = datasets

    def get(self, index: int) -> DatasetItem | None:
        current_index = 0
        for dataset in self.datasets:
            if index < current_index + len(dataset):
                return dataset.get(index - current_index)
            current_index += len(dataset)
        return None

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __repr__(self) -> str:
        return f"CombinedDataset(datasets={self.datasets})"
