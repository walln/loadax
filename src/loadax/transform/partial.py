from loadax.dataset import Dataset
from typing import TypeVar

T = TypeVar("T")


class PartialDataset(Dataset[T]):
    def __init__(self, dataset: Dataset[T], start_index: int, end_index: int):
        self.dataset = dataset
        self.start_index = start_index
        self.end_index = end_index

    @staticmethod
    def split(dataset: Dataset, num_parts: int) -> list[Dataset[T]]:
        total_len = len(dataset)
        batch_size = total_len // num_parts
        remainder = total_len % num_parts
        datasets = []

        current = 0
        for i in range(num_parts):
            start = current
            end = start + batch_size + (1 if i < remainder else 0)
            datasets.append(PartialDataset(dataset, start, end))
            current = end

        return datasets

    def get(self, index: int) -> T | None:
        if index < 0 or index >= len(self):
            # raise IndexError(f"Index {index} out of range")
            return None
        return self.dataset.get(index + self.start_index)
        # index = index + self.start_index
        # if index < self.start_index or index >= self.end_index:
        #     return None
        # return self.dataset.get(index)

    def __len__(self) -> int:
        return min(self.end_index - self.start_index, len(self.dataset))