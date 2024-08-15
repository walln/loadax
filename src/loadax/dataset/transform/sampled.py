import random
from loadax.dataset.protocol import Dataset
from typing import TypeVar

DatasetItem = TypeVar("DatasetItem")


class SampledDatasetWithoutReplacement(Dataset[DatasetItem]):
    def __init__(self, dataset: Dataset[DatasetItem], sample_size: int):
        self.dataset = dataset
        self.sample_size = sample_size
        self.indices = []

    def __len__(self):
        return self.sample_size

    def _index(self) -> int:
        if len(self.indices) == 0:
            self.indices = list(range(len(self.dataset)))
            # shuffle the indices
            random.shuffle(self.indices)
        return self.indices.pop()

    def get(self, index: int) -> DatasetItem | None:
        if index >= self.sample_size or len(self.dataset) == 0:
            return None
        return self.dataset.get(self._index())


class SampledDatasetWithReplacement(Dataset[DatasetItem]):
    def __init__(self, dataset: Dataset[DatasetItem], sample_size: int):
        self.dataset = dataset
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def get(self, index: int) -> DatasetItem | None:
        if index >= self.sample_size or len(self.dataset) == 0:
            return None

        # TODO: Look at using jax PRNG random sampling
        random_index = random.randint(0, len(self.dataset) - 1)
        return self.dataset.get(random_index)
