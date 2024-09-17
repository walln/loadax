"""Huggingface Dataset Integration."""

from typing import TypeVar

import datasets  # type: ignore

from loadax.dataset import Dataset

DatasetItem = TypeVar("DatasetItem")


# TODO: walln - investigate how to optimize dataset.shard
class HuggingFaceDataset(Dataset[DatasetItem]):
    """Huggingface Dataset Integration.

    Args:
        dataset: Huggingface Dataset
    """

    def __init__(self, dataset: datasets.Dataset) -> None:
        """Create a loadax datset from a Huggingface Dataset.

        Args:
            dataset: Huggingface Dataset
        """
        self.dataset = dataset

    def get(self, index: int) -> DatasetItem | None:
        """Get the item at the given index.

        Examples:
            ```python
            from loadax.dataset.huggingface import HuggingFaceDataset
            import datasets as hf_datasets

            train_data = hf_datasets.load_dataset("stanfordnlp/imdb", split="train")

            train_dataset = HuggingFaceDataset(train_data)

            data = train_dataset.get(0)
            print(data)
            ```
        Args:
            index: The index of the item to get.

        Returns:
            DatasetItem | None: The item at the given index, or None if the value cannot
            be resolved.
        """
        item: DatasetItem | None = self.dataset[index]
        return item

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)

    @staticmethod
    def from_hub(
        path: str,
        name: str | None = None,
        split: str | None = None,
        *,
        trust_remote_code: bool = False,
    ) -> "HuggingFaceDataset[DatasetItem]":
        """Load a loadax dataset from the HuggingFace hub.

        Args:
            path: The path to the dataset on the HuggingFace hub.
            name: The name of the dataset on the HuggingFace hub.
            split: The split of the dataset on the HuggingFace hub.
            trust_remote_code: Whether to trust the remote code.

        Returns:
            HuggingFaceDataset[DatasetItem]: The loadax dataset.

        Example:
            ```python
            from loadax.dataset.huggingface import HuggingFaceDataset

            dataset = HuggingFaceDataset.from_hub("stanfordnlp/imdb")
            ```
        """
        dataset = datasets.load_dataset(
            path=path, name=name, split=split, trust_remote_code=trust_remote_code
        )

        dataset.set_format(type="numpy")

        assert isinstance(
            dataset, datasets.Dataset
        ), f"loaded dataset must be a Dataset, got {type(dataset)}"

        return HuggingFaceDataset[DatasetItem](dataset)
