from collections.abc import Iterator
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from loadax.dataset.dataset import Dataset, Example
from loadax.dataset.sharded_dataset import Shardable
from loadax.logger import logger


class HuggingFaceDataset(Shardable[Example], Dataset[Example]):
    """A dataset that integrates with Hugging Face's `datasets` library."""

    def __init__(
        self,
        dataset: HFDataset,
    ):
        """Initialize a huggingface dataset that has already been loaded.

        Any huggingface compatible dataset can be loaded with loadax to leverage
        the rich ecosystem of datasets, tooling, and efficient arrow-backed tables.

        If you are loading large datasets in a multi-host environment it is important
        to think about the order you load data for sharding. If you intend to shard
        your data such that each host is not fully replicated you will need to
        identify how to split the dataset.

        A huggingface dataset is sharded when loaded if using
        `HuggingFaceDataset.from_hub(..., num_shards=n, shard_id=i)`. Otherwise
        you will want to pre-shard it yourself.

        Examples:
            ```python
            from loadax.experimental.dataset.huggingface import HuggingFaceDataset
            import datasets as hf_datasets

            train_data = hf_datasets.load_dataset("stanfordnlp/imdb", split="train")
            train_data.shard(num_shards=2, shard_id=0)

            train_dataset = HuggingFaceDataset(train_data)

            data = train_dataset.get(0)
            print(data)
            ```

        Alternatively you can use ShardedDataset to wrap a HuggingFaceDataset. This
        will perform the same sharding algorithm as the datasets library however based
        on your usage it may not prevent the extra network overhead of loading all
        shards.

        Args:
            dataset: HuggingFace Dataset
        """
        self._dataset = dataset

    @staticmethod
    def from_hub(
        path: str,
        name: str | None = None,
        split: str | None = None,
    ) -> "HuggingFaceDataset[Example]":
        """Load a HuggingFace dataset from the HuggingFace hub.

        Args:
            path: The path to the dataset on the HuggingFace hub.
            name: The name of the dataset on the HuggingFace hub.
            split: The split of the dataset on the HuggingFace hub.

        Returns:
            HuggingFaceDataset[Example]: The HuggingFace dataset.

        Example:
            ```python
            from loadax.experimental.dataset.huggingface import HuggingFaceDataset

            dataset = HuggingFaceDataset.from_hub("stanfordnlp/imdb")
            ```
        """
        dataset = load_dataset(
            path=path, name=name, split=split, trust_remote_code=True
        )
        dataset.set_format(type="numpy")

        assert isinstance(
            dataset, HFDataset
        ), f"loaded dataset must be a Dataset, got {type(dataset)}"

        logger.info(f"Loaded HF dataset with length: {len(dataset)}")
        return HuggingFaceDataset[Example](dataset)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        """Retrieve an example by its index.

        Args:
            index (int): The index of the example to retrieve.

        Returns:
            Any: The data example at the specified index.
        """
        return self.dataset[index]

    def split_dataset_by_node(self, world_size: int, rank: int) -> Dataset[Example]:
        """Split the dataset into shards.

        Args:
            world_size (int): The number of nodes.
            rank (int): The rank of the current node.

        Returns:
            Dataset[Example]: The shard of the dataset for the current node.
        """
        from datasets.distributed import (
            split_dataset_by_node as hf_split_dataset_by_node,
        )

        dataset = hf_split_dataset_by_node(self._dataset, rank, world_size)
        assert isinstance(dataset, HFDataset)
        return HuggingFaceDataset[Example](dataset)

    @property
    def dataset(self) -> HFDataset:
        """The underlying HuggingFace dataset."""
        return self._dataset
