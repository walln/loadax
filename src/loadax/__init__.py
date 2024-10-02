"""Loadax is a library for efficiently loading data from datasets."""

__version__ = "0.0.1"

# Dataloader builder is exposed as a top level import, individual loaders can be
# imported from loadax.dataloader if needed.
from loadax.dataloader import Dataloader as Dataloader
from loadax.dataset.combined_dataset import CombinedDataset as CombinedDataset

## Datasets
# All dataset types are exposed as top level imports.
from loadax.dataset.dataset import Dataset as Dataset
from loadax.dataset.huggingface import HuggingFaceDataset as HuggingFaceDataset
from loadax.dataset.partial_dataset import PartialDataset as PartialDataset
from loadax.dataset.sampled_dataset import SampledDataset as SampledDataset
from loadax.dataset.sharded_dataset import ShardedDataset as ShardedDataset
from loadax.dataset.shuffled_dataset import Shuffleable as Shuffleable
from loadax.dataset.simple import SimpleDataset as SimpleDataset
