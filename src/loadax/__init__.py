"""Loadax is a library for efficiently loading data from datasets."""

__version__ = "0.0.1"


# Batcher is exposed as a top level import as its often imported for convenience.
from loadax.batcher import Batcher as Batcher

# Dataloader builder is exposed as a top level import, individual loaders can be
# imported from loadax.dataloader if needed.
from loadax.dataloader import DataLoader as DataLoader

## Datasets
# All dataset types are exposed as top level imports.
from loadax.dataset import Dataset as Dataset
from loadax.dataset import InMemoryDataset as InMemoryDataset
from loadax.dataset.protocol import DatasetItem as DatasetItem
from loadax.dataset.protocol import DatasetIterator as DatasetIterator
from loadax.dataset.range import RangeDataset as RangeDataset

# Dataset transformations are also exposed as top level imports as they
# are often used in conjunction with other datasets, and are also
# technically dataset types.
from loadax.dataset.transform import CombinedDataset as CombinedDataset
from loadax.dataset.transform import MappedDataset as MappedDataset
from loadax.dataset.transform import (
    PartialDataset as PartialDataset,
)
from loadax.dataset.transform import (
    SampledDatasetWithoutReplacement as SampledDatasetWithoutReplacement,
)
from loadax.dataset.transform import (
    SampledDatasetWithReplacement as SampledDatasetWithReplacement,
)
from loadax.dataset.transform import ShuffledDataset as ShuffledDataset
