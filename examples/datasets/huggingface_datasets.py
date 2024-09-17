"""Example of using the Huggingface Integration.

With loadax you can use huggingface datasets to feed your training loop. The
dataset is eagerly downloaded and cached, but lazily loaded from the memmapped
arrow table. This means that you can access your data with maximum performance
and without loading the entire dataset into memory.

HuggingFaceDataset is also composable with other datasets and transformations because
it implements the Dataset protocol. In this example we can create a lazy split of
the dataset. You can imagine the kinds of things you can do that do not require
preprocessing the entire dataset and bloating your huggingface cache.

You can also construct a HuggingFaceDataset by passing in a datasets.Dataset into
the constructor. This is useful if you want to do some preprocessing and store the
results in the huggingface dataset cache.
"""

from typing import TypedDict

import datasets

from loadax.dataset.huggingface import HuggingFaceDataset
from loadax.dataset.transform import PartialDataset

# You can construct a HuggingFaceDataset from the huggingface hub using the `from_hub`
# static method. This method takes in the path to the dataset on the huggingface hub,
# the name of the dataset, and the split of the dataset. The dataset is lazily loaded
# from the huggingface cache.


# Here we create the type annotation for the dataset. This is not required, but it
# helps with type checking and autocompletion.
class Element(TypedDict):
    """Element is a single row from the dataset.

    The shape of the element is determined by the dataset, check out
    the dataset on the huggingface hub for more information.
    """

    text: str
    label: int


train_dataset = HuggingFaceDataset[Element].from_hub(
    "stanfordnlp/imdb", split="train", trust_remote_code=True
)

dataset = PartialDataset(train_dataset, 0, 10)
assert len(dataset) == 10, "The dataset size is incorrect!"
assert len(dataset) < len(train_dataset), "The dataset size is incorrect!"


data = dataset.get(0)
assert data is not None, "The dataset is empty!"
print(data["text"])

# You can also load the dataset and make your changes outside of loadax
eager_mapped_dataset = datasets.load_dataset(
    "stanfordnlp/imdb", split="train", trust_remote_code=True
)

eager_mapped_dataset.set_format(type="numpy")

transformed_dataset = eager_mapped_dataset.map(
    lambda x: {"text": x["text"], "label": x["label"]}, batched=False
)

assert isinstance(
    transformed_dataset, datasets.Dataset
), f"loaded dataset must be a Dataset, got {type(transformed_dataset)}"

dataset_transformed = HuggingFaceDataset[Element](transformed_dataset)

data_transformed = dataset_transformed.get(0)
assert data_transformed is not None, "The dataset is empty!"
print(data_transformed["text"])
