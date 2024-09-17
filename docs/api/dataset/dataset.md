# Dataset

A dataset is a simple interface that defines how to load data from a source. All datasets must implement this
interface, and it is the responsibility of the dataset to load the data from the underlying source. Because
datasets support random access, they should also know their size.

```python title="Creating a dataset"
from loadax import Dataset

class MyDataset(Dataset[int]):
    def __init__(self, items: list[int]):
        self.items = items

    def get(self, index: int) -> int:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

dataset = MyDataset([1, 2, 3, 4, 5])

for i in range(len(dataset)):
    print(dataset.get(i))

#> 1
#> 2
#> 3
#> 4
#> 5
```

::: loadax.dataset.protocol.Dataset
    selection:
      members: false
