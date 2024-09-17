# Progress

Progress is a simple interface that allows you to track the progress of a dataloader. Because a dataloader is a stateful iterator, you can use the `progress()` method to get the current progress of the iterator. This means no more
manual calculations of the progress, dealing with batch sizes, or forgetting to update your progress tracking.

```python title="Tracking progress"
from loadax import Dataloader, InMemoryDataset, Batcher

dataset = InMemoryDataset([1, 2, 3, 4, 5])
batcher = Batcher(lambda x: x)
dataloader = Dataloader(batcher).batch_size(2).build(dataset)

iterator = iter(dataloader)

for batch in iterator:
    print(batch)
    iterator.progress()

#> [1, 2]
#> Progress(items_processed=2, items_total=5)
#> [3, 4]
#> Progress(items_processed=4, items_total=5)
#> [5]
#> Progress(items_processed=5, items_total=5)
```

::: loadax.dataloader.progress.Progress
    selection:
      members: true  
