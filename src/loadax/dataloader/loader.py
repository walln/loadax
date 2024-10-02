"""Dataloader that loads batches in the background or synchronously."""

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import Generic

from loadax.dataloader.progress import Progress
from loadax.dataset.dataset import Dataset, Example


class DataloaderIterator(Generic[Example]):
    """Iterator for the dataloader."""

    executor: ThreadPoolExecutor | None
    exception: Exception | None
    buffer: Queue[list[Example]]

    def __init__(self, dataloader: "Dataloader[Example]"):
        """Iterator for the dataloader.

        Args:
            dataloader (Dataloader): The dataloader to iterate over.
        """
        self.dataloader = dataloader
        self.current_index = 0
        self.buffer = Queue(maxsize=max(1, self.dataloader.prefetch_factor))
        self.exception = None

        if self.dataloader.num_workers > 0:
            self.executor = ThreadPoolExecutor(max_workers=self.dataloader.num_workers)
            self.stop_event = threading.Event()
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self.prefetch_thread.start()
        else:
            self.executor = None

    def _prefetch_worker(self) -> None:
        while not self.stop_event.is_set():
            if self.current_index >= len(self.dataloader.dataset):
                break
            try:
                batch = self._load_batch(self.current_index)
                # Use a timeout when putting items into the buffer
                timeout = 0.1  # 100ms
                while not self.stop_event.is_set():
                    try:
                        self.buffer.put(batch, timeout=timeout)
                        break
                    except Full:
                        continue
                self.current_index += self.dataloader.batch_size
            except Exception as e:
                self.exception = e
                break

    def _load_batch(self, batch_start: int) -> list[Example]:
        batch_end = min(
            batch_start + self.dataloader.batch_size, len(self.dataloader.dataset)
        )
        return [self.dataloader.dataset[i] for i in range(batch_start, batch_end)]

    def __next__(self) -> list[Example]:
        if self.executor:
            if self.exception:
                raise self.exception
            try:
                batch = self.buffer.get(timeout=0.1)  # 100ms timeout
            except Empty:
                if self.current_index >= len(self.dataloader.dataset):
                    raise StopIteration from None
                return self.__next__()  # Try again
            if not batch and self.current_index >= len(self.dataloader.dataset):
                raise StopIteration
        else:
            if self.current_index >= len(self.dataloader.dataset):
                raise StopIteration
            batch = self._load_batch(self.current_index)
            self.current_index += self.dataloader.batch_size

        if len(batch) < self.dataloader.batch_size and self.dataloader.drop_last:
            raise StopIteration

        return batch

    def __iter__(self) -> "DataloaderIterator[Example]":
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __del__(self) -> None:
        if self.executor:
            self.stop_event.set()
            # Wait for the prefetch thread to finish with a timeout
            self.prefetch_thread.join(timeout=5.0)
            self.executor.shutdown(wait=False)
            # Clear the buffer to unblock any waiting put() calls
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except Empty:
                    break

    def progress(self) -> Progress:
        """Get the progress of the dataloader."""
        total_items = len(self.dataloader.dataset)
        processed_items = min(self.current_index, total_items)
        return Progress(processed_items, total_items)


class Dataloader(Generic[Example]):
    """Dataloader that loads batches in the background or synchronously."""

    def __init__(
        self,
        dataset: Dataset[Example],
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int = 0,
        *,
        drop_last: bool = False,
    ):
        """A dataloader that can load data in the background or synchronously.

        Example:
            ```python
            from loadax.experimental.dataset.simple import SimpleDataset
            from loadax.experimental.loader import Dataloader

            dataset = SimpleDataset([1, 2, 3, 4, 5])
            dataloader = Dataloader(
                dataset=dataset,
                batch_size=2,
                num_workers=2,
                prefetch_factor=2,
                drop_last=False,
            )
            for batch in dataloader:
                print(batch)

            #> [1, 2]
            #> [3, 4]
            #> [5]
            ```

        Args:
            dataset (Dataset): The dataset to load data from.
            batch_size (int): The size of each batch.
            num_workers (int): The number of workers to use for parallel data loading.
                If 0, data will be loaded synchronously.
            prefetch_factor (int): The prefetch factor to use for prefetching.
                If 0, no prefetching will occur.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

    def __iter__(self) -> DataloaderIterator[Example]:
        return DataloaderIterator(self)

    def __len__(self) -> int:
        num_examples = len(self.dataset)
        num_full_batches = num_examples // self.batch_size
        has_partial_batch = (num_examples % self.batch_size) > 0

        if self.drop_last:
            return num_full_batches
        else:
            return num_full_batches + (1 if has_partial_batch else 0)
