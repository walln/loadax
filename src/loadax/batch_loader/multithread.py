from queue import Queue
import threading
from loadax.dataloader import DataLoader, DataLoaderIterator, Progress
from typing import TypeVar

T = TypeVar("Item")


class Message:
    class Batch:
        def __init__(self, index: int, item: T, progress: Progress):
            self.index = index
            self.item = item
            self.progress = progress

    class Done:
        def __init__(self, index: int):
            self.index = index


MAX_QUEUED_ITEMS = 100


class MultiThreadedDataLoaderIterator(DataLoaderIterator[T]):
    def __init__(
        self, queue: Queue, workers: list[threading.Thread], progresses: list[Progress]
    ):
        self.queue = queue
        self.workers = workers
        self.progresses = progresses
        self.num_done = 0

    def __next__(self) -> T:
        if not self.workers:
            raise StopIteration

        while True:
            try:
                item = self.queue.get(timeout=1.0)
            except Queue.Empty:
                if self.num_done == len(self.workers):
                    self._join_workers()
                    raise StopIteration
                continue

            if isinstance(item, Message.Batch):
                self.progresses[item.index] = item.progress
                return item.item
            elif isinstance(item, Message.Done):
                print(f"Worker {item.index} done")
                self.num_done += 1

            if self.num_done == len(self.workers):
                self._join_workers()
                raise StopIteration

    def _join_workers(self):
        print("Joining workers")
        for worker in self.workers:
            worker.join()
        self.workers.clear()

    def __iter__(self):
        return self

    def progress(self) -> Progress:
        items_total = sum(progress.items_total for progress in self.progresses)
        items_processed = sum(progress.items_processed for progress in self.progresses)

        return Progress(items_processed, items_total)


class MultiThreadedBatchDataLoader:
    def __init__(self, dataloaders: list):
        self.dataloaders = dataloaders

    def __iter__(self) -> MultiThreadedDataLoaderIterator[T]:
        queue = Queue(MAX_QUEUED_ITEMS)
        progresses = [
            Progress(0, dataloader.num_items()) for dataloader in self.dataloaders
        ]

        def worker(index: int, dataloader: DataLoader[T]):
            iterator: DataLoaderIterator[T] = iter(dataloader)
            for item in iterator:
                progress = iterator.progress()
                queue.put(Message.Batch(index, item, progress))
            queue.put(Message.Done(index))
            print(f"Worker {index} sent done message")

        workers = []
        for index, dataloader in enumerate(self.dataloaders):
            t = threading.Thread(target=worker, args=(index, dataloader))
            t.start()
            workers.append(t)

        return MultiThreadedDataLoaderIterator(
            queue, workers=workers, progresses=progresses
        )

    # split the dataset into num_threads chunks
    # create a dataloader for each chunk
    # create a new PRNG key for each loader

    def num_items(self) -> int:
        return sum(dataloader.num_items() for dataloader in self.dataloaders)
