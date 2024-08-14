from loadax.batcher import Batcher
from loadax.dataset import InMemoryDataset
from loadax.loader_builder import DataLoaderBuilder


def processing_function(items):
    # print(f"Items: {items}")
    return [2 * item for item in items]


def create_dataset(size):
    dataset = InMemoryDataset([v for v in range(size)])
    assert (
        len(dataset) == size
    ), f"Dataset size mismatch: got {len(dataset)}, expected {size}"
    return dataset


def create_dataloader(dataset, batch_size, num_workers):
    batcher = Batcher(processing_function)
    dataloader = (
        DataLoaderBuilder(batcher)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(dataset)
    )
    return dataloader


def collect_values(dataloader, batch_size, dataset_size):
    batches = []

    for batch_index, batch in enumerate(dataloader):
        val = batch
        # print(f"Value: {val}")
        # print(f"Progress: {iterator.progress()}")
        print(f"Batch index: {batch_index} of size {len(val)}")

        # Check if the sum of items in the batch matches the expected sum
        assert (
            len(val) == batch_size
        ), f"Batch size mismatch: got {len(val)}, expected {batch_size}"
        assert all(
            item % 2 == 0 for item in val
        ), f"Batch values must be even: got {val}"

        batches.append(batch_index)

    return batches


if __name__ == "__main__":
    dataset_size = 1_000
    batch_size = 10
    num_workers = 4

    dataset = create_dataset(dataset_size)
    dataloader = create_dataloader(dataset, batch_size, num_workers)
    batches = collect_values(dataloader, batch_size, dataset_size)

    # Assertions for final results
    total_batches = (dataset_size + batch_size - 1) // batch_size

    print(f"Total batches: {len(batches)}")
