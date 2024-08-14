from loadax.batcher import Batcher
from loadax.dataset import InMemoryDataset
from loadax.loader_builder import DataLoaderBuilder


def processing_function(items):
    # print(f"Items: {items}")
    return sum(items)


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
    values = []
    iterator = iter(dataloader)
    total_batches = (
        dataset_size + batch_size - 1
    ) // batch_size  # Ceiling division to get total batches

    for batch_index in range(total_batches):
        val = next(iterator)
        print(f"Value: {val}")
        values.append(val)
        print(f"Progress: {iterator.progress()}")

        # Check if the sum of items in the batch matches the expected sum
        expected_batch_sum = sum(
            range(
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, dataset_size),
            )
        )
        assert (
            val == expected_batch_sum
        ), f"Batch sum mismatch: got {val}, expected {expected_batch_sum}"

    return values


if __name__ == "__main__":
    dataset_size = 1000
    batch_size = 10
    num_workers = 4

    dataset = create_dataset(dataset_size)
    dataloader = create_dataloader(dataset, batch_size, num_workers)
    values = collect_values(dataloader, batch_size, dataset_size)

    # Assertions for final results
    total_batches = (dataset_size + batch_size - 1) // batch_size
    assert (
        len(values) == total_batches
    ), f"Number of batches mismatch: got {len(values)}, expected {total_batches}"

    print(f"Total values: {len(values)}")
    print(f"Sum of all values: {sum(values)}")

    expected_sum = sum(range(dataset_size))  # Sum of numbers from 0 to dataset_size-1
    actual_sum = sum(values)
    assert (
        actual_sum == expected_sum
    ), f"Sum mismatch: got {actual_sum}, expected {expected_sum}"
