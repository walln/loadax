def compute_shard_boundaries(
    num_shards: int,
    shard_id: int,
    dataset_size: int,
    *,
    contiguous_shards: bool = True,
    drop_remainder: bool = True,
) -> tuple[int, int]:
    """Compute the bounds of a shard.

    This deterministically computes the bounds of a shard given the number of shards,
    the shard ID, and the size of the dataset.
    `compute_shard_boundaries(n, i, dataset_size)` creates a shard with all elements
    of the dataset which have an element index modulo `n` equal to `i`.

    If `contiguous_shards` is True, the shards are contiguous such that the global index
    ordering is preserved. If `dataset_size % num_shards == l` then the first `l`
    shards will be of length `(dataset_size // num_shards) + 1`
    and the remaining shards will have length `(dataset_size // num_shards)`.

    If `drop_remainder` is True, the shards will be of length
    `(dataset_size // num_shards)` and the last `l` values will be dropped.
    This is useful in multi-host training to ensure that each host has the
    same amount of data.

    Args:
        num_shards: The number of shards.
        shard_id: The ID of the shard.
        dataset_size: The size of the dataset.
        contiguous_shards: If True, the shards are contiguous. Otherwise, they are
            non-contiguous.
        drop_remainder: Forces each shard to have the same length, ignoring the last
            `l` values if `l = dataset_size % num_shards` such that `l > 0`.

    Returns:
        A tuple of (start, end) indices for the shard.
    """
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"Invalid shard_id: {shard_id}. Must be in [0, {num_shards}).")

    if dataset_size < num_shards:
        raise ValueError(
            f"Invalid dataset_size: {dataset_size}. Must be greater than or equal "
            f"to num_shards."
        )

    if drop_remainder:
        # Calculate the size of each shard when dropping the remainder
        shard_size = dataset_size // num_shards

        if shard_size == 0:
            # If shard_size is 0, all shards should have (0,0)
            return (0, 0)

        # Calculate the total size that will be used (excluding the remainder)
        total_size = shard_size * num_shards

        if contiguous_shards:
            start = shard_size * shard_id
            end = start + shard_size
        else:
            # For non-contiguous shards, calculate indices based on stride
            start = shard_id
            end = start + shard_size * num_shards
            # Ensure we don't exceed the total_size
            end = min(end, total_size)
    else:
        if contiguous_shards:
            # Calculate base size and the number of shards that will have one
            # extra element
            base_size = dataset_size // num_shards
            remainder = dataset_size % num_shards

            if shard_id < remainder:
                start = (base_size + 1) * shard_id
                end = start + base_size + 1
            else:
                start = (base_size + 1) * remainder + base_size * (shard_id - remainder)
                end = start + base_size
        else:
            # Non-contiguous shards without dropping the remainder
            # Each shard takes every num_shards-th element starting from shard_id
            # The end is set to the dataset_size to cover all possible elements
            start = shard_id
            end = dataset_size

    # Ensure that end does not exceed dataset_size
    end = min(end, dataset_size)

    return (start, end)
