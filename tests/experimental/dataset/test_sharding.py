import pytest

from loadax.experimental.dataset.sharded_dataset import compute_shard_boundaries


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder", "expected"),
    [
        # Contiguous Shards without Drop Remainder
        (
            3,
            10,
            False,
            [
                (0, 4),  # Shard 0: 4 elements
                (4, 7),  # Shard 1: 3 elements
                (7, 10),  # Shard 2: 3 elements
            ],
        ),
        # Contiguous Shards with Drop Remainder
        (
            3,
            10,
            True,
            [
                (0, 3),  # Shard 0: 3 elements
                (3, 6),  # Shard 1: 3 elements
                (6, 9),  # Shard 2: 3 elements
            ],
        ),
        # Single Shard (Contiguous)
        (
            1,
            10,
            False,
            [(0, 10)],
        ),
        (
            1,
            10,
            True,
            [(0, 10)],
        ),
        # Large Dataset Example
        (
            10,
            1000,
            False,
            "large_contiguous_no_drop",
        ),
        (
            10,
            1000,
            True,
            "large_contiguous_drop",
        ),
    ],
)
def test_compute_shard_boundaries(num_shards, dataset_size, drop_remainder, expected):
    if expected == "large_contiguous_no_drop":
        for shard_id in range(num_shards):
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                drop_remainder=drop_remainder,
            )
            expected_size = (dataset_size // num_shards) + (
                1 if shard_id < (dataset_size % num_shards) else 0
            )
            assert (
                end - start == expected_size
            ), f"Shard {shard_id} size incorrect for large_contiguous_no_drop."
    elif expected == "large_contiguous_drop":
        for shard_id in range(num_shards):
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                drop_remainder=drop_remainder,
            )
            expected_size = dataset_size // num_shards
            assert (
                end - start == expected_size
            ), f"Shard {shard_id} size incorrect for large_contiguous_drop."
    elif isinstance(expected, list):
        for shard_id, exp in enumerate(expected):
            # Contiguous Shards
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                drop_remainder=drop_remainder,
            )
            assert (start, end) == exp, f"Shard {shard_id} bounds incorrect."
    else:
        pytest.fail("Invalid expected value for parameterization.")


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        # Cases where dataset_size < num_shards should raise ValueError
        # when drop_remainder=True
        (5, 3, True),
        # Cases where dataset_size < num_shards but drop_remainder=False (allowed)
        (5, 2, False),
        (5, 0, False),
    ],
)
def test_num_shards_greater_than_dataset_size(num_shards, dataset_size, drop_remainder):
    for shard_id in range(num_shards):
        if drop_remainder and dataset_size < num_shards:
            expected_message = rf"Invalid dataset_size: {dataset_size}. Must be >= num_shards \({num_shards}\) when drop_remainder is True\."  # noqa: E501
            with pytest.raises(ValueError, match=expected_message):
                compute_shard_boundaries(
                    num_shards=num_shards,
                    shard_id=shard_id,
                    dataset_size=dataset_size,
                    drop_remainder=drop_remainder,
                )
        else:
            # Should not raise, check boundaries
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                drop_remainder=drop_remainder,
            )
            if drop_remainder:
                # When drop_remainder=True and dataset_size >= num_shards
                expected_size = dataset_size // num_shards
            else:
                # When drop_remainder=False, shard sizes can vary
                base_size = dataset_size // num_shards
                remainder = dataset_size % num_shards
                expected_size = base_size + (1 if shard_id < remainder else 0)
            assert (end - start) == expected_size, f"Shard {shard_id} size incorrect."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 0, True),
        (3, 0, False),
    ],
)
def test_dataset_size_zero(num_shards, dataset_size, drop_remainder):
    for shard_id in range(num_shards):
        if drop_remainder and dataset_size < num_shards:
            expected_message = rf"Invalid dataset_size: {dataset_size}. Must be >= num_shards \({num_shards}\) when drop_remainder is True\."  # noqa: E501
            with pytest.raises(ValueError, match=expected_message):
                compute_shard_boundaries(
                    num_shards=num_shards,
                    shard_id=shard_id,
                    dataset_size=dataset_size,
                    drop_remainder=drop_remainder,
                )
        else:
            # When drop_remainder is False, shards should have (0, 0) boundaries
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                drop_remainder=drop_remainder,
            )
            assert start == 0, f"Shard {shard_id} start index should be 0."
            assert end == 0, f"Shard {shard_id} end index should be 0."
            assert (end - start) == 0, f"Shard {shard_id} length should be 0."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder", "expected"),
    [
        # Single Shard Tests
        (
            1,
            10,
            False,
            [(0, 10)],
        ),
        (
            1,
            10,
            True,
            [(0, 10)],
        ),
    ],
)
def test_single_shard(num_shards, dataset_size, drop_remainder, expected):
    for shard_id in range(num_shards):
        if isinstance(expected[0], tuple) and len(expected[0]) == 2:
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                drop_remainder=drop_remainder,
            )
            assert (start, end) == expected[
                shard_id
            ], "Shard bounds incorrect for single shard."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder", "shard_id"),
    [
        # Invalid Shard IDs
        (3, 10, False, -1),
        (3, 10, False, 3),
        (3, 10, False, 4),
    ],
)
def test_invalid_shard_id(num_shards, dataset_size, drop_remainder, shard_id):
    expected_message = (
        rf"Invalid shard_id: {shard_id}. Must be in \[0, {num_shards}\)\."
    )
    with pytest.raises(ValueError, match=expected_message):
        compute_shard_boundaries(
            num_shards=num_shards,
            shard_id=shard_id,
            dataset_size=dataset_size,
            drop_remainder=drop_remainder,
        )
