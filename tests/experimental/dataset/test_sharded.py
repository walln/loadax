import pytest

from loadax.experimental.dataset.sharded_dataset import compute_shard_boundaries


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "contiguous_shards", "drop_remainder", "expected"),
    [
        # Contiguous Shards without Drop Remainder
        (
            3,
            10,
            True,
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
            True,
            [
                (0, 3),  # Shard 0: 3 elements
                (3, 6),  # Shard 1: 3 elements
                (6, 9),  # Shard 2: 3 elements
            ],
        ),
        # Non-Contiguous Shards without Drop Remainder
        (
            3,
            10,
            False,
            False,
            [
                (0, 10, 3),  # Shard 0: indices [0, 3, 6, 9]
                (1, 10, 3),  # Shard 1: indices [1, 4, 7]
                (2, 10, 3),  # Shard 2: indices [2, 5, 8]
            ],
        ),
        # Non-Contiguous Shards with Drop Remainder
        (
            3,
            10,
            False,
            True,
            [
                (0, 9, 3),  # Shard 0: indices [0, 3, 6]
                (1, 9, 3),  # Shard 1: indices [1, 4, 7]
                (2, 9, 3),  # Shard 2: indices [2, 5, 8]
            ],
        ),
        # Single Shard (both contiguous and non-contiguous)
        (
            1,
            10,
            True,
            False,
            [(0, 10)],
        ),
        (
            1,
            10,
            False,
            True,
            [(0, 10, 1)],
        ),
        # Large Dataset Example
        (
            10,
            1000,
            True,
            False,
            "large_contiguous_no_drop",
        ),
        (
            10,
            1000,
            True,
            True,
            "large_contiguous_drop",
        ),
        (
            10,
            1000,
            False,
            False,
            "large_non_contiguous_no_drop",
        ),
        (
            10,
            1000,
            False,
            True,
            "large_non_contiguous_drop",
        ),
    ],
)
def test_compute_shard_boundaries(
    num_shards, dataset_size, contiguous_shards, drop_remainder, expected
):
    if expected == "large_contiguous_no_drop":
        for shard_id in range(num_shards):
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                contiguous_shards=contiguous_shards,
                drop_remainder=drop_remainder,
            )
            expected_size = (dataset_size // num_shards) + (
                1 if shard_id < (dataset_size % num_shards) else 0
            )
            assert (
                end - start == expected_size
            ), f"Shard {shard_id} size incorrect for large contiguous \
                 without drop_remainder."
    elif expected == "large_contiguous_drop":
        for shard_id in range(num_shards):
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                contiguous_shards=contiguous_shards,
                drop_remainder=drop_remainder,
            )
            expected_size = dataset_size // num_shards
            assert (
                end - start == expected_size
            ), f"Shard {shard_id} size incorrect for large contiguous \
                 with drop_remainder."
    elif expected == "large_non_contiguous_no_drop":
        for shard_id in range(num_shards):
            start, end, step = shard_id, dataset_size, num_shards
            actual_indices = list(range(start, end, step))
            expected_size = (dataset_size - shard_id + num_shards - 1) // num_shards
            assert (
                len(actual_indices) == expected_size
            ), f"Shard {shard_id} size incorrect for large non-contiguous \
                 without drop_remainder."
    elif expected == "large_non_contiguous_drop":
        for shard_id in range(num_shards):
            start, end, step = (
                shard_id,
                dataset_size - (dataset_size % num_shards),
                num_shards,
            )
            actual_indices = list(range(start, end, step))
            expected_size = dataset_size // num_shards
            assert (
                len(actual_indices) == expected_size
            ), f"Shard {shard_id} size incorrect for large non-contiguous \
                 with drop_remainder."
    elif isinstance(expected, list):
        for shard_id, exp in enumerate(expected):
            if isinstance(exp, tuple) and len(exp) == 2:
                # Contiguous Shards
                start, end = compute_shard_boundaries(
                    num_shards=num_shards,
                    shard_id=shard_id,
                    dataset_size=dataset_size,
                    contiguous_shards=contiguous_shards,
                    drop_remainder=drop_remainder,
                )
                assert (start, end) == exp, f"Shard {shard_id} bounds incorrect."
            elif isinstance(exp, tuple) and len(exp) == 3:
                # Non-Contiguous Shards
                start, end, step = exp
                actual_indices = list(range(start, end, step))
                expected_indices = list(range(exp[0], exp[1], exp[2]))
                assert (
                    actual_indices == expected_indices
                ), f"Shard {shard_id} indices incorrect."
    else:
        pytest.fail("Invalid expected value for parameterization.")


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "contiguous_shards", "drop_remainder"),
    [
        # Cases where dataset_size < num_shards should raise ValueError
        (5, 3, True, True),
        (5, 3, False, True),
        (5, 2, True, False),
        (5, 0, False, False),
    ],
)
def test_num_shards_greater_than_dataset_size(
    num_shards, dataset_size, contiguous_shards, drop_remainder
):
    for shard_id in range(num_shards):
        with pytest.raises(ValueError, match="Invalid dataset_size"):
            compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                contiguous_shards=contiguous_shards,
                drop_remainder=drop_remainder,
            )


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "contiguous_shards", "drop_remainder"),
    [
        (3, 0, True, True),
        (3, 0, True, False),
        (3, 0, False, True),
        (3, 0, False, False),
    ],
)
def test_dataset_size_zero(num_shards, dataset_size, contiguous_shards, drop_remainder):
    for shard_id in range(num_shards):
        with pytest.raises(ValueError, match="Invalid dataset_size"):
            compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                contiguous_shards=contiguous_shards,
                drop_remainder=drop_remainder,
            )


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "contiguous_shards", "drop_remainder", "expected"),
    [
        # Single Shard Tests
        (
            1,
            10,
            True,
            False,
            [(0, 10)],
        ),
        (
            1,
            10,
            False,
            True,
            [(0, 10, 1)],
        ),
    ],
)
def test_single_shard(
    num_shards, dataset_size, contiguous_shards, drop_remainder, expected
):
    for shard_id in range(num_shards):
        if isinstance(expected[0], tuple) and len(expected[0]) == 2:
            # Contiguous Shard
            start, end = compute_shard_boundaries(
                num_shards=num_shards,
                shard_id=shard_id,
                dataset_size=dataset_size,
                contiguous_shards=contiguous_shards,
                drop_remainder=drop_remainder,
            )
            assert (start, end) == expected[
                shard_id
            ], "Shard bounds incorrect for single shard."
        elif isinstance(expected[0], tuple) and len(expected[0]) == 3:
            # Non-Contiguous Shard
            start, end, step = expected[shard_id]
            actual_indices = list(range(start, end, step))
            expected_indices = list(range(start, end, step))
            assert (
                actual_indices == expected_indices
            ), "Shard indices incorrect for single shard."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "contiguous_shards", "drop_remainder", "shard_id"),
    [
        # Invalid Shard IDs
        (3, 10, True, False, -1),
        (3, 10, True, False, 3),
        (3, 10, True, False, 4),
    ],
)
def test_invalid_shard_id(
    num_shards, dataset_size, contiguous_shards, drop_remainder, shard_id
):
    with pytest.raises(ValueError, match="Invalid shard_id"):
        compute_shard_boundaries(
            num_shards=num_shards,
            shard_id=shard_id,
            dataset_size=dataset_size,
            contiguous_shards=contiguous_shards,
            drop_remainder=drop_remainder,
        )
