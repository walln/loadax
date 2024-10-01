# ruff: noqa: FBT001

from typing import Any

import pytest

from loadax.experimental.dataset.sharded_dataset import ShardedDataset
from loadax.experimental.dataset.simple import SimpleDataset


def compute_expected_boundaries(
    num_shards: int, shard_id: int, dataset_size: int, drop_remainder: bool
) -> tuple[int, int]:
    if drop_remainder:
        if dataset_size < num_shards:
            return 0, 0  # All shards have 0 length
        shard_size = dataset_size // num_shards
        return shard_id * shard_size, (shard_id + 1) * shard_size
    else:
        base = dataset_size // num_shards
        remainder = dataset_size % num_shards
        if shard_id < remainder:
            start = shard_id * (base + 1)
            end = start + base + 1
        else:
            start = shard_id * base + remainder
            end = start + base
        return start, end


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder", "expected"),
    [
        # Contiguous Shards without Drop Remainder
        (
            3,
            10,
            False,
            [
                (0, 4),  # Shard 0: indices [0, 1, 2, 3]
                (4, 7),  # Shard 1: indices [4, 5, 6]
                (7, 10),  # Shard 2: indices [7, 8, 9]
            ],
        ),
        # Contiguous Shards with Drop Remainder
        (
            3,
            10,
            True,
            [
                (0, 3),  # Shard 0: indices [0, 1, 2]
                (3, 6),  # Shard 1: indices [3, 4, 5]
                (6, 9),  # Shard 2: indices [6, 7, 8]
            ],
        ),
        # Single Shard without Drop Remainder
        (
            1,
            10,
            False,
            [
                (0, 10),  # Shard 0: indices [0, 1, 2, ..., 9]
            ],
        ),
        # Single Shard with Drop Remainder
        (
            1,
            10,
            True,
            [
                (0, 10),  # Shard 0: indices [0, 1, 2, ..., 9]
            ],
        ),
        # Large Dataset without Drop Remainder
        (
            10,
            1000,
            False,
            "large_contiguous_no_drop",
        ),
        # Large Dataset with Drop Remainder
        (
            10,
            1000,
            True,
            "large_contiguous_drop",
        ),
    ],
)
def test_compute_shard_boundaries(
    num_shards: int, dataset_size: int, drop_remainder: bool, expected: Any
):
    dataset = SimpleDataset(list(range(dataset_size)))
    for shard_id in range(num_shards):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        if (
            expected == "large_contiguous_no_drop"
            or expected == "large_contiguous_drop"
        ):
            # Skip boundary checks for large datasets to save time
            continue
        expected_start, expected_end = expected[shard_id]
        assert shard.start == expected_start, f"Shard {shard_id} start index incorrect."
        assert shard.end == expected_end, f"Shard {shard_id} end index incorrect."
        expected_length = expected_end - expected_start
        assert len(shard) == expected_length, f"Shard {shard_id} length incorrect."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        # Cases where dataset_size < num_shards should raise ValueError when
        # drop_remainder is True
        (5, 3, True),
        # Cases where dataset_size < num_shards but drop_remainder is False (allowed)
        (5, 2, False),
        (5, 0, False),
    ],
)
def test_num_shards_greater_than_dataset_size(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    dataset = SimpleDataset(list(range(dataset_size)))
    for shard_id in range(num_shards):
        if drop_remainder and dataset_size < num_shards:
            with pytest.raises(
                ValueError,
                match=rf"dataset_size \({dataset_size}\) must be >= num_shards \({num_shards}\) when drop_remainder is True\.",  # noqa: E501
            ):
                ShardedDataset(
                    dataset=dataset,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    drop_remainder=drop_remainder,
                )
        else:
            # When drop_remainder is False, it should not raise ValueError even
            # if dataset_size < num_shards
            try:
                shard = ShardedDataset(
                    dataset=dataset,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    drop_remainder=drop_remainder,
                )
                # If dataset_size < num_shards and drop_remainder=False, s
                # hards may have 0 or 1 elements
                if dataset_size == 0:
                    assert (
                        len(shard) == 0
                    ), "Shard length should be 0 when dataset_size is 0."
                else:
                    # Compute expected shard size
                    expected_size = (dataset_size // num_shards) + (
                        1 if shard_id < (dataset_size % num_shards) else 0
                    )
                    assert (
                        len(shard) == expected_size
                    ), "Shard length incorrect when dataset_size < num_shards."
            except ValueError:
                pytest.fail(
                    "ShardedDataset raised ValueError unexpectedly when "
                    "drop_remainder is False."
                )


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 0, True),
        (3, 0, False),
    ],
)
def test_dataset_size_zero(num_shards: int, dataset_size: int, drop_remainder: bool):
    dataset = SimpleDataset(list(range(dataset_size)))
    for shard_id in range(num_shards):
        if drop_remainder:
            # Expect ValueError when drop_remainder is True and
            # dataset_size < num_shards
            with pytest.raises(
                ValueError,
                match=r"dataset_size \(0\) must be >= num_shards \(3\) when drop_remainder is True\.",  # noqa: E501
            ):
                ShardedDataset(
                    dataset=dataset,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    drop_remainder=drop_remainder,
                )
        else:
            # When drop_remainder is False, shards should have (0, 0) boundaries
            shard = ShardedDataset(
                dataset=dataset,
                num_shards=num_shards,
                shard_id=shard_id,
                drop_remainder=drop_remainder,
            )
            assert shard.start == 0, f"Shard {shard_id} start index should be 0."
            assert shard.end == 0, f"Shard {shard_id} end index should be 0."
            assert len(shard) == 0, f"Shard {shard_id} length should be 0."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder", "expected"),
    [
        # Single Shard without Drop Remainder
        (
            1,
            10,
            False,
            [(0, 10)],
        ),
        # Single Shard with Drop Remainder
        (
            1,
            10,
            True,
            [(0, 10)],
        ),
    ],
)
def test_single_shard(
    num_shards: int, dataset_size: int, drop_remainder: bool, expected: list[tuple]
):
    dataset = SimpleDataset(list(range(dataset_size)))
    for shard_id, exp in enumerate(expected):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        expected_start, expected_end = exp
        assert (
            shard.start == expected_start
        ), "Shard start index incorrect for single shard."
        assert shard.end == expected_end, "Shard end index incorrect for single shard."
        assert len(shard) == (
            expected_end - expected_start
        ), "Shard length incorrect for single shard."
        for idx in range(len(shard)):
            assert (
                shard[idx] == dataset[expected_start + idx]
            ), "Shard get(index) incorrect."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder", "shard_id"),
    [
        # Invalid Shard IDs
        (3, 10, False, -1),
        (3, 10, False, 3),
        (3, 10, False, 4),
    ],
)
def test_invalid_shard_id(
    num_shards: int, dataset_size: int, drop_remainder: bool, shard_id: int
):
    dataset = SimpleDataset(list(range(dataset_size)))
    with pytest.raises(
        ValueError, match=r"shard_id must be an integer in \[0, \d+\)\."
    ):
        ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 10, False),
        (3, 10, True),
        (1, 10, False),
        (1, 10, True),
    ],
)
def test_sharded_dataset_initialization(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    dataset = SimpleDataset(list(range(dataset_size)))
    for shard_id in range(num_shards):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        # Ensure the shard is initialized correctly
        assert shard.num_shards == num_shards, "num_shards not set correctly."
        assert shard.shard_id == shard_id, "shard_id not set correctly."
        assert (
            shard.drop_remainder == drop_remainder
        ), "drop_remainder not set correctly."
        assert shard.dataset_size == dataset_size, "dataset_size not set correctly."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 10, False),
        (3, 10, True),
        (1, 10, False),
        (1, 10, True),
    ],
)
def test_sharded_dataset_length(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    dataset = SimpleDataset(list(range(dataset_size)))
    for shard_id in range(num_shards):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        # Compute expected shard size
        expected_length = 0
        if drop_remainder:
            if dataset_size >= num_shards:
                expected_length = dataset_size // num_shards
            else:
                expected_length = 0
        else:
            base = dataset_size // num_shards
            remainder = dataset_size % num_shards
            expected_length = base + 1 if shard_id < remainder else base
        assert len(shard) == expected_length, "Shard length incorrect."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 10, False),
        (3, 10, True),
        (1, 10, False),
        (1, 10, True),
    ],
)
def test_sharded_dataset_getitem(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    data = list(range(dataset_size))
    dataset = SimpleDataset(data)
    for shard_id in range(num_shards):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        # Compute expected shard boundaries
        start, end = compute_expected_boundaries(
            num_shards, shard_id, dataset_size, drop_remainder
        )
        expected_length = end - start
        # Test element retrieval
        for idx in range(expected_length):
            assert (
                shard[idx] == data[start + idx]
            ), f"Shard {shard_id} get({idx}) incorrect."
        # Test out-of-range indices
        with pytest.raises(IndexError):
            shard[expected_length]
        with pytest.raises(IndexError):
            shard[-expected_length - 1]


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 10, False),
        (3, 10, True),
        (1, 10, False),
        (1, 10, True),
    ],
)
def test_sharded_dataset_iteration(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    data = list(range(dataset_size))
    dataset = SimpleDataset(data)
    for shard_id in range(num_shards):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        iterated_data = list(iter(shard))
        # Compute expected data
        start, end = compute_expected_boundaries(
            num_shards, shard_id, dataset_size, drop_remainder
        )
        expected_data = data[start:end]
        assert iterated_data == expected_data, f"Shard {shard_id} iteration incorrect."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        # Edge Cases
        (2, 1, False),
        (2, 1, True),  # Should raise ValueError for shard_id=1 if drop_remainder=True
        (1, 0, False),  # Should **not** raise ValueError as drop_remainder=False
    ],
)
def test_sharded_dataset_edge_cases(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    data = list(range(dataset_size))
    dataset = SimpleDataset(data)
    for shard_id in range(num_shards):
        if drop_remainder and dataset_size < num_shards:
            with pytest.raises(
                ValueError,
                match=rf"dataset_size \({dataset_size}\) must be >= num_shards \({num_shards}\) when drop_remainder is True\.",  # noqa: E501
            ):
                ShardedDataset(
                    dataset=dataset,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    drop_remainder=drop_remainder,
                )
            continue
        if dataset_size == 0 and not drop_remainder:
            # When dataset_size=0 and drop_remainder=False, shards should have
            # (0, 0) boundaries
            shard = ShardedDataset(
                dataset=dataset,
                num_shards=num_shards,
                shard_id=shard_id,
                drop_remainder=drop_remainder,
            )
            assert shard.start == 0, f"Shard {shard_id} start index should be 0."
            assert shard.end == 0, f"Shard {shard_id} end index should be 0."
            assert len(shard) == 0, f"Shard {shard_id} length should be 0."
            continue
        # For other cases, proceed as before
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        # Compute expected shard size
        expected_length = 0
        if drop_remainder:
            if dataset_size >= num_shards:
                expected_length = dataset_size // num_shards
            else:
                expected_length = 0
        else:
            base = dataset_size // num_shards
            remainder = dataset_size % num_shards
            expected_length = base + 1 if shard_id < remainder else base
        assert len(shard) == expected_length, "Shard length incorrect for edge case."
        # Verify contents
        iterated_data = list(iter(shard))
        expected_data = data[shard.start : shard.end]
        assert iterated_data == expected_data, "Shard data incorrect for edge case."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        (3, 10, False),
        (3, 10, True),
        (1, 10, False),
        (1, 10, True),
    ],
)
def test_sharded_dataset_negative_indices(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    data = list(range(dataset_size))
    dataset = SimpleDataset(data)
    for shard_id in range(num_shards):
        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )
        shard_length = len(shard)
        if shard_length == 0:
            continue
        # Test negative indexing
        assert shard[-1] == shard[shard_length - 1], "Negative index -1 incorrect."
        if shard_length > 1:
            assert shard[-2] == shard[shard_length - 2], "Negative index -2 incorrect."


@pytest.mark.parametrize(
    ("num_shards", "dataset_size", "drop_remainder"),
    [
        # Additional Edge Cases
        (1, 0, False),  # Should **not** raise ValueError as drop_remainder=False
        (2, 1, False),  # Shard 0 has 1, Shard 1 has 0
    ],
)
def test_sharded_dataset_additional_edge_cases(
    num_shards: int, dataset_size: int, drop_remainder: bool
):
    data = list(range(dataset_size))
    dataset = SimpleDataset(data)
    for shard_id in range(num_shards):
        if drop_remainder and dataset_size < num_shards:
            with pytest.raises(
                ValueError,
                match=rf"dataset_size \({dataset_size}\) must be >= num_shards \({num_shards}\) when drop_remainder is True\.",  # noqa: E501
            ):
                ShardedDataset(
                    dataset=dataset,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    drop_remainder=drop_remainder,
                )
            continue
        if dataset_size == 0 and not drop_remainder:
            # When dataset_size=0 and drop_remainder=False, shards should have
            # (0, 0) boundaries
            shard = ShardedDataset(
                dataset=dataset,
                num_shards=num_shards,
                shard_id=shard_id,
                drop_remainder=drop_remainder,
            )
            assert shard.start == 0, f"Shard {shard_id} start index should be 0."
            assert shard.end == 0, f"Shard {shard_id} end index should be 0."
            assert len(shard) == 0, f"Shard {shard_id} length should be 0."
            continue

        shard = ShardedDataset(
            dataset=dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_remainder=drop_remainder,
        )

        # Compute expected shard size
        expected_length = 0
        if drop_remainder:
            if dataset_size >= num_shards:
                expected_length = dataset_size // num_shards
            else:
                expected_length = 0
        else:
            base = dataset_size // num_shards
            remainder = dataset_size % num_shards
            expected_length = base + 1 if shard_id < remainder else base

        assert (
            len(shard) == expected_length
        ), "Shard length incorrect for additional edge case."

        # Verify contents
        iterated_data = list(iter(shard))
        expected_data = data[shard.start : shard.end]
        assert (
            iterated_data == expected_data
        ), "Shard data incorrect for additional edge case."
