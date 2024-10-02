import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from loadax.sharding.partition_spec import DataPartitionType
from loadax.sharding.placement import (
    global_to_host_array,
    host_to_global_device_array,
    with_sharding_constraint,
)


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_host_to_global_device_array(simulated_xla_devices):
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))
    with mesh:
        host_array = np.array([[1, 2], [3, 4]])
        global_array = host_to_global_device_array(
            host_array, partition=DataPartitionType.FULL
        )

        assert isinstance(global_array, jax.Array)
        assert global_array.shape == host_array.shape
        assert np.array_equal(np.array(global_array), host_array)


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_global_to_host_array(simulated_xla_devices):
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))
    with mesh:
        global_array = jax.numpy.array([[1, 2], [3, 4]])
        host_array = global_to_host_array(
            global_array, partition=DataPartitionType.FULL
        )

        assert isinstance(host_array, np.ndarray)
        assert host_array.shape == global_array.shape
        assert np.array_equal(host_array, np.array(global_array))


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_with_sharding_constraint(simulated_xla_devices):
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))
    with mesh:
        x = jax.numpy.array([1, 2, 3, 4])
        sharded_x = with_sharding_constraint(x, PartitionSpec("data"))

        assert isinstance(sharded_x, jax.Array)
        assert np.array_equal(np.array(sharded_x), np.array(x))


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
@pytest.mark.parametrize(
    "partition", [DataPartitionType.FULL, DataPartitionType.REPLICATED]
)
def test_host_to_global_device_array_partition_types(simulated_xla_devices, partition):
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))
    device_count = len(simulated_xla_devices)
    with mesh:
        host_array = np.array([[i, i + 1] for i in range(0, device_count * 2, 2)])
        global_array = host_to_global_device_array(host_array, partition=partition)

        assert isinstance(global_array, jax.Array)
        assert global_array.shape == host_array.shape
        assert np.array_equal(np.array(global_array), host_array)

        if partition == DataPartitionType.FULL:
            assert len(global_array.sharding.device_set) == device_count
        elif partition == DataPartitionType.REPLICATED:
            assert len(global_array.sharding.device_set) == 1


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_host_to_global_device_array_nested(simulated_xla_devices):
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))
    with mesh:
        host_nested = {"a": np.array([1, 2]), "b": {"c": np.array([3, 4])}}
        global_nested = host_to_global_device_array(
            host_nested, partition=DataPartitionType.FULL
        )

        assert isinstance(global_nested, dict)
        assert isinstance(global_nested["a"], jax.Array)
        assert isinstance(global_nested["b"]["c"], jax.Array)
        assert np.array_equal(np.array(global_nested["a"]), host_nested["a"])
        assert np.array_equal(np.array(global_nested["b"]["c"]), host_nested["b"]["c"])


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_global_to_host_array_nested(simulated_xla_devices):
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))
    with mesh:
        global_nested = {
            "a": jax.numpy.array([1, 2]),
            "b": {"c": jax.numpy.array([3, 4])},
        }
        host_nested = global_to_host_array(
            global_nested, partition=DataPartitionType.FULL
        )

        assert isinstance(host_nested, dict)
        assert isinstance(host_nested["a"], np.ndarray)
        assert isinstance(host_nested["b"]["c"], np.ndarray)
        assert np.array_equal(host_nested["a"], np.array(global_nested["a"]))
        assert np.array_equal(host_nested["b"]["c"], np.array(global_nested["b"]["c"]))


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_host_to_global_device_array_multi_device(simulated_xla_devices):
    device_count = len(simulated_xla_devices)
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))

    with mesh:
        host_array = np.array([[i, i + 1] for i in range(0, device_count * 2, 2)])
        global_array = host_to_global_device_array(
            host_array, partition=DataPartitionType.FULL
        )

        assert isinstance(global_array, jax.Array)
        assert global_array.shape == host_array.shape
        assert np.array_equal(np.array(global_array), host_array)
        assert len(global_array.sharding.device_set) == device_count


@pytest.mark.parametrize("simulated_xla_devices", [4, 8], indirect=True)
def test_global_to_host_array_multi_device(simulated_xla_devices):
    device_count = len(simulated_xla_devices)
    mesh = jax.sharding.Mesh(simulated_xla_devices, ("data",))

    with mesh:
        global_array = jax.numpy.array(
            [[i, i + 1] for i in range(0, device_count * 2, 2)]
        )
        host_array = global_to_host_array(
            global_array, partition=DataPartitionType.FULL
        )

        assert isinstance(host_array, np.ndarray)
        assert host_array.shape == global_array.shape
        assert np.array_equal(host_array, np.array(global_array))
