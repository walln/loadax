import os
import re

import jax
import pytest

from loadax.sharding.mesh_shape import HybridMeshShape, MeshConfig


def test_hybrid_mesh_shape_initialization():
    hybrid_shape = HybridMeshShape(ici_mesh_shape=(1, 4), dcn_mesh_shape=(1, 1))
    assert hybrid_shape.ici_mesh_shape == (1, 4)
    assert hybrid_shape.dcn_mesh_shape == (1, 1)
    assert len(hybrid_shape) == 2


def test_hybrid_mesh_shape_validation():
    with pytest.raises(
        ValueError, match="The mesh shapes must have the same number of axes"
    ):
        HybridMeshShape(ici_mesh_shape=(1, 4), dcn_mesh_shape=(1, 1, 1))


@pytest.mark.parametrize(
    ("mesh_shape", "mesh_axis_names", "batch_axis_names"),
    [
        ((2, 4), ("data", "model"), "data"),
        ((1, 2, 8), ("pipeline", "data", "model"), ["data"]),
        (HybridMeshShape((1, 4), (2, 1)), ("data", "model"), "data"),
    ],
)
def test_mesh_config_initialization(mesh_shape, mesh_axis_names, batch_axis_names):
    config = MeshConfig(
        mesh_shape=mesh_shape,
        mesh_axis_names=mesh_axis_names,
        batch_axis_names=batch_axis_names,
    )
    assert config.mesh_shape == mesh_shape
    assert config.mesh_axis_names == mesh_axis_names
    assert config.batch_axis_names == batch_axis_names


@pytest.fixture(params=[4, 8])
def set_xla_flags(request):
    device_count = request.param
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
    yield device_count
    del os.environ["XLA_FLAGS"]


def test_mesh_config_create_device_mesh():
    device_count = 4
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
    try:
        mesh_shape = (1, 4)
        config = MeshConfig(mesh_shape=mesh_shape, mesh_axis_names=("data", "model"))
        mesh = config.create_device_mesh()

        assert isinstance(mesh, jax.sharding.Mesh)
        assert dict(zip(config.mesh_axis_names, mesh_shape, strict=False)) == dict(
            mesh.shape
        )
        assert mesh.axis_names == ("data", "model")
        assert len(mesh.devices.flatten()) == device_count

        # Check if the mesh shape is correct for the given device count
        if isinstance(config.mesh_shape, HybridMeshShape):
            assert (
                config.mesh_shape.ici_mesh_shape[0]
                * config.mesh_shape.ici_mesh_shape[1]
                == device_count
            )
        else:
            assert config.mesh_shape[0] * config.mesh_shape[1] == device_count
    finally:
        del os.environ["XLA_FLAGS"]


def test_mesh_config_hosts_and_host_id():
    config = MeshConfig(mesh_shape=(2, 2), mesh_axis_names=("data", "model"))

    assert config.hosts == jax.process_count()
    assert config.host_id == jax.process_index()


@pytest.mark.parametrize(
    ("mesh_rules", "mesh_selector", "expected_shape", "device_count"),
    [
        ([("cpu", (2, 2))], "cpu", (2, 2), 4),
        ([("cpu", (4, 1))], "cpu", (4, 1), 4),
        ([("cpu", (1, 4))], "cpu", (1, 4), 4),
        ([("gpu", (2, 2)), ("cpu", (1, 4))], "cpu", (1, 4), 4),
        ([("gpu", (2, 2)), ("cpu", (1, 4))], "gpu", (2, 2), 4),
    ],
)
def test_mesh_config_with_rules(
    mesh_rules, mesh_selector, expected_shape, device_count, monkeypatch
):
    monkeypatch.setenv(
        "XLA_FLAGS", f"--xla_force_host_platform_device_count={device_count}"
    )
    config = MeshConfig(
        mesh_shape=(1, device_count),  # Default shape
        mesh_axis_names=("data", "model"),
        mesh_rules=mesh_rules,
    )

    # Override the mesh_shape based on the selected rule
    for rule, shape in mesh_rules:
        if re.match(rule, mesh_selector):
            config.mesh_shape = shape
            break

    mesh = config.create_device_mesh()
    assert dict(zip(config.mesh_axis_names, expected_shape, strict=False)) == dict(
        mesh.shape
    )
    assert len(mesh.devices.flatten()) == device_count

    # Verify that all devices are of the expected type (CPU in this case)
    assert all(device.platform.lower() == "cpu" for device in mesh.devices.flatten())
