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


@pytest.mark.parametrize("simulated_xla_devices", [4], indirect=True)
def test_mesh_config_create_device_mesh(simulated_xla_devices):
    device_count = len(simulated_xla_devices)
    mesh_shape = (1, device_count)
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
            config.mesh_shape.ici_mesh_shape[0] * config.mesh_shape.ici_mesh_shape[1]
            == device_count
        )
    else:
        assert config.mesh_shape[0] * config.mesh_shape[1] == device_count


def test_mesh_config_hosts_and_host_id():
    config = MeshConfig(mesh_shape=(2, 2), mesh_axis_names=("data", "model"))

    assert config.hosts == jax.process_count()
    assert config.host_id == jax.process_index()


# TODO: walln - tests for mesh_rules if ther are going to stay in the API
