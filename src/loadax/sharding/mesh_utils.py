import collections
import math
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from jax.experimental import mesh_utils

from loadax.logger import logger
from loadax.sharding.mesh_shape import HybridMeshShape, MeshShape

Devices = np.ndarray[tuple[int, ...], Any]


def create_hybrid_device_mesh(
    mesh_shape: HybridMeshShape,
    *,
    devices: Devices,
    process_is_granule: bool = False,
) -> Devices:
    """Extends the method to have an option to fall back to naive mesh.

    Reference:
    https://github.com/google/jax/blob/1189d61bc086fcfb548e73235a601ec46c3623c5/jax/experimental/mesh_utils.py#L324

    Args:
        mesh_shape: Shape of the logical mesh for both ICI and DCN.
            The ICI mesh corresponds to the faster/inner network, ordered by increasing
            network intensity, e.g. [data, fsdp, model] where model has the most network
            communicationrequirements.
            The DCN mesh corresponds to the slower/outer network in the same order as
            the ICI mesh. We expect the shapes to be fully specified, i.e., they should
            not contain -1 dims.
        devices: The devices to construct a mesh for.
        process_is_granule: If True, this function will treat processes as the units of
            the slower/outer network by looking for "process_index" attributes on
            devices. Otherwise it will treat slices as the units and look for
            "slice_index" attributes on devices.

    Raises:
        ValueError: If the number of granules to which the `devices` belong doesn't
            equal the product of `dcn_mesh_shape`, or if the number of devices
            belonging to any single granule does not equal the product of `mesh_shape`.

    Returns:
        A np.ndarray of JAX devices with `ici_mesh_shape * dcn_mesh_shape` as its shape
        that can be fed into jax.sharding.Mesh for hybrid parallelism.
    """
    attr = "process_index" if process_is_granule else "slice_index"
    assert hasattr(devices[0], attr)
    granule_dict = collections.defaultdict(list)
    for dev in devices:
        granule_dict[getattr(dev, attr)].append(dev)
    granules = [granule_dict[key] for key in sorted(granule_dict.keys())]
    if np.prod(mesh_shape.dcn_mesh_shape) != len(granules):
        raise ValueError(
            f"Number of slices/granules {len(granules)} must equal the product of "
            f"dcn_mesh_shape {mesh_shape.dcn_mesh_shape}"
        )
    per_granule_meshes = [
        build_standard_mesh(mesh_shape.ici_mesh_shape, devices=np.asarray(granule))
        for granule in granules
    ]
    granule_mesh = np.arange(len(granules)).reshape(mesh_shape.dcn_mesh_shape)
    blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(
        granule_mesh
    )
    device_mesh = np.block(blocks.tolist())
    return device_mesh


def create_device_mesh(
    mesh_shape: MeshShape | HybridMeshShape,
    *,
    devices: Devices | None = None,
) -> Devices:
    """Constructs a device mesh.

    If `mesh_shape` is specified as a `HybridMeshShape`, we use the `ici_mesh_shape` and
    `dcn_mesh_shape` directly to construct the mesh.

    If `mesh_shape` is specified as a `MeshShape`, we first determine whether we are
    running in a TPU or GPU environment.
        - If running in a TPU environment:
            - If multi-slice/granule, we split the first non-singleton axis of the
                configured meshshape across the slices.
        - If running in a GPU environment:
            - If multi-node, and the first non-singleton axis divides the number
                of processes (GPU-nodes/granules), we split the first axis
                across the processes.

    In all other cases we construct a standard mesh according to the
    configured mesh_shape.

    Args:
        mesh_shape: The desired logical mesh shape.
        devices: The devices that will be used to construct the mesh.
            If None, defaults to jax.devices().

    Returns:
        A numpy array containing the JAX devices with shape determined by the
        config mesh_shape.

    Raises:
        NotImplementedError: If not all devices have the same platform.
    """
    devices = np.asarray(devices if devices is not None else jax.devices())

    assert devices is not None, "devices must be provided."

    # Check if the devices are part of a multi-granule configuration.
    # <https://github.com/google/jax/blob/b81b79c1b0d2ec/jax/experimental/mesh_utils.py#L313>
    device_platform = devices[0].platform
    attr = "process_index" if device_platform != "tpu" else "slice_index"
    is_multi_granule_env = hasattr(devices[0], attr)
    if not all(el.platform == device_platform for el in devices):
        raise NotImplementedError(f"Not all devices had platform: {device_platform}.")

    num_granules = (
        max(getattr(el, attr) for el in devices.flatten()) + 1
        if is_multi_granule_env
        else 1
    )
    num_devices = len(devices)
    assert (
        num_devices % num_granules == 0
    ), "Number of devices should divide number of granules."
    num_devices_per_granule = num_devices // num_granules

    # Fallback to a standard mesh if on GPU with incompatible multi-granule mesh.
    if (
        device_platform == "gpu"
        and isinstance(mesh_shape, Sequence)  # MeshShape is an alias for Sequence[int]
        and mesh_shape[0] % num_granules != 0
    ):
        logger.warning(
            "Falling back to ICI-only mesh on GPU, performance may be reduced."
        )
        return build_standard_mesh(mesh_shape, devices=devices)

    # Canonicalize to HybridMeshShape. If DCN mesh is not specified,
    # break the first non-singleton device axis (the least communication intensive)
    # over the number of slices/granules. If all axes are singletons, this is
    # effectively a no-op, since this implies a single-granule environment.
    if isinstance(mesh_shape, Sequence):  # MeshShape is an alias for Sequence[int]
        mesh_shape = infer_mesh_shape(mesh_shape, num_devices=num_devices)
        for axis, dim in enumerate(mesh_shape):
            if dim % num_granules == 0:
                break
            elif dim != 1:
                raise ValueError(
                    f"First non-singleton mesh axis {axis} with value {dim} does not "
                    f"divide the number of slices/granules {num_granules}."
                )
        else:
            raise ValueError(
                f"At least one axis of {mesh_shape=} must divide {num_granules=}."
            )

        if num_granules > 1:
            logger.info("Building multi-slice/granule device mesh over axis %s.", axis)
        # Truncate intra-slice/granule mesh.
        mesh_shape = (*mesh_shape[:axis], dim // num_granules, *mesh_shape[axis + 1 :])
        logger.info("Inferred intra-slice/granule mesh shape: %s", mesh_shape)
        # Configure data center (inter-slice/granule) mesh.
        dcn_mesh_shape = (
            (1,) * axis + (num_granules,) + (1,) * len(mesh_shape[axis + 1 :])
        )
        logger.info("Inferred inter-slice/granule mesh shape: %s", dcn_mesh_shape)

        mesh_shape = HybridMeshShape(
            ici_mesh_shape=mesh_shape, dcn_mesh_shape=dcn_mesh_shape
        )
    else:
        # Infer -1 values in the mesh.
        mesh_shape = HybridMeshShape(
            ici_mesh_shape=infer_mesh_shape(
                mesh_shape.ici_mesh_shape, num_devices=num_devices_per_granule
            ),
            dcn_mesh_shape=infer_mesh_shape(
                mesh_shape.dcn_mesh_shape, num_devices=num_granules
            ),
        )
    logger.info("Using hybrid mesh shape: %s.", mesh_shape)

    # Check that we have the right number of devices.
    assert num_granules * num_devices_per_granule == len(devices)
    if np.prod(mesh_shape.dcn_mesh_shape) != num_granules:
        raise ValueError(
            f"Product of DCN mesh {mesh_shape.dcn_mesh_shape} does not "
            f"match {num_granules=}."
        )
    if np.prod(mesh_shape.ici_mesh_shape) != num_devices_per_granule:
        raise ValueError(
            f"Product of ICI mesh {mesh_shape.ici_mesh_shape} does not match "
            f"{num_devices_per_granule=}."
        )

    # Return a standard mesh if not a multi-granule env.
    if num_granules == 1:
        return build_standard_mesh(mesh_shape.ici_mesh_shape, devices=devices)

    return create_hybrid_device_mesh(
        mesh_shape,
        devices=devices,
        process_is_granule=attr == "process_index",
    )


def build_standard_mesh(mesh_shape: MeshShape, *, devices: Devices) -> Devices:
    """Build a standard mesh from a given mesh shape.

    Args:
        mesh_shape: The mesh shape to build the mesh from.
        devices: The devices to build the mesh from.

    Returns:
        A numpy array containing the JAX devices with shape determined by the
        config mesh_shape.
    """
    logger.info("Building device mesh.")
    mesh_shape = infer_mesh_shape(mesh_shape, num_devices=devices.size)
    try:
        # devices implements the Sequence protocol
        return mesh_utils.create_device_mesh(mesh_shape, devices=devices)  # type: ignore
    except NotImplementedError as e:
        logger.warning(
            "mesh_utils.create_device_mesh cannot handle shape %s: %s. "
            "Falling back to the naive mesh. Performance may be reduced.",
            mesh_shape,
            e,
        )
        return devices.reshape(mesh_shape)


def infer_mesh_shape(
    mesh_shape: MeshShape, *, num_devices: int | None = None
) -> MeshShape:
    """Infer value for -1 from len(jax.devices()) and other dims if -1 in mesh shape.

    Args:
        mesh_shape: The original MeshShape, which might have -1 in one axis.
        num_devices: The devices that will be used to construct the mesh.
            If None, defaults to len(jax.devices()).

    Returns:
        A new MeshShape with inferred value for -1.
    """
    if -1 not in mesh_shape:
        return mesh_shape

    if mesh_shape.count(-1) > 1:
        raise ValueError(f"Only one axis can be -1 in {mesh_shape=}.")

    # Handle the case with one -1.
    prod = math.prod(mesh_shape, start=-1)
    if num_devices is None:
        num_devices = len(jax.devices())
    if num_devices % prod != 0:
        raise ValueError(
            f"Unable to infer -1 in mesh shape {mesh_shape} as num_devices "
            f"{num_devices} is not a multiple of the product {prod} of mesh axes."
        )

    return tuple(x if x != -1 else num_devices // prod for x in mesh_shape)
