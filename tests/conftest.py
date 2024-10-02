# conftest.py

import importlib
import os

import pytest


@pytest.fixture(
    scope="session",
    params=[2, 4],
)
def xla_device_count(request):
    device_count = request.param
    os.environ["XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT"] = str(device_count)

    # Reload JAX to apply the new environment variable
    try:
        importlib.import_module("jax")
        importlib.import_module("jaxlib")
    except ImportError as err:
        raise ImportError("JAX and JAXLIB must be installed to run tests.") from err

    import jax

    devices = jax.devices()
    yield devices

    # Cleanup if necessary
    del os.environ["XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT"]


# Alternatively, without lambda for 'params'
@pytest.fixture(scope="session")
def simulated_xla_devices(request):
    device_count = request.param
    os.environ["XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT"] = str(device_count)

    # Reload JAX to apply the new environment variable
    try:
        importlib.import_module("jax")
        importlib.import_module("jaxlib")
    except ImportError as err:
        raise ImportError("JAX and JAXLIB must be installed to run tests.") from err

    import jax

    devices = jax.devices()
    yield devices

    # Cleanup if necessary
    del os.environ["XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT"]
