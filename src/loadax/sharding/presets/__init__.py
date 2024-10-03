"""Sharding configuration presets."""

from loadax.sharding.presets.ddp import (
    make_ddp_mesh_config as make_ddp_mesh_config,
)
from loadax.sharding.presets.fsdp import (
    make_fsdp_mesh_config as make_fsdp_mesh_config,
)
