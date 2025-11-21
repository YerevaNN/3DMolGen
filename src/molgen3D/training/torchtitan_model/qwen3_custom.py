from __future__ import annotations

from dataclasses import replace

from torchtitan.models import qwen3 as qwen3_module
from torchtitan.protocols.train_spec import register_train_spec

from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_molgen_dataloader,
    build_molgen_validator,
)


_BASE_SPEC = qwen3_module.get_train_spec()

register_train_spec(
    "molgen_qwen3",
    replace(
        _BASE_SPEC,
        build_dataloader_fn=build_molgen_dataloader,
        build_validator_fn=build_molgen_validator,
    ),
)

