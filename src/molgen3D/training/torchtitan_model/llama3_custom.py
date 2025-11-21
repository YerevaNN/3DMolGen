from __future__ import annotations

from dataclasses import replace

from torchtitan.models import llama3 as llama3_module
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.protocols.train_spec import register_train_spec

from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_molgen_dataloader,
)

__all__ = ["register_llama3_spec"]


def _ensure_1b_args(spec):
    if "1B" not in spec.model_args:
        spec.model_args["1B"] = TransformerModelArgs(
            dim=2048,
            n_layers=24,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=128256,
            multiple_of=256,
            ffn_dim_multiplier=1.3,
            rope_theta=500000,
            max_seq_len=2048,
        )


def register_llama3_spec() -> None:
    spec = llama3_module.get_train_spec()
    _ensure_1b_args(spec)
    register_train_spec(
        "molgen_llama3",
        replace(spec, build_dataloader_fn=build_molgen_dataloader),
    )


register_llama3_spec()

