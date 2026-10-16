"""The BindGPT 108M GPT-NeoX language model, instantiated from scratch.

The architecture comes from the public BindGPT config -- ``gpt_neox``, 15
layers, hidden 768, 12 heads, intermediate 3,072, rotary, 2,048 context, untied
embeddings -- read from the pinned local copy. Only the config is used; no
pretrained weight is ever loaded.

Both arms share the union vocabulary, so their embedding and output shapes are
identical and one seed gives bitwise-identical initial parameters. That makes
the transformer-body initialization matched by construction rather than by
careful copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from molgen3D.experiments.bindgpt_fsq import config as experiment_config
from molgen3D.experiments.bindgpt_fsq import public_tokenizer


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter accounting, split so vocabulary effects stay visible."""

    total: int
    transformer_body: int
    input_embedding: int
    output_embedding: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "transformer_body": self.transformer_body,
            "input_embedding": self.input_embedding,
            "output_embedding": self.output_embedding,
        }


def load_architecture(directory: Optional[Path] = None):
    """Return the public GPT-NeoX config, verified against the pinned fields."""
    from transformers import AutoConfig

    directory = Path(directory or experiment_config.PUBLIC_TOKENIZER_DIR)
    files = public_tokenizer.PublicTokenizerFiles(directory)
    public_tokenizer.verify_architecture(files)
    architecture = AutoConfig.from_pretrained(str(directory), local_files_only=True)
    # The public card declares a FlashGPTNeoX wrapper class that is not part of
    # Transformers; the model type is plain gpt_neox, so drop the wrapper name
    # and let AutoModelForCausalLM resolve GPTNeoXForCausalLM.
    architecture.architectures = ["GPTNeoXForCausalLM"]
    return architecture


def build_model(
    vocab_size: int,
    seed: int,
    pad_token_id: int,
    eos_token_id: int,
    bos_token_id: int,
    dtype: torch.dtype = torch.float32,
    directory: Optional[Path] = None,
):
    """Instantiate the model from scratch under a fixed seed."""
    from transformers import AutoModelForCausalLM

    architecture = load_architecture(directory)
    architecture.vocab_size = int(vocab_size)
    architecture.pad_token_id = int(pad_token_id)
    architecture.eos_token_id = int(eos_token_id)
    architecture.bos_token_id = int(bos_token_id)
    architecture.use_cache = True
    architecture.torch_dtype = dtype

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_config(architecture)
    return model.to(dtype=dtype)


def count_parameters(model) -> ParameterCounts:
    """Split the parameter count into body and vocabulary-dependent parts."""
    total = sum(p.numel() for p in model.parameters())
    input_embedding = sum(
        p.numel() for p in model.get_input_embeddings().parameters()
    )
    output_embedding = sum(
        p.numel() for p in model.get_output_embeddings().parameters()
    )
    return ParameterCounts(
        total=total,
        transformer_body=total - input_embedding - output_embedding,
        input_embedding=input_embedding,
        output_embedding=output_embedding,
    )


def initial_parameter_checksum(model) -> str:
    """SHA-256 over the model's parameters, for cross-arm init comparison."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()
