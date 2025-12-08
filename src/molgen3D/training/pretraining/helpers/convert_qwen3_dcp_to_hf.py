from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import shutil

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
import tyro
from transformers import Qwen3Config, Qwen3ForCausalLM

from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
# from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Hard-coded Qwen3 vocab layout for your setup
# ---------------------------------------------------------------------------

BASE_VOCAB: int = 151_669
PADDED_VOCAB: int = 151_936
NUM_NEW_TOKENS: int = 4

EXTRA_START: int = BASE_VOCAB
EXTRA_END: int = BASE_VOCAB + NUM_NEW_TOKENS  # 151673

EMBED_WEIGHT_KEYS: Tuple[str, ...] = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "model.tok_embeddings.weight",
    "tok_embeddings.weight",
    "model.input_embeddings.weight",
    "input_embeddings.weight",
)

HEAD_WEIGHT_KEYS: Tuple[str, ...] = (
    "lm_head.weight",
    "model.lm_head.weight",
    "output.weight",
    "model.output.weight",
)

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------

def _fmt(details: Dict[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in details.items())


def _find_tensor_key_from_metadata(
    tensor_metadata: Dict[str, dcp.TensorStorageMetadata],
    candidates: Tuple[str, ...],
) -> Optional[str]:
    """Find a tensor name in DCP metadata, exact match first, then suffix match."""
    for k in candidates:
        if k in tensor_metadata:
            return k
    suffixes = tuple(k.split(".", 1)[-1] for k in candidates)
    for name in tensor_metadata.keys():
        for suf in suffixes:
            if name.endswith(suf):
                return name
    return None


def _load_full_dcp_state(step_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load all tensors from a TorchTitan DCP step directory into a flat CPU state_dict.
    """
    if not step_dir.exists():
        raise FileNotFoundError(f"DCP step directory does not exist: {step_dir}")

    has_distcp = any(step_dir.glob("*.distcp"))
    if not has_distcp:
        raise FileNotFoundError(
            f"No .distcp shards found under {step_dir}; not a valid DCP step directory."
        )

    reader = dcp.FileSystemReader(str(step_dir))
    try:
        metadata = reader.read_metadata()
    except Exception as exc:
        raise RuntimeError(f"Failed to read DCP metadata: {exc}") from exc
    tmeta = metadata.state_dict_metadata  # type: ignore[assignment]
    if not tmeta:
        raise RuntimeError("DCP metadata is empty or corrupted; no tensors described.")

    state_dict: Dict[str, torch.Tensor] = {}
    skipped: list[tuple[str, str]] = []
    for name, tm in tmeta.items():
        size = getattr(tm, "size", None)
        properties = getattr(tm, "properties", None)
        dtype = getattr(properties, "dtype", None) if properties is not None else None
        if size is None or dtype is None:
            skipped.append((name, tm.__class__.__name__))
            continue
        try:
            state_dict[name] = torch.empty(size, dtype=dtype, device="cpu")
        except Exception as exc:
            skipped.append((f"{name} (error: {exc})", tm.__class__.__name__))

    dcp.load(state_dict, storage_reader=reader)
    print(f"[INFO] Loaded {len(state_dict)} tensors from {step_dir.name}")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} non-tensor entries (e.g. {skipped[0][0]})")
    return state_dict


def _resolve_embed_and_head(
    state_dict: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """Pick the embedding and LM head tensors from a flat Titan state dict."""
    keys = list(state_dict.keys())

    def _resolve(candidates: Tuple[str, ...]) -> Tuple[torch.Tensor, str]:
        # exact first
        for k in candidates:
            if k in state_dict:
                return state_dict[k], k
        # suffix fallback
        suffixes = tuple(k.split(".", 1)[-1] for k in candidates)
        for name in keys:
            for suf in suffixes:
                if name.endswith(suf):
                    return state_dict[name], name
        raise RuntimeError(f"Could not resolve tensor for candidates: {candidates}")

    embed, embed_key = _resolve(EMBED_WEIGHT_KEYS)
    head, head_key = _resolve(HEAD_WEIGHT_KEYS)
    return embed, head, embed_key, head_key


_CONFIG_FILENAMES = ("config.json", "job_config.json")


def _load_run_config(step_dir: Path) -> Dict:
    """
    Locate the saved config JSON near the checkpoint directory.
    Prefer config.json in the checkpoint root, fall back to job_config.json.
    """
    search_roots = []
    current = step_dir
    for _ in range(3):
        if current is None:
            break
        search_roots.append(current if current.is_dir() else current.parent)
        current = search_roots[-1].parent

    for root in search_roots:
        for filename in _CONFIG_FILENAMES:
            cfg_path = root / filename
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

    raise FileNotFoundError(
        f"Unable to locate config.json/job_config.json for checkpoint {step_dir}. "
        "Ensure training saved the config next to the checkpoint directory."
    )


def _collect_run_metadata(job_cfg: Dict) -> Dict[str, Optional[object]]:
    model_cfg = job_cfg.get("model", {})
    data_cfg = job_cfg.get("molgen_data", {})
    training_cfg = job_cfg.get("training", {})

    base_vocab = int(model_cfg.get("tokenizer_base_vocab_size", BASE_VOCAB))
    added_tokens = int(model_cfg.get("tokenizer_added_tokens", NUM_NEW_TOKENS))
    padded_vocab = int(model_cfg.get("padded_vocab_size", PADDED_VOCAB))
    seq_len = int(training_cfg.get("seq_len", 0))

    tokenizer_path = (
        model_cfg.get("tokenizer_override")
        or data_cfg.get("tokenizer_override")
        or model_cfg.get("hf_assets_path")
    )

    return {
        "base_vocab": base_vocab,
        "new_tokens": added_tokens,
        "padded_vocab": padded_vocab,
        "seq_len": seq_len,
        "tokenizer_path": tokenizer_path,
    }


def _resolve_expected_dtype(job_cfg: Dict) -> Tuple[str, Optional[torch.dtype]]:
    training_cfg = job_cfg.get("training", {})
    dtype_str = str(training_cfg.get("dtype", "float32")).lower()
    dtype = _DTYPE_MAP.get(dtype_str)
    return dtype_str, dtype


def _dtype_rank(dt: torch.dtype) -> int:
    """Order floating dtypes by fidelity for safe upcasting."""
    if dt == torch.float16:
        return 0
    if dt == torch.bfloat16:
        return 1
    if dt == torch.float32:
        return 2
    # Default to highest rank for unknown/extended precisions to avoid downcast.
    return 3


def _harmonize_state_dict_dtype(
    state_dict: Dict[str, torch.Tensor],
    expected_dtype: Optional[torch.dtype],
) -> torch.dtype:
    """
    Harmonize floating tensor dtypes by only upcasting to the highest precision seen
    (or the expected dtype if it is higher). Never downcasts to avoid fidelity loss.
    """
    float_dtypes = {t.dtype for t in state_dict.values() if t.is_floating_point()}
    if not float_dtypes:
        raise ValueError("Checkpoint contains no floating-point tensors to normalize.")

    target_dtype = max(
        float_dtypes.union({expected_dtype} if expected_dtype is not None else set()),
        key=_dtype_rank,
    )

    for name, tensor in list(state_dict.items()):
        if tensor.is_floating_point() and tensor.dtype != target_dtype:
            state_dict[name] = tensor.to(target_dtype)

    remaining = {tensor.dtype for tensor in state_dict.values() if tensor.is_floating_point()}
    if len(remaining) != 1:
        raise ValueError(f"Inconsistent tensor dtypes after harmonization: {remaining}")

    if expected_dtype is not None and target_dtype != expected_dtype:
        print(
            f"[WARN] Checkpoint dtype ({target_dtype}) differs from training config ({expected_dtype}); "
            "using higher-precision target to avoid downcasting."
        )
    return target_dtype


# ---------------------------------------------------------------------------
# Fidelity checks
# ---------------------------------------------------------------------------

def _sanity_check_qwen3_vocab_and_tie(
    titan_embed: torch.Tensor,
    titan_head: torch.Tensor,
    *,
    expected_base_vocab: int,
    expected_extra_tokens: int,
    expected_padded_vocab: int,
) -> None:
    """Checks that Titan embeddings/head look like your intended Qwen3 layout."""
    emb_vocab, emb_dim = titan_embed.shape
    head_vocab, head_dim = titan_head.shape

    if emb_vocab != expected_padded_vocab:
        raise ValueError(
            f"Embedding vocab {emb_vocab} != expected PADDED_VOCAB {expected_padded_vocab}"
        )
    if emb_vocab != head_vocab or emb_dim != head_dim:
        raise ValueError(
            "Embedding/head shape mismatch: "
            f"embed={tuple(titan_embed.shape)}, head={tuple(titan_head.shape)}"
        )

    tied = torch.equal(titan_embed, titan_head)
    if not tied:
        raise ValueError("Titan LM head is not tied to embedding weights (bitwise diff).")

    if not torch.isfinite(titan_embed).all():
        raise ValueError("Titan embedding contains non-finite values (NaN/Inf).")

    # Check extra rows
    extra_start = expected_base_vocab
    extra_end = expected_base_vocab + expected_extra_tokens
    extra_slice = titan_embed[extra_start:extra_end]
    if extra_slice.numel() == 0:
        raise ValueError(
            "Extra embedding slice is empty; NUM_NEW_TOKENS or BASE_VOCAB is wrong."
        )
    if not torch.isfinite(extra_slice).all():
        raise ValueError("Extra embedding rows contain non-finite values.")

    num_nonzero = torch.count_nonzero(extra_slice).item()
    if num_nonzero == 0:
        print("[WARN] All extra embedding rows are zero; they may not have been properly initialized/updated.")

    print(
        f"[INFO] Verified Titan embeddings: vocab={emb_vocab}, "
        f"extra_tokens={expected_extra_tokens}, nonzero_elements={num_nonzero}"
    )
    print("[SUCCESS] Titan checkpoint verification passed")


def _build_hf_model_from_base_assets(
    hf_assets_path: Path,
    *,
    expected_padded_vocab: int,
    expected_seq_len: int,
    target_dtype: torch.dtype,
) -> Qwen3ForCausalLM:
    """
    Build an HF Qwen3 model on CPU using the original base assets,
    but with vocab_size forced to PADDED_VOCAB.
    """
    if not hf_assets_path.exists():
        raise FileNotFoundError(f"HF assets path does not exist: {hf_assets_path}")

    config = Qwen3Config.from_pretrained(str(hf_assets_path))
    config.vocab_size = expected_padded_vocab
    config.dtype = target_dtype
    if expected_seq_len > 0:
        if hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = expected_seq_len
        if hasattr(config, "seq_length"):
            config.seq_length = expected_seq_len
        if hasattr(config, "model_max_length"):
            config.model_max_length = expected_seq_len

    hf_model = Qwen3ForCausalLM(config)
    hf_model.to(device="cpu", dtype=target_dtype)
    hf_model.eval()
    return hf_model


def _map_titan_to_hf_state(
    titan_state: Dict[str, torch.Tensor],
    hf_model: Qwen3ForCausalLM,
    hf_assets_path: Path,
) -> Dict[str, torch.Tensor]:
    """
    Use Torchtitan's Qwen3StateDictAdapter to convert Titan layout -> HF layout.
    """
    adapter = Qwen3StateDictAdapter(hf_model.config, str(hf_assets_path))  # tweak if your ctor differs

    if not hasattr(adapter, "to_hf"):
        raise RuntimeError(
            "Qwen3StateDictAdapter does not provide to_hf(...); "
            "add a Titan->HF mapping helper or upgrade Torchtitan."
        )

    hf_state_dict = adapter.to_hf(titan_state)  # type: ignore[attr-defined]
    return hf_state_dict


def _sanity_compare_titan_and_hf_embeddings(
    titan_state: Dict[str, torch.Tensor],
    hf_model: Qwen3ForCausalLM,
) -> None:
    """Check that HF embedding & LM head match Titan's, bitwise (or extremely close)."""
    titan_embed, titan_head, _, _ = _resolve_embed_and_head(titan_state)

    # Run Titan verification first
    hf_embed = hf_model.model.embed_tokens.weight.detach().cpu()
    hf_head = hf_model.lm_head.weight.detach().cpu()

    if hf_embed.shape != titan_embed.shape:
        raise ValueError(
            f"HF embed shape {tuple(hf_embed.shape)} != Titan embed {tuple(titan_embed.shape)}"
        )

    if not torch.equal(hf_embed, hf_head):
        raise ValueError("HF LM head is not tied to HF embed_tokens.weight bitwise.")

    if not torch.equal(titan_embed, hf_embed):
        max_diff = (titan_embed - hf_embed).abs().max().item()
        raise ValueError(
            f"HF embedding tensor differs from Titan embedding (max abs diff={max_diff:.3e})."
        )

    print("[SUCCESS] HF and Titan embeddings are identical")


def _sanity_compare_full_state(
    hf_state_dict: Dict[str, torch.Tensor],
    hf_model: Qwen3ForCausalLM,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """
    Ensure every tensor loaded into the HF model matches the converted HF state dict.
    This guards against silent dtype truncation or shape mismatches across all layers.
    """
    model_state = hf_model.state_dict()
    mismatches: list[str] = []
    extra_model_keys = []
    for key, src_tensor in hf_state_dict.items():
        if key not in model_state:
            raise ValueError(f"HF model missing tensor key after load: {key}")
        tgt_tensor = model_state[key]
        if src_tensor.shape != tgt_tensor.shape:
            raise ValueError(f"Shape mismatch for {key}: src={src_tensor.shape}, tgt={tgt_tensor.shape}")
        if src_tensor.dtype != tgt_tensor.dtype:
            raise ValueError(f"Dtype mismatch for {key}: src={src_tensor.dtype}, tgt={tgt_tensor.dtype}")
        if src_tensor.is_floating_point():
            if not torch.allclose(src_tensor, tgt_tensor, atol=atol, rtol=rtol):
                max_diff = (src_tensor - tgt_tensor).abs().max().item()
                mismatches.append(f"{key} (max_diff={max_diff:.3e})")
        else:
            if not torch.equal(src_tensor, tgt_tensor):
                mismatches.append(f"{key} (non-floating tensor differs)")

    extra_model_keys = [k for k in model_state.keys() if k not in hf_state_dict]
    if extra_model_keys:
        raise ValueError(f"HF model has unexpected keys after load_state_dict: {extra_model_keys[:5]}")

    if mismatches:
        sample = mismatches[:3]
        raise ValueError(
            f"{len(mismatches)} tensor(s) differ after load_state_dict, e.g. {sample}"
        )
    print(f"[SUCCESS] Verified {len(hf_state_dict)} tensors match after load_state_dict")


def _validate_config_matches_job_cfg(
    config: Qwen3Config,
    job_cfg: Dict,
) -> None:
    """Ensure the HF config matches the recorded training config (source of truth)."""
    model_cfg = job_cfg.get("model", {})

    def _expect(name: str, cfg_val: Optional[int], actual: Optional[int]) -> None:
        if cfg_val is None or actual is None:
            return
        if int(cfg_val) != int(actual):
            raise ValueError(f"Config mismatch for {name}: json={cfg_val}, model={actual}")

    _expect("hidden_size", model_cfg.get("hidden_size"), getattr(config, "hidden_size", None))
    _expect("intermediate_size", model_cfg.get("intermediate_size"), getattr(config, "intermediate_size", None))
    _expect("num_hidden_layers", model_cfg.get("num_hidden_layers"), getattr(config, "num_hidden_layers", None))
    _expect("num_attention_heads", model_cfg.get("num_attention_heads"), getattr(config, "num_attention_heads", None))

    # Rope / positional settings
    json_rope_theta = model_cfg.get("rope_theta")
    cfg_rope_theta = getattr(config, "rope_theta", None) if hasattr(config, "rope_theta") else None
    if json_rope_theta is not None and cfg_rope_theta is not None and float(json_rope_theta) != float(cfg_rope_theta):
        raise ValueError(f"Config mismatch for rope_theta: json={json_rope_theta}, model={cfg_rope_theta}")

    json_rope_scaling = model_cfg.get("rope_scaling")
    cfg_rope_scaling = getattr(config, "rope_scaling", None) if hasattr(config, "rope_scaling") else None
    if json_rope_scaling is not None and cfg_rope_scaling is not None and json_rope_scaling != cfg_rope_scaling:
        raise ValueError(f"Config mismatch for rope_scaling: json={json_rope_scaling}, model={cfg_rope_scaling}")

    print("[INFO] Validated HF config matches job_config.json (structure & rope settings)")


def _tiny_forward_fidelity_check(
    titan_state: Dict[str, torch.Tensor],
    hf_model: Qwen3ForCausalLM,
    vocab_limit: int = 1024,
    seq_len: int = 8,
    batch_size: int = 2,
) -> None:
    """
    Cheap CPU-only forward fidelity check on the embeddings+LM head.
    """
    titan_embed, titan_head, _, _ = _resolve_embed_and_head(titan_state)
    emb_vocab, emb_dim = titan_embed.shape
    vocab_limit = min(vocab_limit, emb_vocab)

    # Build Titan layers
    titan_emb_layer = nn.Embedding(
        emb_vocab,
        emb_dim,
        _weight=titan_embed.to("cpu"),
    )
    titan_head_layer = nn.Linear(
        emb_dim,
        emb_vocab,
        bias=False,
        dtype=titan_head.dtype,
        device="cpu",
    )
    with torch.no_grad():
        titan_head_layer.weight.copy_(titan_head)

    # Get HF layers
    hf_embed = hf_model.model.embed_tokens
    hf_head = hf_model.lm_head

    # Generate test input
    input_ids = torch.randint(
        low=0,
        high=vocab_limit,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device="cpu",
    )

    # Run forward passes
    with torch.no_grad():
        titan_hidden = titan_emb_layer(input_ids)     # [B, T, D]
        titan_logits = titan_head_layer(titan_hidden) # [B, T, V]

        hf_hidden = hf_embed(input_ids)
        hf_logits = hf_head(hf_hidden)

    if titan_logits.shape != hf_logits.shape:
        raise ValueError(
            f"Titan/HF logits shapes differ: Titan={tuple(titan_logits.shape)}, "
            f"HF={tuple(hf_logits.shape)}"
        )

    max_diff = (titan_logits - hf_logits).abs().max().item()
    if max_diff > 1e-6:
        raise ValueError(
            f"Forward logits mismatch between Titan and HF (max abs diff={max_diff:.3e})."
        )

    print(f"[INFO] Forward fidelity test: batch_size={batch_size}, seq_len={seq_len}, vocab_limit={vocab_limit}, max_diff={max_diff:.2e}")
    print("[SUCCESS] Forward fidelity check passed")


def _full_model_forward_check(
    hf_model: Qwen3ForCausalLM,
    *,
    seq_len: int = 16,
    batch_size: int = 2,
    vocab_limit: int = 2048,
) -> None:
    """
    Run a lightweight forward through the entire HF model to ensure all layers
    produce finite logits. This is a structural fidelity check, not a perf test.
    """
    config_vocab = hf_model.config.vocab_size
    vocab_limit = min(vocab_limit, config_vocab)
    hf_model.eval()
    input_ids = torch.randint(
        low=0,
        high=vocab_limit,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device="cpu",
    )
    with torch.no_grad():
        outputs = hf_model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits
    if not torch.isfinite(logits).all():
        raise ValueError("Full-model forward produced non-finite logits.")
    max_abs = logits.abs().max().item()
    print(
        f"[INFO] Full-model forward check: batch_size={batch_size}, seq_len={seq_len}, "
        f"vocab_limit={vocab_limit}, max_abs_logit={max_abs:.2f}"
    )
    print("[SUCCESS] Full-model forward check passed")


def _copy_tokenizer_assets(
    tokenizer_path: Optional[str],
    out_dir: Path,
) -> None:
    if not tokenizer_path:
        print("[WARN] No tokenizer path recorded in job_config.json; skipping tokenizer export.")
        return

    src = Path(tokenizer_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Tokenizer path recorded in job_config.json does not exist: {src}"
        )

    if src.is_file():
        target = out_dir / src.name
        if not target.exists():
            shutil.copy(src, target)
        return

    copied = []
    for item in src.iterdir():
        if not item.is_file():
            continue
        target = out_dir / item.name
        if target.exists():
            continue
        shutil.copy(item, target)
        copied.append(item.name)

    if copied:
        print(f"[INFO] Copied tokenizer artifacts: {', '.join(sorted(copied))}")
    else:
        print("[WARN] No tokenizer files copied (all targets existed already).")


def _format_exception(exc: Exception, limit: int = 512) -> str:
    msg = str(exc)
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Public helper: export DCP step -> HF checkpoint
# ---------------------------------------------------------------------------

def export_qwen3_dcp_step_to_hf(
    step_dir: Path | str,
    hf_assets_path: Path | str,
    *,
    expected_specs: Dict[str, Optional[object]],
    expected_dtype: Optional[torch.dtype],
    job_cfg: Optional[Dict] = None,
    tokenizer=None,
    out_dir: Path | str | None = None,
) -> Path:
    """
    Convert a TorchTitan DCP step directory into an HF Qwen3 checkpoint.
    """
    step_dir = Path(step_dir)
    if step_dir.is_file() and step_dir.name.endswith(".distcp"):
        step_dir = step_dir.parent

    if not step_dir.exists():
        raise FileNotFoundError(f"Step directory does not exist: {step_dir}")

    hf_assets_path = Path(hf_assets_path)

    if out_dir is None:
        out_dir = step_dir.with_name(step_dir.name + "-hf")
    out_dir = Path(out_dir)
    if out_dir.exists():
        print(f"[INFO] Removing existing HF checkpoint at {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting checkpoint {step_dir.name} from {step_dir.parent.name}")

    # Load and verify Titan checkpoint
    print("[INFO] Loading Titan checkpoint...")
    titan_state = _load_full_dcp_state(step_dir)
    target_dtype = _harmonize_state_dict_dtype(titan_state, expected_dtype)
    print(f"[INFO] Harmonized Titan checkpoint dtype to {target_dtype}")
    titan_embed, titan_head, _, _ = _resolve_embed_and_head(titan_state)
    print(f"[INFO] Titan model: {titan_embed.shape[0]} vocab, {titan_embed.shape[1]} hidden dim")

    if titan_embed.shape[0] != PADDED_VOCAB:
        raise ValueError(
            f"Embedding vocab {titan_embed.shape[0]} != expected PADDED_VOCAB {PADDED_VOCAB}"
        )
    if titan_embed.shape[0] < BASE_VOCAB + NUM_NEW_TOKENS:
        raise ValueError(
            "Embedding rows are insufficient for base + new tokens; invariants broken."
        )
    if titan_head.shape != titan_embed.shape:
        raise ValueError(
            f"LM head shape {tuple(titan_head.shape)} != embed shape {tuple(titan_embed.shape)}"
        )

    base_vocab = int(expected_specs["base_vocab"])
    extra_tokens = int(expected_specs["new_tokens"])
    padded_vocab = int(expected_specs["padded_vocab"])
    seq_len = int(expected_specs.get("seq_len") or 0)

    _sanity_check_qwen3_vocab_and_tie(
        titan_embed,
        titan_head,
        expected_base_vocab=base_vocab,
        expected_extra_tokens=extra_tokens,
        expected_padded_vocab=padded_vocab,
    )

    # Build and convert to HF
    print("[INFO] Building HF model and converting state dict...")
    hf_model = _build_hf_model_from_base_assets(
        hf_assets_path,
        expected_padded_vocab=padded_vocab,
        expected_seq_len=seq_len,
        target_dtype=target_dtype,
    )
    if job_cfg is not None:
        _validate_config_matches_job_cfg(hf_model.config, job_cfg)
    hf_state_dict = _map_titan_to_hf_state(titan_state, hf_model, hf_assets_path)

    missing, unexpected = hf_model.load_state_dict(hf_state_dict, strict=True)
    if missing or unexpected:
        raise ValueError(f"HF load_state_dict issues; missing={missing}, unexpected={unexpected}")

    # Enforce weight tying after load in case adapter didn't tie
    hf_model.lm_head.weight = hf_model.model.embed_tokens.weight
    if hf_model.lm_head.weight.data_ptr() != hf_model.model.embed_tokens.weight.data_ptr():
        raise ValueError("HF LM head is not tied to embedding weights after load.")

    # Verify conversion
    print("[INFO] Verifying conversion quality...")
    _sanity_compare_full_state(hf_state_dict, hf_model)
    _sanity_compare_titan_and_hf_embeddings(titan_state, hf_model)
    _tiny_forward_fidelity_check(titan_state, hf_model)
    _full_model_forward_check(hf_model)

    # Save result
    print(f"[INFO] Saving HF checkpoint to {out_dir}")
    hf_model.save_pretrained(out_dir, safe_serialization=True)
    print(f"[SUCCESS] Converted {step_dir.name} ({len(titan_state)} tensors, {titan_embed.shape[0]} vocab) -> {out_dir.name}")
    print(f"[INFO] Verified: embeddings tied, finite weights, forward fidelity")
    return out_dir


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

@dataclass
class DcpToHfConfig:
    """
    Convert TorchTitan Qwen3 DCP checkpoints to HF format.

    dcp_path:
        Either a single step directory (…/step-200) or a run root directory
        that contains multiple step-* subdirectories. If you pass a specific
        .distcp file, we will automatically strip to its parent step-XXX dir.

    dry_run:
        If True, only print what would be exported without writing any HF
        checkpoints.
    """
    dcp_path: str
    dry_run: bool = False


def _find_step_dirs(root: Path) -> List[Path]:
    """
    If root is a step-* directory -> return [root].
    Otherwise -> find all direct subdirectories matching step-*.
    """
    if root.is_file():
        if root.name.endswith(".distcp"):
            root = root.parent
        else:
            raise RuntimeError(
                f"Unsupported file path: {root}. "
                "Pass a DCP directory or a *.distcp shard."
            )

    root = root.resolve()
    if root.name.startswith("step-"):
        if root.exists():
            return [root]
        raise FileNotFoundError(f"{root} does not exist")

    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    step_dirs = [
        p for p in sorted(root.iterdir())
        if p.is_dir() and p.name.startswith("step-")
    ]
    if not step_dirs:
        raise RuntimeError(
            f"No step-* directories found under {root}. "
            "Pass a specific step-XXX directory or a run root containing steps."
        )
    return step_dirs


def main(cfg: DcpToHfConfig) -> None:
    root = Path(cfg.dcp_path)

    step_dirs = _find_step_dirs(root)

    if cfg.dry_run:
        print(f"[INFO] DRY RUN: Would process {len(step_dirs)} checkpoint(s)")
        for step_dir in step_dirs[:3]:  # Show first few
            print(f"  - {step_dir.name}")
        if len(step_dirs) > 3:
            print(f"  ... and {len(step_dirs) - 3} more")
        return

    for i, step_dir in enumerate(step_dirs, 1):
        if len(step_dirs) > 1:
            print(f"[INFO] Processing {i}/{len(step_dirs)}: {step_dir.name}")
        try:
            job_cfg = _load_run_config(step_dir)
            run_metadata = _collect_run_metadata(job_cfg)
            dtype_str, dtype = _resolve_expected_dtype(job_cfg)
            hf_assets = job_cfg.get("model", {}).get("hf_assets_path")
            if not hf_assets:
                raise ValueError(
                    "job_config.json does not contain model.hf_assets_path; "
                    "re-run training with normalized paths."
                )
            hf_base = Path(hf_assets)
            if not hf_base.exists():
                raise FileNotFoundError(
                    f"HF assets path recorded in config does not exist: {hf_base}"
                )
            out_dir = export_qwen3_dcp_step_to_hf(
                step_dir,
                hf_base,
                out_dir=None,
                expected_specs=run_metadata,
                expected_dtype=dtype,
                job_cfg=job_cfg,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to convert {step_dir.name}: {_format_exception(exc)}")


if __name__ == "__main__":
    cfg = tyro.cli(DcpToHfConfig)
    main(cfg)
