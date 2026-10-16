"""Scratch training for one arm.

The recipe is BindGPT's GEOM supervised-fine-tuning recipe applied from random
initialization: AdamW, max lr 5e-4, 100 warmup steps, cosine decay, weight decay
1e-2, global-norm clipping at 1.0, bfloat16 compute, uniform token loss weight,
no SMILES randomization, no coordinate rotation.

The run refuses to start or continue when an integrity condition is violated:
mismatched paired example counts, a sequence over context, a non-finite loss, or
a moved FSQ parameter.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from molgen3D.experiments.bindgpt_fsq import dataset as data_module
from molgen3D.experiments.bindgpt_fsq import model as model_module
from molgen3D.experiments.bindgpt_fsq import vocabulary
from molgen3D.experiments.bindgpt_fsq.config import ExperimentConfig
from molgen3D.experiments.bindgpt_fsq.sequences import EOS, LIGAND, PAD, XYZ, Arm


class IntegrityError(RuntimeError):
    """Raised when a training integrity check fails; the run must stop."""


@dataclass
class RunState:
    """Cumulative counters logged every step."""

    step: int = 0
    examples: int = 0
    enabled_tokens: int = 0
    gpu_seconds: float = 0.0
    over_context: int = 0
    sequence_lengths: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed() -> tuple[int, int, int]:
    if not _is_distributed():
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cosine_with_warmup(step: int, warmup: int, total: int, peak: float, floor: float = 0.0) -> float:
    """BindGPT's schedule: linear warmup then cosine decay to ``floor``."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if total <= warmup:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    geometry_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return summed total loss, summed geometry-suffix loss and their counts.

    Every non-padding token carries weight 1.0. Sums (not means) are returned so
    that gradient accumulation across uneven micro-batches stays exact.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_geometry = geometry_mask[:, 1:]

    flat = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=data_module.LABEL_IGNORE,
        reduction="none",
    ).view_as(shift_labels)

    valid = shift_labels != data_module.LABEL_IGNORE
    total = (flat * valid).sum()
    geometry_valid = valid & shift_geometry
    geometry = (flat * geometry_valid).sum()
    return total, geometry, int(valid.sum().item()), int(geometry_valid.sum().item())


def train(
    run_config: ExperimentConfig,
    packed_dir: str | Path,
    valid_packed_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    validation_examples: int = 20_000,
    log_every: int = 10,
    validate_every: int = 250,
) -> Dict[str, object]:
    """Train one arm for the configured number of epochs."""
    rank, world_size, local_rank = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    settings = run_config.training
    arm = Arm(run_config.arm)

    union = vocabulary.build_vocabulary()
    table = union.token_to_id
    pad_id, xyz_id = table[PAD], table[XYZ]

    packed = data_module.PackedExamples(Path(packed_dir), arm.value)
    counterpart = Arm.FSQ if arm is Arm.COORD else Arm.COORD
    other = data_module.PackedExamples(Path(packed_dir), counterpart.value)
    if len(packed) != len(other):
        raise IntegrityError(
            f"paired example counts differ: {arm.value}={len(packed)}, "
            f"{counterpart.value}={len(other)}"
        )
    if int(packed.lengths.max()) > settings.max_sequence_length:
        raise IntegrityError(
            f"packed example of {int(packed.lengths.max())} tokens exceeds the "
            f"{settings.max_sequence_length}-token context; the build step must "
            f"exclude it rather than the trainer truncate it"
        )

    validation = (
        data_module.PackedExamples(Path(valid_packed_dir), arm.value)
        if valid_packed_dir
        else None
    )

    net = model_module.build_model(
        vocab_size=len(union),
        seed=settings.seed,
        pad_token_id=pad_id,
        eos_token_id=table[EOS],
        bos_token_id=table[LIGAND],
        dtype=torch.float32,
    ).to(device)
    counts = model_module.count_parameters(net)
    init_checksum = model_module.initial_parameter_checksum(net)

    wrapped = (
        DistributedDataParallel(net, device_ids=[local_rank] if device.type == "cuda" else None)
        if world_size > 1
        else net
    )

    decay, no_decay = [], []
    for name, parameter in net.named_parameters():
        if parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings.learning_rate,
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_eps,
    )

    steps_per_epoch = len(packed) // settings.global_example_batch_size
    total_steps = steps_per_epoch * settings.epochs
    if total_steps <= 0:
        raise IntegrityError(
            f"{len(packed)} examples cannot fill one global batch of "
            f"{settings.global_example_batch_size}"
        )
    checkpoint_steps = {
        max(1, int(round(fraction * total_steps))): f"{int(fraction * 100):03d}pct"
        for fraction in settings.checkpoint_fractions
    }

    output_dir = Path(output_dir or run_config.checkpoint_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.yaml").write_text(run_config.to_yaml())
        (output_dir / "model_parameters.json").write_text(
            json.dumps(
                {
                    **counts.as_dict(),
                    "union_vocab_size": len(union),
                    "init_parameter_sha256": init_checksum,
                    "seed": settings.seed,
                },
                indent=1,
            )
        )

    fsq_fingerprint_before = _fsq_fingerprint(run_config, rank)
    state = RunState()
    history: List[Dict[str, object]] = []
    autocast_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float32
    log_path = output_dir / "training_log.jsonl"
    log_handle = open(log_path, "a") if rank == 0 else None

    try:
        for epoch in range(settings.epochs):
            order = data_module.epoch_order(len(packed), settings.seed, epoch)
            batches = data_module.iter_global_batches(
                packed,
                order,
                settings.global_example_batch_size,
                settings.micro_batch_size,
                pad_id,
                xyz_id,
                rank=rank,
                world_size=world_size,
            )
            for local_step, micro_batches in batches:
                started = time.perf_counter()
                learning_rate = cosine_with_warmup(
                    state.step, settings.warmup_steps, total_steps, settings.learning_rate
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                optimizer.zero_grad(set_to_none=True)
                if not micro_batches:
                    raise IntegrityError(
                        f"rank {rank} received no examples at step {state.step}; "
                        f"lower the global example batch or the world size"
                    )
                # Uniform token weighting needs the *global* scored-token count
                # before the backward pass: DDP averages gradients over ranks,
                # so each rank divides by global_total / world_size and the
                # averaged gradient is exactly the global token mean.
                rank_scored = float(
                    sum(micro.predicted_tokens for micro in micro_batches)
                )
                scored = torch.tensor([rank_scored], device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.all_reduce(scored, op=dist.ReduceOp.SUM)
                normalizer = max(1.0, float(scored.item()) / world_size)
                totals = torch.zeros(4, device=device, dtype=torch.float64)
                for position, micro in enumerate(micro_batches):
                    inputs = micro.input_ids.to(device, non_blocking=True)
                    attention = micro.attention_mask.to(device, non_blocking=True)
                    labels = micro.labels.to(device, non_blocking=True)
                    geometry = micro.geometry_mask.to(device, non_blocking=True)

                    sync = position == len(micro_batches) - 1
                    context = (
                        wrapped.no_sync()
                        if world_size > 1 and not sync
                        else _nullcontext()
                    )
                    with context:
                        with torch.autocast(
                            device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda",
                        ):
                            logits = wrapped(
                                input_ids=inputs, attention_mask=attention
                            ).logits
                        loss_sum, geometry_sum, n_tokens, n_geometry = token_losses(
                            logits, labels, geometry
                        )
                        if not torch.isfinite(loss_sum):
                            raise IntegrityError(
                                f"non-finite loss at step {state.step}"
                            )
                        (loss_sum / normalizer).backward()

                    totals += torch.tensor(
                        [
                            loss_sum.detach().double(),
                            geometry_sum.detach().double(),
                            float(n_tokens),
                            float(n_geometry),
                        ],
                        device=device,
                        dtype=torch.float64,
                    )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), settings.max_grad_norm
                )
                optimizer.step()

                if world_size > 1:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                loss_sum, geometry_sum, n_tokens, n_geometry = totals.tolist()

                state.step += 1
                state.examples += settings.global_example_batch_size
                state.enabled_tokens += int(n_tokens)
                state.gpu_seconds += (time.perf_counter() - started) * world_size

                if rank == 0 and (state.step % log_every == 0 or state.step == 1):
                    record = {
                        "step": state.step,
                        "epoch": epoch,
                        "lr": learning_rate,
                        "loss": loss_sum / max(1.0, n_tokens),
                        "geometry_loss": geometry_sum / max(1.0, n_geometry),
                        "grad_norm": float(grad_norm),
                        "examples": state.examples,
                        "enabled_tokens": state.enabled_tokens,
                        "tokens_this_step": int(n_tokens),
                        "examples_per_second": settings.global_example_batch_size
                        / max(1e-9, time.perf_counter() - started),
                        "gpu_hours": state.gpu_seconds / 3600.0,
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                    }
                    history.append(record)
                    log_handle.write(json.dumps(record) + "\n")
                    log_handle.flush()

                if validation is not None and state.step % validate_every == 0:
                    metrics = evaluate(
                        wrapped, validation, settings, pad_id, xyz_id, device,
                        autocast_dtype, validation_examples, rank, world_size,
                    )
                    if rank == 0:
                        record = {"step": state.step, **metrics}
                        history.append(record)
                        log_handle.write(json.dumps(record) + "\n")
                        log_handle.flush()

                if state.step in checkpoint_steps and rank == 0:
                    _save(net, output_dir, checkpoint_steps[state.step], state, run_config)

    finally:
        if log_handle is not None:
            log_handle.close()

    fsq_fingerprint_after = _fsq_fingerprint(run_config, rank)
    if (
        fsq_fingerprint_before
        and fsq_fingerprint_after
        and fsq_fingerprint_before["parameter_sha256"]
        != fsq_fingerprint_after["parameter_sha256"]
    ):
        raise IntegrityError("frozen FSQ parameters changed during training")

    summary = {
        "run_id": run_config.run_id,
        "arm": arm.value,
        "seed": settings.seed,
        "examples": len(packed),
        "paired_examples": len(other),
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "state": state.as_dict(),
        "parameters": counts.as_dict(),
        "init_parameter_sha256": init_checksum,
        "sequence_length_histogram": {
            str(int(bucket)): int(count)
            for bucket, count in zip(
                *np.unique(packed.lengths // 64 * 64, return_counts=True)
            )
        },
        "sequences_near_context": int(
            (packed.lengths > 0.9 * settings.max_sequence_length).sum()
        ),
        "sequences_over_context": int(
            (packed.lengths > settings.max_sequence_length).sum()
        ),
        "fsq_fingerprint_before": fsq_fingerprint_before,
        "fsq_fingerprint_after": fsq_fingerprint_after,
    }
    if rank == 0:
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=1))
        _save(net, output_dir, "final", state, run_config)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


@torch.no_grad()
def evaluate(
    net,
    packed: data_module.PackedExamples,
    settings,
    pad_id: int,
    xyz_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    limit: int,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Total and geometry-suffix validation loss."""
    net.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for index, micro in enumerate(
        data_module.validation_batches(
            packed, settings.micro_batch_size, pad_id, xyz_id, limit=limit
        )
    ):
        if index % world_size != rank:
            continue
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            logits = net(
                input_ids=micro.input_ids.to(device),
                attention_mask=micro.attention_mask.to(device),
            ).logits
        loss_sum, geometry_sum, n_tokens, n_geometry = token_losses(
            logits, micro.labels.to(device), micro.geometry_mask.to(device)
        )
        totals += torch.tensor(
            [loss_sum.double(), geometry_sum.double(), float(n_tokens), float(n_geometry)],
            device=device,
            dtype=torch.float64,
        )
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    net.train()
    loss_sum, geometry_sum, n_tokens, n_geometry = totals.tolist()
    return {
        "validation_loss": loss_sum / max(1.0, n_tokens),
        "validation_geometry_loss": geometry_sum / max(1.0, n_geometry),
        "validation_tokens": int(n_tokens),
    }


def _save(net, output_dir: Path, tag: str, state: RunState, run_config: ExperimentConfig) -> Path:
    """Write an atomic checkpoint that a Hugging Face loader can read back."""
    destination = output_dir / f"step-{state.step:07d}-{tag}"
    destination.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(str(destination), safe_serialization=True)
    (destination / "run_state.json").write_text(
        json.dumps({"tag": tag, **state.as_dict(), "run_id": run_config.run_id}, indent=1)
    )
    return destination


def _fsq_fingerprint(run_config: ExperimentConfig, rank: int) -> Optional[Dict[str, object]]:
    """Fingerprint the frozen tokenizer, on rank 0 only."""
    if rank != 0:
        return None
    try:
        from molgen3D.experiments.bindgpt_fsq.fsq_frozen import FrozenFSQTokenizer

        tokenizer = FrozenFSQTokenizer(run_config.fsq_checkpoint, device="cpu")
        tokenizer.assert_no_gradients()
        return tokenizer.fingerprint().as_dict()
    except Exception as exc:  # noqa: BLE001 - recorded, never silently skipped
        return {"error": str(exc)}
