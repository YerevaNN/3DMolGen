from __future__ import annotations

import json
import random
import cloudpickle
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from loguru import logger

try:
    from torchtitan.components.dataloader import BaseDataLoader
    from torchtitan.components.validate import BaseValidator, Validator
    from torchtitan.tools.logging import logger as titan_logger
except Exception:
    class BaseDataLoader:  # type: ignore[too-many-ancestors]
        pass
    BaseValidator = None
    Validator = None
    titan_logger = None

from molgen3D.training.pretraining.config.custom_job_config import (
    JobConfig as MolGenJobConfig,
)
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.evaluation.utils import extract_between
from molgen3D.utils.utils import get_best_rmsd
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_dataloader,
    _resolve_special_token_id,
    _resolve_tokenizer_path,
)

NUM_NUMERICAL_VALIDATION_PROMPTS = 200
PRETOKENIZED_PROMPTS_PATH = Path("/auto/home/vover/3DMolGen/data/pretokenized_prompts.json")
VALIDATION_PICKLE_PATH = Path("/auto/home/vover/3DMolGen/data/valid_set.pickle")

def _is_primary_rank() -> bool:
    """
    Return True only for the primary logging rank to avoid duplicated work.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def _resolve_validation_path(job_config: MolGenJobConfig) -> str:
    validation_path = getattr(job_config.validation, "dataset_path", "")
    if not validation_path:
        raise ValueError(
            "MolGen validation requires a dataset_path. "
            "Set validation.dataset_path (e.g. molgen_data validation alias "
            "in paths.yaml)."
        )
    return validation_path

MolGenValidatorClass = None

if Validator is not None:
    class MolGenValidator(Validator):
        def __init__(
            self,
            job_config: MolGenJobConfig,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            parallel_dims,
            loss_fn,
            validation_context,
            maybe_enable_amp,
            metrics_processor,
            validation_dataloader: BaseDataLoader,
            pp_schedule=None,
            pp_has_first_stage=None,
            pp_has_last_stage=None,
        ):
            self.job_config = job_config
            self.tokenizer = tokenizer
            self.parallel_dims = parallel_dims
            self.loss_fn = loss_fn
            self.validation_dataloader = validation_dataloader
            self.validation_context = validation_context
            self.maybe_enable_amp = maybe_enable_amp
            self.metrics_processor = metrics_processor
            self.pp_schedule = pp_schedule
            self.pp_has_first_stage = pp_has_first_stage
            self.pp_has_last_stage = pp_has_last_stage
            self._conformer_start_id = self._resolve_token_id("[CONFORMERS]", "[CONFORMER]")
            self._conformer_end_id = self._resolve_token_id(
                "[/CONFORMERS]", "[/CONFORMER]"
            )
            self._eos_id = _resolve_special_token_id(
                tokenizer,
                "eos_id",
                (
                    getattr(tokenizer, "eos_token", None),
                    "<|endoftext|>",
                ),
            )
            self._pad_id = _resolve_special_token_id(
                tokenizer,
                "pad_id",
                (
                    getattr(tokenizer, "pad_token", None),
                    "<|endoftext|>",
                ),
            )

            if self.job_config.validation.steps == -1 and titan_logger is not None:
                titan_logger.warning(
                    "Setting validation steps to -1 might cause hangs because of "
                    "unequal sample counts across ranks when dataset is exhausted."
                )

        def _resolve_token_id(self, *tokens: str) -> Optional[int]:
            for token in tokens:
                if not token:
                    continue
                
                # Handle wrapped tokenizers (e.g. TorchTitan's HuggingFaceTokenizer)
                tokenizer = self.tokenizer
                if not hasattr(tokenizer, "convert_tokens_to_ids") and hasattr(tokenizer, "tokenizer"):
                    tokenizer = tokenizer.tokenizer

                if hasattr(tokenizer, "convert_tokens_to_ids"):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                elif hasattr(tokenizer, "encode"):
                    encoded = tokenizer.encode(token, add_special_tokens=False)
                    if isinstance(encoded, list) and len(encoded) == 1:
                        token_id = encoded[0]
                    else:
                        token_id = None
                else:
                    token_id = None

                if isinstance(token_id, int) and token_id >= 0:
                    return int(token_id)
            return None

        def _build_prompt_tensor(
            self, token_ids: Sequence[Union[int, float]], device: torch.device
        ) -> Optional[torch.Tensor]:
            ids: List[int] = []
            for tid in token_ids:
                try:
                    ids.append(int(tid))
                except (TypeError, ValueError):
                    continue

            if self._conformer_start_id is None:
                logger.warning(
                    "Skipping numerical validation because conformer start token is missing."
                )
                return None

            ids.append(self._conformer_start_id)
            return torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)

        def _greedy_decode(
            self,
            model: torch.nn.Module,
            prompt: torch.Tensor,
            max_new_tokens: int,
            max_total_len: int,
        ) -> torch.Tensor:
            generated = prompt
            for _ in range(max_new_tokens):
                logits = model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

                token_id = int(next_token.item())
                if self._conformer_end_id is not None and token_id == self._conformer_end_id:
                    break
                if self._eos_id is not None and token_id == self._eos_id:
                    break
                if generated.shape[1] >= max_total_len:
                    break
            return generated

        def _extract_conformer_text(self, token_tensor: torch.Tensor) -> str:
            decoded = self.tokenizer.decode(
                token_tensor[0].tolist(), skip_special_tokens=False
            )
            conformer = extract_between(decoded, "[CONFORMERS]", "[/CONFORMERS]")
            if not conformer:
                conformer = extract_between(decoded, "[CONFORMER]", "[/CONFORMER]")
            return conformer.strip()

        def _compute_rmsd_stats(
            self, generated_mol, ground_truths: Sequence
        ) -> Tuple[float, float, float]:
            rmsds: List[float] = []
            for gt in ground_truths:
                try:
                    rmsds.append(
                        float(get_best_rmsd(generated_mol, gt, use_alignmol=False))
                    )
                except Exception:
                    rmsds.append(float("nan"))

            if not rmsds:
                return float("nan"), float("nan"), float("nan")

            arr = np.array(rmsds, dtype=float)
            if np.isnan(arr).all():
                return float("nan"), float("nan"), float("nan")

            return (
                float(np.nanmin(arr)),
                float(np.nanmax(arr)),
                float(np.nanmean(arr)),
            )

        def _log_numerical_metrics(
            self,
            min_rmsds: List[float],
            max_rmsds: List[float],
            avg_rmsds: List[float],
            failures: int,
            step: int,
        ) -> None:
            valid_min = np.array(min_rmsds, dtype=float)
            valid_min = valid_min[~np.isnan(valid_min)]

            valid_max = np.array(max_rmsds, dtype=float)
            valid_max = valid_max[~np.isnan(valid_max)]

            valid_avg = np.array(avg_rmsds, dtype=float)
            valid_avg = valid_avg[~np.isnan(valid_avg)]

            metrics: Dict[str, float] = {
                "numerical_val/failures": float(failures),
                "numerical_val/successes": float(valid_min.size),
            }
            if valid_min.size > 0:
                metrics.update(
                    {
                        "numerical_val/rmsd_min_min": float(np.nanmin(valid_min)),
                        "numerical_val/rmsd_min_max": float(np.nanmax(valid_min)),
                        "numerical_val/rmsd_min_mean": float(np.nanmean(valid_min)),
                        "numerical_val/rmsd_min_std": float(np.nanstd(valid_min)),
                        "numerical_val/rmsd_max_mean": float(np.nanmean(valid_max)),
                        "numerical_val/rmsd_avg_mean": float(np.nanmean(valid_avg)),
                    }
                )

            successes = int(metrics.get("numerical_val/successes", 0))
            suffix = (
                f" | rmsd min_mean={metrics.get('numerical_val/rmsd_min_mean', float('nan')):.4f}"
                f" max_mean={metrics.get('numerical_val/rmsd_max_mean', float('nan')):.4f}"
                f" avg_mean={metrics.get('numerical_val/rmsd_avg_mean', float('nan')):.4f}"
                if valid_min.size > 0
                else ""
            )
            logger.info(
                f"Numerical validation (step {step}): successes={successes} failures={failures}{suffix}"
            )

            try:  # best effort W&B logging
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except ModuleNotFoundError:
                logger.info("W&B not installed; skipping numerical validation logging.")
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(f"Failed to log numerical metrics to W&B: {exc}")
            if not PRETOKENIZED_PROMPTS_PATH.exists():
                logger.warning(
                    f"Numerical validation prompts file not found at {PRETOKENIZED_PROMPTS_PATH}"
                )
                return []
            try:
                with open(PRETOKENIZED_PROMPTS_PATH, "r") as fh:
                    payload = json.load(fh)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(
                    f"Failed to load pretokenized prompts from {PRETOKENIZED_PROMPTS_PATH}: {exc}"
                )
                return []

            if not isinstance(payload, dict):
                logger.warning(
                    f"Pretokenized prompts should be a dict of SMILES->token list, got {type(payload)}"
                )
                return []

            items = sorted(payload.items(), key=lambda kv: kv[0])[
                :NUM_NUMERICAL_VALIDATION_PROMPTS
            ]
            normalized: List[Tuple[str, List[int]]] = []
            for key, value in items:
                if isinstance(value, list):
                    try:
                        normalized.append((key, [int(v) for v in value]))
                    except Exception:
                        normalized.append((key, []))
                else:
                    normalized.append((key, []))
            return normalized

        def _load_ground_truths(self) -> Dict[str, List]:
            if not VALIDATION_PICKLE_PATH.exists():
                logger.warning(
                    f"Ground truth conformers not found at {VALIDATION_PICKLE_PATH}"
                )
                return {}
            try:
                with open(VALIDATION_PICKLE_PATH, "rb") as fh:
                    data = cloudpickle.load(fh)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(
                    f"Failed to load ground truth conformers from {VALIDATION_PICKLE_PATH}: {exc}"
                )
                return {}

            if not isinstance(data, dict):
                logger.warning(
                    f"Expected dict[str, list[Mol]] in {VALIDATION_PICKLE_PATH}, got {type(data)}"
                )
                return {}
            return data

        def _log_numerical_metrics(
            self,
            min_rmsds: List[float],
            max_rmsds: List[float],
            avg_rmsds: List[float],
            failures: int,
            step: int,
        ) -> None:
            valid_min = np.array(min_rmsds, dtype=float)
            valid_min = valid_min[~np.isnan(valid_min)]

            valid_max = np.array(max_rmsds, dtype=float)
            valid_max = valid_max[~np.isnan(valid_max)]

            valid_avg = np.array(avg_rmsds, dtype=float)
            valid_avg = valid_avg[~np.isnan(valid_avg)]

            metrics: Dict[str, float] = {
                "numerical_val/failures": float(failures),
                "numerical_val/successes": float(valid_min.size),
            }
            if valid_min.size > 0:
                metrics.update(
                    {
                        "numerical_val/rmsd_min_min": float(np.nanmin(valid_min)),
                        "numerical_val/rmsd_min_max": float(np.nanmax(valid_min)),
                        "numerical_val/rmsd_min_mean": float(np.nanmean(valid_min)),
                        "numerical_val/rmsd_min_std": float(np.nanstd(valid_min)),
                        # New metrics
                        "numerical_val/rmsd_max_mean": float(np.nanmean(valid_max)),
                        "numerical_val/rmsd_avg_mean": float(np.nanmean(valid_avg)),
                    }
                )

            successes = int(metrics.get("numerical_val/successes", 0))
            suffix = (
                f" | rmsd min_mean={metrics.get('numerical_val/rmsd_min_mean', float('nan')):.4f}"
                f" max_mean={metrics.get('numerical_val/rmsd_max_mean', float('nan')):.4f}"
                f" avg_mean={metrics.get('numerical_val/rmsd_avg_mean', float('nan')):.4f}"
                if valid_min.size > 0
                else ""
            )
            logger.info(
                f"Numerical validation (step {step}): successes={successes} failures={failures}{suffix}"
            )

            try:  # best effort W&B logging
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except ModuleNotFoundError:
                logger.info("W&B not installed; skipping numerical validation logging.")
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(f"Failed to log numerical metrics to W&B: {exc}")

        def _run_numerical_validation(
            self, model_parts: List[torch.nn.Module], step: int
        ) -> None:
            if not getattr(self.job_config.validation, "numerical_validation", False):
                return
            # Removed primary rank check to support FSDP synchronization
            if self.parallel_dims.pp_enabled or len(model_parts) != 1:
                logger.warning(
                    "Numerical validation currently supports single-stage models; skipping."
                )
                return
            if self._conformer_start_id is None or self._conformer_end_id is None:
                logger.warning(
                    "Conformer tokens missing from tokenizer; numerical validation skipped."
                )
                return

            prompts = self._load_prompts()
            ground_truths = self._load_ground_truths()
            if not prompts or not ground_truths:
                return

            model = model_parts[0]
            device = next(model.parameters()).device
            max_seq_len = int(
                getattr(self.job_config.validation, "seq_len", 2048)
                or getattr(self.job_config.training, "seq_len", 2048)
                or 2048
            )

            min_rmsds: List[float] = []
            max_rmsds: List[float] = []
            avg_rmsds: List[float] = []
            failures = 0
            was_training = model.training
            model.eval()

            with torch.inference_mode():
                for key, token_ids in prompts:
                    gt_confs = ground_truths.get(key)
                    if not gt_confs:
                        failures += 1
                        continue

                    prompt_tensor = self._build_prompt_tensor(token_ids, device)
                    if prompt_tensor is None:
                        failures += 1
                        continue

                    available = max(max_seq_len - prompt_tensor.shape[1], 1)
                    max_new_tokens = min(512, available)

                    generated_ids = self._greedy_decode(
                        model, prompt_tensor, max_new_tokens, max_seq_len
                    )
                    conformer_text = self._extract_conformer_text(generated_ids)
                    if not conformer_text:
                        failures += 1
                        continue

                    try:
                        generated_mol = decode_cartesian_raw(conformer_text)
                    except Exception:
                        failures += 1
                        continue

                    min_val, max_val, avg_val = self._compute_rmsd_stats(
                        generated_mol, gt_confs
                    )

                    if np.isnan(min_val):
                        failures += 1
                    else:
                        min_rmsds.append(min_val)
                        max_rmsds.append(max_val)
                        avg_rmsds.append(avg_val)

            if was_training:
                model.train()

            if _is_primary_rank():
                self._log_numerical_metrics(
                    min_rmsds, max_rmsds, avg_rmsds, failures, step
                )

        @torch.no_grad()
        def validate(
            self,
            model_parts: list[torch.nn.Module],
            step: int,
        ) -> None:
            super().validate(model_parts, step)
            try:
                self._run_numerical_validation(model_parts, step)
            except Exception as exc:  # pragma: no cover - safety
                logger.warning("Numerical validation failed: %s", exc)

    MolGenValidatorClass = MolGenValidator

def build_molgen_validator(
    job_config: MolGenJobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer,
    parallel_dims,
    loss_fn,
    validation_context,
    maybe_enable_amp,
    metrics_processor,
    pp_schedule=None,
    pp_has_first_stage=None,
    pp_has_last_stage=None,
) -> BaseValidator:
    if MolGenValidatorClass is None:
        raise RuntimeError(
            "Torchtitan validator bindings are unavailable. Install torchtitan "
            "and ensure the environment exposes torchtitan.components.validate."
        )

    data_cfg = getattr(job_config, "molgen_data", None)
    if data_cfg is None:
        raise ValueError(
            "Missing 'molgen_data' section in the job config. "
            "Set job.custom_config_module="
            "'molgen3D.training.pretraining.config.custom_job_config'."
        )

    # Use fewer workers for validation to reduce memory usage
    # Validation doesn't need as many workers as training since it's not as performance-critical
    val_num_workers = min(data_cfg.num_workers, 2)  # Cap at 2 workers for validation
    infinite_validation = job_config.validation.steps != -1
    validation_dataloader = build_dataloader(
        train_path=_resolve_validation_path(job_config),
        tokenizer_path=_resolve_tokenizer_path(data_cfg, job_config),
        tokenizer=tokenizer,
        seq_len=job_config.validation.seq_len,
        batch_size=job_config.validation.local_batch_size,
        num_workers=val_num_workers,
        pin_memory=data_cfg.pin_memory,
        shuffle_lines=False,
        # Mirror TorchTitan’s default: only allow finite validation when the user
        # explicitly sets steps=-1, otherwise keep the loader infinite so every
        # rank can always advance to the requested step count.
        infinite=infinite_validation,
        seed=data_cfg.seed if data_cfg.seed is not None else job_config.training.seed,
        min_emb_len=data_cfg.min_emb_len,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=min(data_cfg.prefetch_factor or 2, 2),  # Reduce prefetch for validation
        world_size=dp_world_size,
        rank=dp_rank,
    )

    return MolGenValidatorClass(  # type: ignore[arg-type]
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        validation_dataloader=validation_dataloader,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )

