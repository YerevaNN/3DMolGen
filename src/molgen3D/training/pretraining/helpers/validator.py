from __future__ import annotations

import json
import cloudpickle
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from loguru import logger

try:
    from torchtitan.components.dataloader import BaseDataLoader
    from torchtitan.components.validate import BaseValidator
except Exception:
    class BaseDataLoader:  # type: ignore[too-many-ancestors]
        pass
    BaseValidator = None

from transformers import AutoTokenizer

from molgen3D.training.pretraining.config.custom_job_config import (
    JobConfig as MolGenJobConfig,
)

from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.utils.utils import get_best_rmsd
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_molgen_validator as _build_molgen_validator_base,
    _resolve_special_token_id,
    _resolve_tokenizer_path,
    MolGenValidatorClass as BaseMolGenValidatorClass,
)
from molgen3D.config.paths import get_data_path
MAX_CONFORMER_TOKENS = 2000  # Typical conformer is 200-500 tokens
PRETOKENIZED_PROMPTS_PATH = get_data_path("pretokenized_prompts")
VALIDATION_PICKLE_PATH = get_data_path("validation_pickle")

# Failure type constants
FAIL_NO_CLOSING_TAG = "no_closing_tag"
FAIL_EMPTY_CONFORMER = "empty_conformer"
FAIL_PARSING_ERROR = "parsing_error"
FAIL_SMILES_MISMATCH = "smiles_mismatch"
FAIL_RMSD_NAN = "rmsd_nan"
FAIL_NO_GROUND_TRUTH = "no_ground_truth"

def _is_primary_rank() -> bool:
    """
    Return True only for the primary logging rank to avoid duplicated work.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class MolGenNumericalValidator(BaseMolGenValidatorClass):
    """
    Extended MolGen validator with numerical conformer validation.
    
    Extends the base MolGenValidator from dataloader.py with additional
    functionality for generating conformers and computing RMSD metrics.
    """
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
        # Call base class __init__
        super().__init__(
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
        
        # Store output directory for saving failed generations
        self._output_dir = Path(job_config.job.dump_folder)
        
        # Load a proper AutoTokenizer for token resolution
        # TorchTitan's HuggingFaceTokenizer doesn't load added_tokens.json properly
        data_cfg = getattr(job_config, "molgen_data", None)
        if data_cfg is not None:
            tokenizer_path = _resolve_tokenizer_path(data_cfg, job_config)
            self._token_resolver = AutoTokenizer.from_pretrained(
                tokenizer_path, use_fast=True
            )
            if _is_primary_rank():
                logger.info(f"Loaded AutoTokenizer from {tokenizer_path} for token resolution")
        else:
            self._token_resolver = None
            if _is_primary_rank():
                logger.warning("No molgen_data config; using wrapped tokenizer for token resolution")
        
        if _is_primary_rank():
            logger.info("Resolving conformer tokens with debug=True...")
        self._conformer_start_id = self._token_resolver.convert_tokens_to_ids("[CONFORMER]")
        self._conformer_end_id = self._token_resolver.convert_tokens_to_ids("[/CONFORMER]")
        if _is_primary_rank():
            # Debug: show tokenizer type and attributes to help diagnose issues
            tok_type = type(tokenizer).__name__
            has_inner = hasattr(tokenizer, "tokenizer")
            inner_type = type(tokenizer.tokenizer).__name__ if has_inner else "N/A"
            logger.info(
                f"MolGenNumericalValidator tokenizer: type={tok_type}, has_inner={has_inner}, "
                f"inner_type={inner_type}"
            )
            logger.info(
                f"MolGenNumericalValidator token resolution: conformer_start_id={self._conformer_start_id}, "
                f"conformer_end_id={self._conformer_end_id}"
            )
        self._eos_id = self._token_resolver.eos_token_id
        self._pad_id = self._token_resolver.pad_token_id
        

    def _build_prompt_tensor(
        self, token_ids: Sequence[Union[int, float]], device: torch.device
    ) -> Optional[torch.Tensor]:
        ids = token_ids + [self._conformer_start_id]
        return torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)


    def _greedy_decode(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        max_new_tokens: int,
        max_total_len: int,
    ) -> torch.Tensor:
        """Single-sequence greedy decoding."""
        generated = prompt
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            token_id = int(next_token.item())
            if token_id == self._conformer_end_id:
                break
            if token_id == self._eos_id:
                break
            if generated.shape[1] >= max_total_len:
                break
        return generated

    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens using the best available tokenizer.
        
        Uses _token_resolver (AutoTokenizer) when available because it properly
        handles custom tokens like [CONFORMER]. Falls back to wrapped tokenizer.
        """
        decoder = self._token_resolver if self._token_resolver is not None else self.tokenizer
        return decoder.decode(tokens, skip_special_tokens=False)

    def _extract_conformer_text(self, token_tensor: torch.Tensor) -> str:
        """Extract conformer text from a single sequence."""
        if token_tensor.dim() == 2:
            tokens = token_tensor[0].tolist()
        else:
            tokens = token_tensor.tolist()
        decoded = self._decode_tokens(tokens)
        conformer = extract_between(decoded, "[CONFORMER]", "[/CONFORMER]")
        return conformer.strip() if conformer else ""

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

    def _save_failed_generations(
        self,
        failed_generations: List[Tuple[str, str, str]],  # (prompt, full_generated, fail_type)
        step: int,
    ) -> None:
        """Save all failed generations to a text file in the job's output directory."""
        failed_dir = self._output_dir / "failed_generations"
        failed_dir.mkdir(parents=True, exist_ok=True)
        output_path = failed_dir / f"failed_step_{step}.txt"
        
        with open(output_path, "w") as f:
            f.write(f"Failed Generations Report - Step {step}\n")
            f.write(f"Total failures: {len(failed_generations)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (prompt, full_generated, fail_type) in enumerate(failed_generations, 1):
                f.write(f"--- Failure #{i} ---\n")
                f.write(f"Prompt:\n{prompt}\n\n")
                f.write(f"Full Generated String:\n{full_generated}\n\n")
                f.write(f"Fail Type: {fail_type}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Saved {len(failed_generations)} failed generations to {output_path}")

    def _log_numerical_metrics(
        self,
        min_rmsds: List[float],
        max_rmsds: List[float],
        avg_rmsds: List[float],
        failure_counts: Dict[str, int],
        step: int,
        sample_failures: List[Tuple[str, str, str]],  # (smiles, failure_type, details)
    ) -> None:
        valid_min = np.array(min_rmsds, dtype=float)
        valid_min = valid_min[~np.isnan(valid_min)]

        valid_max = np.array(max_rmsds, dtype=float)
        valid_max = valid_max[~np.isnan(valid_max)]

        valid_avg = np.array(avg_rmsds, dtype=float)
        valid_avg = valid_avg[~np.isnan(valid_avg)]

        total_failures = sum(failure_counts.values())
        metrics: Dict[str, float] = {
            "numerical_val/failures": float(total_failures),
            "numerical_val/successes": float(valid_min.size),
        }
        
        # Add per-failure-type metrics
        for fail_type, count in failure_counts.items():
            metrics[f"numerical_val/fail_{fail_type}"] = float(count)
        
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
        
        # Log failure breakdown
        failure_str = ", ".join(f"{k}={v}" for k, v in sorted(failure_counts.items()) if v > 0)
        logger.info(
            f"Numerical validation (step {step}): successes={successes} failures={total_failures} "
            f"[{failure_str}]{suffix}"
        )
        
        # Log sample failures for debugging (first 3)
        if sample_failures:
            logger.info(f"Sample failures (first {min(3, len(sample_failures))}):")
            for smiles, fail_type, details in sample_failures[:3]:
                logger.info(f"  [{fail_type}] {smiles[:50]}... : {details[:100]}")

        try:  # best effort W&B logging
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ModuleNotFoundError:
            logger.info("W&B not installed; skipping numerical validation logging.")
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning(f"Failed to log numerical metrics to W&B: {exc}")

    def _load_prompts(self, num_prompts: int) -> List[Tuple[str, List[int]]]:
        if not PRETOKENIZED_PROMPTS_PATH.exists():
            logger.warning(
                f"Numerical validation prompts file not found at {PRETOKENIZED_PROMPTS_PATH}"
            )
            return []
        with open(PRETOKENIZED_PROMPTS_PATH, "r") as fh:
            payload = json.load(fh)

        if not isinstance(payload, dict):
            logger.warning(
                f"Pretokenized prompts should be a dict of SMILES->token list, got {type(payload)}"
            )
            return []

        items = sorted(payload.items(), key=lambda kv: kv[0])[:num_prompts]
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

        with open(VALIDATION_PICKLE_PATH, "rb") as fh:
            ground_truths = cloudpickle.load(fh)
        return ground_truths

    def _run_numerical_validation(
        self, model_parts: List[torch.nn.Module], step: int
    ) -> None:
        if not getattr(self.job_config.validation, "numerical_validation", False):
            return
        if self.parallel_dims.pp_enabled or len(model_parts) != 1:
            if _is_primary_rank():
                logger.warning(
                    "Numerical validation currently supports single-stage models; skipping."
                )
            return


        num_val_molecules = getattr(self.job_config.validation, "num_val_molecules", 10)
        prompts = self._load_prompts(num_val_molecules)
        ground_truths = self._load_ground_truths()
        if not prompts or not ground_truths:
            if _is_primary_rank():
                logger.warning(
                    f"Numerical validation skipped: prompts={len(prompts) if prompts else 0}, "
                    f"ground_truths={len(ground_truths) if ground_truths else 0}"
                )
            return

        model = model_parts[0]
        device = next(model.parameters()).device
        max_seq_len = int(
            getattr(self.job_config.validation, "seq_len", 2048)
            or getattr(self.job_config.training, "seq_len", 2048)
            or 2048
        )

        # Filter prompts to those with ground truths
        valid_prompts = [(key, token_ids) for key, token_ids in prompts if ground_truths.get(key)]
        skipped_no_gt = len(prompts) - len(valid_prompts)
        
        if _is_primary_rank():
            logger.info(
                f"Starting numerical validation: {len(valid_prompts)} prompts, step={step}"
            )

        min_rmsds: List[float] = []
        max_rmsds: List[float] = []
        avg_rmsds: List[float] = []
        
        # Track failure types
        failure_counts: Dict[str, int] = {
            FAIL_NO_CLOSING_TAG: 0,
            FAIL_EMPTY_CONFORMER: 0,
            FAIL_PARSING_ERROR: 0,
            FAIL_SMILES_MISMATCH: 0,
            FAIL_RMSD_NAN: 0,
            FAIL_NO_GROUND_TRUTH: skipped_no_gt,
        }
        sample_failures: List[Tuple[str, str, str]] = []  # (smiles, fail_type, details)
        all_failed_generations: List[Tuple[str, str, str]] = []  # (prompt, full_generated, fail_type)
        
        was_training = model.training
        model.eval()

        with torch.inference_mode():
            for i, (key, token_ids) in enumerate(valid_prompts):
                # Build prompt tensor
                prompt = self._build_prompt_tensor(token_ids, device)
                if prompt is None:
                    failure_counts[FAIL_PARSING_ERROR] += 1
                    if len(sample_failures) < 10:
                        sample_failures.append((key, FAIL_PARSING_ERROR, "failed to build prompt"))
                    continue
                
                # Calculate max new tokens
                available = max(max_seq_len - prompt.shape[1], 1)
                max_new_tokens = min(MAX_CONFORMER_TOKENS, available)
                
                # Generate
                generated = self._greedy_decode(model, prompt, max_new_tokens, max_seq_len)
                
                # Decode full generated string for logging
                tokens = generated[0].tolist()
                full_decoded = self._decode_tokens(tokens)
                
                # Extract conformer text
                conformer_text = self._extract_conformer_text(generated)
                
                # Check for empty/missing conformer
                if not conformer_text:
                    # Check if [/CONFORMER] is missing
                    if "[/CONFORMER]" not in full_decoded:
                        failure_counts[FAIL_NO_CLOSING_TAG] += 1
                        all_failed_generations.append((key, full_decoded, FAIL_NO_CLOSING_TAG))
                        if len(sample_failures) < 10:
                            # Show what was generated after [CONFORMER]
                            conf_start_idx = full_decoded.find("[CONFORMER]")
                            if conf_start_idx >= 0:
                                gen_part = full_decoded[conf_start_idx:conf_start_idx + 150]
                            else:
                                gen_part = full_decoded[-150:]
                            sample_failures.append((key, FAIL_NO_CLOSING_TAG, f"generated: {gen_part}"))
                    else:
                        failure_counts[FAIL_EMPTY_CONFORMER] += 1
                        all_failed_generations.append((key, full_decoded, FAIL_EMPTY_CONFORMER))
                        if len(sample_failures) < 10:
                            sample_failures.append((key, FAIL_EMPTY_CONFORMER, f"decoded: {full_decoded[-100:]}"))
                    continue
                
                # Try to parse the conformer
                try:
                    generated_mol = decode_cartesian_v2(conformer_text)
                except Exception as e:
                    failure_counts[FAIL_PARSING_ERROR] += 1
                    all_failed_generations.append((key, full_decoded, FAIL_PARSING_ERROR))
                    if len(sample_failures) < 10:
                        sample_failures.append((key, FAIL_PARSING_ERROR, f"{e}: {conformer_text[:80]}"))
                    continue
                
                # Check SMILES match using same_molecular_graph
                # Extract SMILES from the conformer text using strip_smiles
                generated_smiles = strip_smiles(conformer_text)
                if not same_molecular_graph(key, generated_smiles):
                    failure_counts[FAIL_SMILES_MISMATCH] += 1
                    all_failed_generations.append((key, full_decoded, FAIL_SMILES_MISMATCH))
                    if len(sample_failures) < 10:
                        sample_failures.append((key, FAIL_SMILES_MISMATCH, f"got: {generated_smiles}"))
                    continue
                
                gt_confs = ground_truths.get(key, [])
                min_val, max_val, avg_val = self._compute_rmsd_stats(
                    generated_mol, gt_confs
                )
                
                if np.isnan(min_val):
                    failure_counts[FAIL_RMSD_NAN] += 1
                    all_failed_generations.append((key, full_decoded, FAIL_RMSD_NAN))
                    if len(sample_failures) < 10:
                        sample_failures.append((key, FAIL_RMSD_NAN, f"RMSD returned NaN"))
                else:
                    min_rmsds.append(min_val)
                    max_rmsds.append(max_val)
                    avg_rmsds.append(avg_val)
                
                # Progress logging
                if _is_primary_rank() and (i + 1) % 5 == 0:
                    total_failures = sum(failure_counts.values())
                    logger.info(
                        f"Numerical validation progress: {i+1}/{len(valid_prompts)}, "
                        f"successes={len(min_rmsds)}, failures={total_failures}"
                    )

        if was_training:
            model.train()

        if _is_primary_rank():
            self._log_numerical_metrics(
                min_rmsds, max_rmsds, avg_rmsds, failure_counts, step, sample_failures
            )
            # Save all failed generations to file
            if all_failed_generations:
                self._save_failed_generations(all_failed_generations, step)

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[torch.nn.Module],
        step: int,
    ) -> None:
        # Run regular validation (inherited from base class)
        # Note: TorchTitan only calls validate() at validation.freq intervals,
        # so regular validation will run whenever this method is called
        super().validate(model_parts, step)
        
        # Run numerical validation independently based on its own frequency
        numerical_val_freq = getattr(self.job_config.validation, "numerical_validation_freq", 0)
        numerical_val_enabled = getattr(self.job_config.validation, "numerical_validation", False)
        
        # Check if numerical validation should run at this step
        should_run_numerical = (
            numerical_val_enabled
            and numerical_val_freq > 0
            and step % numerical_val_freq == 0
            and step > 1
        )
        
        if should_run_numerical:
            try:
                self._run_numerical_validation(model_parts, step)
            except Exception as exc:  # pragma: no cover - safety
                import traceback
                logger.warning(f"Numerical validation failed: {exc}\n{traceback.format_exc()}")

MolGenNumericalValidatorClass = MolGenNumericalValidator

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
) -> Any:
    """
    Build the MolGen numerical validator with conformer generation and RMSD metrics.
    
    This uses the base build_molgen_validator from dataloader.py but passes in the
    extended MolGenNumericalValidator class for numerical validation functionality.
    """
    if MolGenNumericalValidatorClass is None:
        raise RuntimeError(
            "Torchtitan validator bindings are unavailable. Install torchtitan "
            "and ensure the environment exposes torchtitan.components.validate."
        )

    return _build_molgen_validator_base(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
        validator_class=MolGenNumericalValidatorClass,
    )
