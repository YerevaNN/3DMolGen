"""
Numerical validation for GRPO training.

This module implements numerical validation for GRPO runs, similar to the
pretraining numerical validator but adapted for the GRPO training setup.
It performs actual conformer generation during validation and computes
RMSD metrics against ground truth conformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import random
import time
import math

import cloudpickle
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_v2,
    strip_smiles,
)
from molgen3D.evaluation.utils import (
    DEFAULT_THRESHOLDS,
    covmat_metrics,
    extract_between,
    same_molecular_graph,
)
from molgen3D.utils.utils import get_best_rmsd
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.config.paths import get_data_path

# Failure type constants
FAIL_NO_CLOSING_TAG = "no_closing_tag"
FAIL_EMPTY_CONFORMER = "empty_conformer"
FAIL_PARSING_ERROR = "parsing_error"
FAIL_SMILES_MISMATCH = "smiles_mismatch"
FAIL_RMSD_NAN = "rmsd_nan"


class GRPONumericalValidator:
    """
    Numerical validator for GRPO training.

    Performs actual conformer generation during validation and computes RMSD
    metrics against ground truth conformers to validate model performance.
    """

    def __init__(
        self,
        config,
        tokenizer,
        stats,
        output_dir: str,
    ):
        """
        Initialize the numerical validator.

        Args:
            config: GRPO configuration
            tokenizer: Tokenizer for decoding
            stats: Run statistics tracker
            output_dir: Output directory for saving results
        """
        self.config = config
        self.tokenizer = tokenizer
        self.stats = stats
        self.output_dir = Path(output_dir)

        self._validation_prompts: List[str] = []
        self._ground_truths: Dict[str, List] = {}

        self._conformer_start_id = self.tokenizer.convert_tokens_to_ids("[CONFORMER]")
        self._conformer_end_id = self.tokenizer.convert_tokens_to_ids("[/CONFORMER]")
        self._eos_id = self.tokenizer.eos_token_id

        self._pad_id = self.tokenizer.pad_token_id or self._eos_id or 0

        if self._conformer_start_id is None or self._conformer_end_id is None:
            raise ValueError("Tokenizer must define [CONFORMER] and [/CONFORMER] tokens.")

    def _load_validation_prompts(self) -> List[str]:
        """
        Load a sample of SMILES strings for validation.

        For GRPO numerical validation we re-use the validation ground truths
        from `validation_pickle` (valid_set.pickle) defined in paths.yaml.
        We simply sample SMILES keys from that pickle.
        """
        gt_path = get_data_path("validation_pickle")
        logger.info(f"Loading numerical validation keys from {gt_path}")

        with open(gt_path, "rb") as fh:
            gt_dict = cloudpickle.load(fh)

        all_keys = list(gt_dict.keys())
        rng = np.random.default_rng(self._get_validation_seed())
        rng.shuffle(all_keys)

        prompts: List[str] = []
        for key in all_keys:
            # Keys are SMILES strings for which we have ground-truth conformers.
            prompts.append(str(key).strip())
            if len(prompts) >= self.config.validation.num_val_molecules:
                break
        ground_truths = {key: gt_dict[key] for key in prompts}

        logger.info(f"Loaded {len(prompts)} validation SMILES keys from validation_pickle")
        return prompts, ground_truths

    def _build_prompt_tensor(
        self, smiles: str, device: torch.device
    ) -> torch.Tensor:
    def _get_validation_seed(self) -> int | None:
        """Return the seed to use for deterministic numerical validation."""
        validation_seed = getattr(self.config.validation, "seed", None)
        if validation_seed is not None:
            return int(validation_seed)
        grpo_cfg = getattr(self.config, "grpo", None)
        if grpo_cfg is not None and hasattr(grpo_cfg, "seed"):
            return int(grpo_cfg.seed)
        return None

        """Build a prompt tensor for conformer generation."""
        prompt_text = f"[SMILES]{smiles}[/SMILES][CONFORMER]"
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)



    def _compute_rmsd_stats(
        self, generated_mol, ground_truths: List
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Compute RMSD statistics against ground truths.

        Returns:
            Tuple of (min_rmsd, max_rmsd, avg_rmsd, rmsd_vector)
            where rmsd_vector has shape (num_ground_truths,) and may contain NaNs.
        """
        rmsds: List[float] = []
        for gt in ground_truths:
            try:
                rmsd = float(get_best_rmsd(generated_mol, gt, use_alignmol=False))
                rmsds.append(rmsd)
            except Exception as e:
                logger.debug(f"RMSD calculation failed: {e}")
                rmsds.append(float("nan"))

        arr = np.array(rmsds, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return float("nan"), float("nan"), float("nan"), arr

        return (
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
            float(np.nanmean(arr)),
            arr,
        )

    def _save_failed_generations(
        self,
        failed_generations: List[Tuple[str, str, str]],
        step: int,
    ) -> None:
        """Save failed generations to a file."""
        if not self.config.validation.save_failed_generations:
            return

        failed_dir = self.output_dir / "numerical_validation" / "failed_generations"
        failed_dir.mkdir(parents=True, exist_ok=True)
        output_path = failed_dir / f"failed_step_{step}.txt"

        with open(output_path, "w") as f:
            f.write(f"Failed Numerical Validation Generations - Step {step}\n")
            f.write(f"Total failures: {len(failed_generations)}\n")
            f.write("Format: Prompt | error code | generated text\n")
            f.write("=" * 80 + "\n")

            for smiles, full_generated, fail_type in failed_generations:
                # Keep the full generated text as-is (don't clean newlines)
                f.write(f"{smiles} | {fail_type} | {full_generated}\n")

        logger.info(f"Saved {len(failed_generations)} failed generations to {output_path}")

    def _log_validation_metrics(
        self,
        min_rmsds: List[float],
        max_rmsds: List[float],
        avg_rmsds: List[float],
        failure_counts: Dict[str, int],
        step: int,
        sample_failures: List[Tuple[str, str, str]],
        cov_r_mean: float | None = None,
        cov_p_mean: float | None = None,
        mat_r_mean: float | None = None,
        mat_p_mean: float | None = None,
        cov_r_by_threshold: List[float] | None = None,
        cov_p_by_threshold: List[float] | None = None,
        # Pre-computed global statistics (for DDP to avoid large data transfers)
        global_rmsd_min_min: float | None = None,
        global_rmsd_min_max: float | None = None,
        global_rmsd_min_mean: float | None = None,
        global_rmsd_min_std: float | None = None,
        global_rmsd_max_mean: float | None = None,
        global_rmsd_avg_mean: float | None = None,
        global_successes: int | None = None,
    ) -> Dict[str, float]:
        """Log numerical validation metrics."""
        # Use pre-computed global statistics if provided (for DDP), otherwise compute from arrays
        has_global_stats = global_successes is not None and global_rmsd_min_min is not None

        valid_min = np.array([], dtype=float)
        valid_max = np.array([], dtype=float)
        valid_avg = np.array([], dtype=float)

        if has_global_stats:
            # Use pre-computed global statistics from DDP aggregation
            total_failures = sum(failure_counts.values())
            metrics = {
                "numerical_val/failures": float(total_failures),
                "numerical_val/successes": float(global_successes),
            }

            for fail_type, count in failure_counts.items():
                metrics[f"numerical_val/fail_{fail_type}"] = float(count)

            if global_successes > 0:
                metrics.update({
                    "numerical_val/rmsd_min_min": global_rmsd_min_min,
                    "numerical_val/rmsd_min_max": global_rmsd_min_max,
                    "numerical_val/rmsd_min_mean": global_rmsd_min_mean,
                    "numerical_val/rmsd_min_std": global_rmsd_min_std,
                    "numerical_val/rmsd_max_mean": global_rmsd_max_mean,
                    "numerical_val/rmsd_avg_mean": global_rmsd_avg_mean,
                })
            has_rmsd_metrics = global_successes > 0
        else:
            # Original logic: compute from arrays
            valid_min = np.array(min_rmsds, dtype=float)
            valid_min = valid_min[~np.isnan(valid_min)]

            valid_max = np.array(max_rmsds, dtype=float)
            valid_max = valid_max[~np.isnan(valid_max)]

            valid_avg = np.array(avg_rmsds, dtype=float)
            valid_avg = valid_avg[~np.isnan(valid_avg)]

            total_failures = sum(failure_counts.values())
            metrics = {
                "numerical_val/failures": float(total_failures),
                "numerical_val/successes": float(valid_min.size),
            }

            for fail_type, count in failure_counts.items():
                metrics[f"numerical_val/fail_{fail_type}"] = float(count)

            if valid_min.size > 0:
                metrics.update({
                    "numerical_val/rmsd_min_min": float(np.nanmin(valid_min)),
                    "numerical_val/rmsd_min_max": float(np.nanmax(valid_min)),
                    "numerical_val/rmsd_min_mean": float(np.nanmean(valid_min)),
                    "numerical_val/rmsd_min_std": float(np.nanstd(valid_min)),
                    "numerical_val/rmsd_max_mean": float(np.nanmean(valid_max)),
                    "numerical_val/rmsd_avg_mean": float(np.nanmean(valid_avg)),
                })
            has_rmsd_metrics = valid_min.size > 0

        # Add covmat-style metrics if available (matching evaluation pipeline)
        if cov_r_mean is not None:
            metrics["numerical_val/COV-R_mean"] = float(cov_r_mean)
        if cov_p_mean is not None:
            metrics["numerical_val/COV-P_mean"] = float(cov_p_mean)
        if mat_r_mean is not None:
            metrics["numerical_val/MAT-R_mean"] = float(mat_r_mean)
        if mat_p_mean is not None:
            metrics["numerical_val/MAT-P_mean"] = float(mat_p_mean)

        # Optional per-threshold coverage logging (useful for parity with evaluation pipeline)
        if cov_r_by_threshold:
            for idx, value in enumerate(cov_r_by_threshold):
                if value is None or not np.isfinite(value):
                    continue
                threshold = float(DEFAULT_THRESHOLDS[idx])
                metrics[f"numerical_val/COV-R_mean@{threshold:g}"] = float(value)
        if cov_p_by_threshold:
            for idx, value in enumerate(cov_p_by_threshold):
                if value is None or not np.isfinite(value):
                    continue
                threshold = float(DEFAULT_THRESHOLDS[idx])
                metrics[f"numerical_val/COV-P_mean@{threshold:g}"] = float(value)

        successes = int(metrics.get("numerical_val/successes", 0))
        covmat_suffix = ""
        if all(
            k in metrics
            for k in (
                "numerical_val/COV-R_mean",
                "numerical_val/COV-P_mean",
                "numerical_val/MAT-R_mean",
                "numerical_val/MAT-P_mean",
            )
        ):
            covmat_suffix = (
                f" | COV-R_mean={metrics['numerical_val/COV-R_mean']:.4f}"
                f" COV-P_mean={metrics['numerical_val/COV-P_mean']:.4f}"
                f" MAT-R_mean={metrics['numerical_val/MAT-R_mean']:.4f}"
                f" MAT-P_mean={metrics['numerical_val/MAT-P_mean']:.4f}"
            )

        suffix = ""
        # Check if we have RMSD metrics (either from DDP global stats or computed arrays)
        if has_rmsd_metrics:
            suffix = (
                f" | rmsd min_mean={metrics.get('numerical_val/rmsd_min_mean', float('nan')):.4f}"
                f" max_mean={metrics.get('numerical_val/rmsd_max_mean', float('nan')):.4f}"
                f" avg_mean={metrics.get('numerical_val/rmsd_avg_mean', float('nan')):.4f}"
            )
        suffix += covmat_suffix

        failure_str = ", ".join(f"{k}={v}" for k, v in sorted(failure_counts.items()) if v > 0)
        logger.info(
            f"Numerical validation (step {step}): successes={successes} failures={total_failures} "
            f"[{failure_str}]{suffix}"
        )

        # Log sample failures for debugging
        if sample_failures:
            logger.info(f"Sample failures (first {min(50, len(sample_failures))}):")
            for smiles, fail_type, details in sample_failures[:50]:
                logger.info(f"  [{fail_type}] {smiles[:50]}... : {details[:200]}")

        # Log to W&B if available (following GRPO logging pattern without explicit step)
        try:
            import wandb
            if wandb.run is not None:
                wandb_keys = {
                    "numerical_val/COV-R_mean",
                    "numerical_val/COV-P_mean",
                    "numerical_val/MAT-R_mean",
                    "numerical_val/MAT-P_mean",
                    "numerical_val/successes",
                    "numerical_val/failures",
                }
                wandb_payload = {
                    key: metrics[key]
                    for key in wandb_keys
                    if key in metrics
                }
                for key, value in metrics.items():
                    if key.startswith("numerical_val/rmsd_"):
                        wandb_payload[key] = value
                if wandb_payload:
                    wandb.log(wandb_payload)
        except ModuleNotFoundError:
            logger.debug("Could not log numerical validation metrics to W&B (not installed).")

        return metrics

    @torch.no_grad()
    def run_validation(
        self,
        model: torch.nn.Module,
        step: int,
        max_seq_len: int = 2048
    ) -> Dict[str, float]:
        """
        Run numerical validation.

        Args:
            model: The model to validate
            step: Current training step
            max_seq_len: Maximum sequence length for generation

        Returns:
            Dictionary of validation metrics
        """
        if not self.config.validation.enable_numerical_validation:
            return {}

        if not self._validation_prompts:
            self._validation_prompts, self._ground_truths = self._load_validation_prompts()


        device = next(model.parameters()).device

        # Detect DDP / distributed context
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        valid_prompts = self._validation_prompts

        base_seed = self._get_validation_seed()
        per_rank_seed: int | None = None
        if base_seed is not None:
            per_rank_seed = int(base_seed) + int(step) + rank * 100_003
            random.seed(per_rank_seed)
            np.random.seed(per_rank_seed % (2**32 - 1))
            torch.manual_seed(per_rank_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(per_rank_seed)

        if rank == 0:
            logger.info(
                f"Starting numerical validation: {len(valid_prompts)} prompts, {sum(len(self._ground_truths[smiles]) for smiles in valid_prompts)} ground truths"
                f"step={step}, world_size={world_size}"
            )

        # Start timing the validation process
        validation_start_time = time.time()

        min_rmsds: List[float] = []
        max_rmsds: List[float] = []
        avg_rmsds: List[float] = []

        # Store per-molecule RMSD vectors for covmat-style metrics (Coverage/Matching)
        per_smiles_rmsd_vectors: Dict[str, List[np.ndarray]] = {}

        failure_counts: Dict[str, int] = {
            FAIL_NO_CLOSING_TAG: 0,
            FAIL_EMPTY_CONFORMER: 0,
            FAIL_PARSING_ERROR: 0,
            FAIL_SMILES_MISMATCH: 0,
            FAIL_RMSD_NAN: 0,
        }
        sample_failures: List[Tuple[str, str, str]] = []  # (smiles, fail_type, details)
        all_failed_generations: List[Tuple[str, str, str]] = []  # (smiles, full_generated, fail_type)

        was_training = model.training
        model.eval()

        try:
            # Get validation batch size from config, scale down for DDP to reduce memory pressure
            validation_batch_size = getattr(self.config.validation, "validation_batch_size", 8)
            if world_size > 1:
                # Scale batch size down for multi-GPU to avoid OOM
                # Each rank processes validation_batch_size sequences simultaneously
                validation_batch_size = max(1, validation_batch_size // world_size)

            # Determine if sampling is enabled for validation
            gen_cfg = sampling_configs[self.config.validation.sampling_config]
            sampling_enabled = bool(getattr(gen_cfg, "do_sample", False))

            # Expand prompts (per-molecule conformer requests)
            prepared_texts: List[str] = []
            prepared_smiles: List[str] = []

            for smiles in valid_prompts:
                prompt_text = f"[SMILES]{smiles}[/SMILES][CONFORMER]"

                # Number of generations to request for this molecule
                if sampling_enabled:
                    num_ground_truths = len(self._ground_truths.get(smiles, []))
                    # Target ~2k conformers per molecule (k = num_ground_truths)
                    num_generations = max(2 * num_ground_truths, 1)
                else:
                    num_generations = 1

                for _ in range(num_generations):
                    prepared_texts.append(prompt_text)
                    prepared_smiles.append(smiles)

            total_prompts_before_sharding = len(prepared_texts)

            if world_size > 1 and total_prompts_before_sharding > 0:
                rank_indices = [
                    idx for idx in range(total_prompts_before_sharding) if idx % world_size == rank
                ]
                prepared_texts = [prepared_texts[idx] for idx in rank_indices]
                prepared_smiles = [prepared_smiles[idx] for idx in rank_indices]

            rank_prompt_count = len(prepared_texts)

            if rank_prompt_count == 0:
                # Log for all ranks to ensure all are aware when validation is skipped
                logger.warning(
                    f"[rank {rank}] No valid prompts for batched generation (after sharding). "
                    f"Total prompts before sharding: {total_prompts_before_sharding}, "
                    f"world_size: {world_size}"
                )
            else:
                # Process prompts in batches
                for batch_start in range(0, rank_prompt_count, validation_batch_size):
                    batch_end = min(batch_start + validation_batch_size, rank_prompt_count)
                    batch_texts = prepared_texts[batch_start:batch_end]
                    batch_smiles = prepared_smiles[batch_start:batch_end]

                    if rank == 0:
                        logger.info(
                            f"[rank {rank}] Processing batch {batch_start // validation_batch_size + 1}: "
                            f"{batch_end - batch_start} prompts"
                        )

                    if not batch_texts:
                        continue
                    
                    encodings = self.tokenizer(
                        batch_texts,
                        padding=True,
                        pad_to_multiple_of=8,
                        return_tensors="pt",
                    )

                    # Move to device and ensure proper types
                    input_ids = encodings['input_ids'].to(device)
                    attention_mask = encodings['attention_mask'].to(device)

                    # Calculate max new tokens based on padded prompt length
                    max_prompt_len = input_ids.shape[1]
                    available = max(max_seq_len - max_prompt_len, 1)
                    max_new_tokens = min(
                        self.config.validation.max_conformer_tokens, available
                    )
                    if rank == 0:
                        logger.info(
                            f"Generation limits: max_seq_len={max_seq_len}, "
                            f"max_prompt_len={max_prompt_len}, "
                            f"available={available}, max_new_tokens={max_new_tokens}"
                        )

                    
                    generate_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "do_sample": sampling_configs[self.config.validation.sampling_config].do_sample,
                        "top_p": sampling_configs[self.config.validation.sampling_config].top_p,
                        "temperature": sampling_configs[self.config.validation.sampling_config].temperature,
                        "max_new_tokens": max_new_tokens,
                        "pad_token_id": self._pad_id,
                        "eos_token_id": self._conformer_end_id,
                    }
                    if per_rank_seed is not None:
                        batch_generator = torch.Generator(device=device)
                        batch_generator.manual_seed(per_rank_seed + batch_start)
                        generate_kwargs["generator"] = batch_generator

                    generated_outputs = model.generate(
                        **generate_kwargs
                    )
                    if rank == 0:
                        logger.info(f"Generated outputs shape: {generated_outputs.shape}")

                    sequences = generated_outputs.detach().cpu()
                    decoded_batch = self.tokenizer.batch_decode(
                        sequences, skip_special_tokens=True
                    )
                    del sequences, generated_outputs

                    # Process each generated sequence in the batch
                    for batch_idx, (smiles, full_decoded, prompt_text) in enumerate(
                        zip(batch_smiles, decoded_batch, batch_texts)
                    ):
                        generated_only_decoded = (
                            full_decoded[len(prompt_text):]
                            if full_decoded.startswith(prompt_text)
                            else full_decoded
                        )

                        # Extract SMILES and conformer similar to inference.py
                        canonical_smiles = extract_between(full_decoded, "[SMILES]", "[/SMILES]")
                        conformer_text = extract_between(full_decoded, "[CONFORMER]", "[/CONFORMER]")

                        # Check for empty/missing conformer
                        if not conformer_text:
                            # Check if [/CONFORMER] is missing
                            if (
                                "[/CONFORMER]" not in full_decoded
                            ):
                                failure_counts[FAIL_NO_CLOSING_TAG] += 1
                                all_failed_generations.append(
                                    (smiles, generated_only_decoded, FAIL_NO_CLOSING_TAG)
                                )
                                if len(sample_failures) < 1000:
                                    sample_failures.append(
                                        (
                                            smiles,
                                            FAIL_NO_CLOSING_TAG,
                                            f"generated: {generated_only_decoded[:200]}",
                                        )
                                    )
                            else:
                                failure_counts[FAIL_EMPTY_CONFORMER] += 1
                                all_failed_generations.append(
                                    (smiles, generated_only_decoded, FAIL_EMPTY_CONFORMER)
                                )
                                if len(sample_failures) < 1000:
                                    sample_failures.append(
                                        (
                                            smiles,
                                            FAIL_EMPTY_CONFORMER,
                                            f"generated: {generated_only_decoded[:200]}",
                                        )
                                    )
                            continue

                        # Try to parse the conformer
                        try:
                            generated_mol = decode_cartesian_v2(conformer_text)
                        except Exception as e:
                            failure_counts[FAIL_PARSING_ERROR] += 1
                            all_failed_generations.append(
                                (smiles, generated_only_decoded, FAIL_PARSING_ERROR)
                            )
                            if len(sample_failures) < 10:
                                sample_failures.append(
                                    (
                                        smiles,
                                        FAIL_PARSING_ERROR,
                                        f"{e}: {conformer_text[:80]}",
                                    )
                                )
                            continue

                        # Check SMILES match (similar to inference.py)
                        generated_smiles = strip_smiles(conformer_text)
                        if not same_molecular_graph(canonical_smiles, generated_smiles):
                            failure_counts[FAIL_SMILES_MISMATCH] += 1
                            all_failed_generations.append(
                                (smiles, generated_only_decoded, FAIL_SMILES_MISMATCH)
                            )
                            if len(sample_failures) < 10:
                                sample_failures.append(
                                    (
                                        smiles,
                                        FAIL_SMILES_MISMATCH,
                                        f"canonical: {canonical_smiles}, got: {generated_smiles}",
                                    )
                                )
                            continue

                        # Compute RMSD statistics
                        gt_confs = self._ground_truths[smiles]
                        min_val, max_val, avg_val, rmsd_vec = self._compute_rmsd_stats(
                            generated_mol, gt_confs
                        )

                        if np.isnan(min_val):
                            failure_counts[FAIL_RMSD_NAN] += 1
                            all_failed_generations.append(
                                (smiles, generated_only_decoded, FAIL_RMSD_NAN)
                            )
                            if len(sample_failures) < 10:
                                sample_failures.append(
                                    (smiles, FAIL_RMSD_NAN, "RMSD returned NaN")
                                )
                        else:
                            min_rmsds.append(min_val)
                            max_rmsds.append(max_val)
                            avg_rmsds.append(avg_val)

                            # Store per-molecule RMSD vector for covmat-style metrics
                            if rmsd_vec.size > 0:
                                per_smiles_rmsd_vectors.setdefault(smiles, []).append(
                                    rmsd_vec
                                )

                        # Progress logging (per-rank)
                        global_sample_idx = batch_start + batch_idx + 1
                        if global_sample_idx % 5 == 0 and rank == 0:
                            total_failures = sum(failure_counts.values())
                            logger.info(
                                f"[rank {rank}] Numerical validation progress: "
                                f"{global_sample_idx}/{rank_prompt_count}, "
                                f"successes={len(min_rmsds)}, failures={total_failures}"
                            )

        finally:
            if was_training:
                model.train()

        # Compute covmat metrics lists (computed once, used for both single-GPU and DDP aggregation)
        cov_r_list: List[np.ndarray] = []
        cov_p_list: List[np.ndarray] = []
        mat_r_list: List[float] = []
        mat_p_list: List[float] = []

        if per_smiles_rmsd_vectors:
            thresholds = DEFAULT_THRESHOLDS
            for smiles, rmsd_vecs in per_smiles_rmsd_vectors.items():
                if not rmsd_vecs:
                    continue
                try:
                    rmsd_matrix = np.stack(rmsd_vecs, axis=1)
                except ValueError:
                    continue
                if np.isnan(rmsd_matrix).all():
                    continue
                cov_r, mat_r, cov_p, mat_p = covmat_metrics(rmsd_matrix, thresholds)
                cov_r_list.append(cov_r)
                cov_p_list.append(cov_p)
                mat_r_list.append(mat_r)
                mat_p_list.append(mat_p)

        # Compute final covmat metrics based on world_size
        cov_r_mean = cov_p_mean = mat_r_mean = mat_p_mean = None
        cov_r_by_threshold = cov_p_by_threshold = None
        if cov_r_list:
            if world_size == 1:
                # Single-GPU case: compute means directly (matching evaluation pipeline)
                cov_r_matrix = np.vstack(cov_r_list)  # shape: (num_molecules, num_thresholds)
                cov_p_matrix = np.vstack(cov_p_list)
                if cov_r_matrix.size > 0:
                    cov_r_mean = float(np.mean(cov_r_matrix))
                    cov_r_by_threshold = np.mean(cov_r_matrix, axis=0).tolist()
                if cov_p_matrix.size > 0:
                    cov_p_mean = float(np.mean(cov_p_matrix))
                    cov_p_by_threshold = np.mean(cov_p_matrix, axis=0).tolist()
                mat_r_mean = float(np.mean(mat_r_list)) if mat_r_list else None
                mat_p_mean = float(np.mean(mat_p_list)) if mat_p_list else None

        log_min_rmsds = min_rmsds
        log_max_rmsds = max_rmsds
        log_avg_rmsds = avg_rmsds
        global_logging_kwargs = {}

        # MINIMAL DDP COMMUNICATION - Avoid OOM and heavy network traffic
        if world_size > 1:
            # Compute all metrics locally on each rank - NO LARGE DATA TRANSFERS

            # 1. Compute final RMSD metrics locally
            valid_min = np.array(min_rmsds, dtype=float) if min_rmsds else np.array([])
            valid_max = np.array(max_rmsds, dtype=float) if max_rmsds else np.array([])
            valid_avg = np.array(avg_rmsds, dtype=float) if avg_rmsds else np.array([])

            if valid_min.size > 0:
                valid_min = valid_min[~np.isnan(valid_min)]
            if valid_max.size > 0:
                valid_max = valid_max[~np.isnan(valid_max)]
            if valid_avg.size > 0:
                valid_avg = valid_avg[~np.isnan(valid_avg)]

            # 2. Prepare covmat metrics for DDP aggregation (reuse already computed lists)
            num_thresholds = len(DEFAULT_THRESHOLDS)
            cov_r_sum_array = np.zeros(num_thresholds, dtype=np.float64)
            cov_p_sum_array = np.zeros(num_thresholds, dtype=np.float64)
            mat_r_sum_value = float(np.sum(mat_r_list)) if mat_r_list else 0.0
            mat_p_sum_value = float(np.sum(mat_p_list)) if mat_p_list else 0.0
            cov_mol_count = 0
            mat_count = len(mat_r_list)
            if cov_r_list:  # cov_r_list was already computed above
                # Stack coverage arrays to compute proper aggregation (matching evaluation code)
                cov_r_matrix = np.vstack(cov_r_list)  # shape: (num_molecules, num_thresholds)
                cov_p_matrix = np.vstack(cov_p_list)

                cov_r_sum_array = np.sum(cov_r_matrix, axis=0)
                cov_p_sum_array = np.sum(cov_p_matrix, axis=0)
                cov_mol_count = cov_r_matrix.shape[0]
                mat_count = len(mat_r_list)

            cov_r_tensor = torch.tensor(cov_r_sum_array, dtype=torch.float32, device=device)
            cov_p_tensor = torch.tensor(cov_p_sum_array, dtype=torch.float32, device=device)

            # 3. Prepare minimal tensors for reduction (only essential scalars)
            base_stats = torch.tensor([
                # Basic counts
                len(valid_min),  # successes
                sum(failure_counts.values()),  # total failures
                # RMSD stats (send sums for proper weighted averaging)
                float(np.sum(valid_min)) if valid_min.size > 0 else 0.0,
                float(np.sum(valid_max)) if valid_max.size > 0 else 0.0,
                float(np.sum(valid_avg)) if valid_avg.size > 0 else 0.0,
                # Matching metrics
                mat_r_sum_value,
                mat_p_sum_value,
                float(cov_mol_count),  # number of molecules contributing to coverage
                float(mat_count),      # number of molecules contributing to matching
            ], dtype=torch.float32, device=device)

            local_stats = torch.cat([base_stats, cov_r_tensor, cov_p_tensor])

            # 4. Reduce to rank 0 with SUM operation (works for counts and means)
            dist.reduce(local_stats, dst=0, op=dist.ReduceOp.SUM)

            base_len = 9
            cov_r_offset = base_len
            cov_p_offset = base_len + num_thresholds
            base_values = local_stats[:base_len]
            cov_r_sum_tensor = local_stats[cov_r_offset:cov_r_offset + num_thresholds]
            cov_p_sum_tensor = local_stats[cov_p_offset:cov_p_offset + num_thresholds]

            # 5. Handle global min/max using reduce for better scaling
            # Each rank computes its local min/max of RMSD minimums
            local_min_of_mins = float(np.min(valid_min)) if valid_min.size > 0 else float("inf")
            local_max_of_mins = float(np.max(valid_min)) if valid_min.size > 0 else float("-inf")

            # Reduce to rank 0 using MIN and MAX operations
            global_min_tensor = torch.tensor([local_min_of_mins], dtype=torch.float32, device=device)
            global_max_tensor = torch.tensor([local_max_of_mins], dtype=torch.float32, device=device)

            dist.reduce(global_min_tensor, dst=0, op=dist.ReduceOp.MIN)
            dist.reduce(global_max_tensor, dst=0, op=dist.ReduceOp.MAX)

            # 6. Reduce failure counts
            failure_tensor = torch.tensor([
                failure_counts[FAIL_NO_CLOSING_TAG],
                failure_counts[FAIL_EMPTY_CONFORMER],
                failure_counts[FAIL_PARSING_ERROR],
                failure_counts[FAIL_SMILES_MISMATCH],
                failure_counts[FAIL_RMSD_NAN],
            ], dtype=torch.long, device=device)
            dist.reduce(failure_tensor, dst=0, op=dist.ReduceOp.SUM)

            # 7. For sample failures: gather only 1 example per rank (minimal)
            # Use rank 0's samples only to avoid unnecessary communication
            if rank == 0:
                gathered_samples = [sample_failures[:1] if sample_failures else []]
            else:
                gathered_samples = [[]]

            # 8. For failed generations: use file-based approach (no network transfer)
            if self.config.validation.save_failed_generations and all_failed_generations:
                rank_file = self.output_dir / "numerical_validation" / f"failures_rank_{rank}.txt"
                rank_file.parent.mkdir(parents=True, exist_ok=True)
                with open(rank_file, "w") as f:
                    for i, (smiles, gen_text, fail_type) in enumerate(all_failed_generations):  # Save ALL failures
                        f.write(f"{i+1}. {smiles} | {fail_type} | {gen_text}\n")  # Save FULL text

            # 9. Only rank 0 processes final results
            if rank != 0:
                return {}

            # Rank 0: extract aggregated results
            total_successes = int(base_values[0].item())
            total_failures = int(base_values[1].item())

            # Extract global min/max from reduced tensors
            min_value = global_min_tensor.item()
            max_value = global_max_tensor.item()
            global_min_of_mins = float(min_value) if math.isfinite(min_value) else math.nan
            global_max_of_mins = float(max_value) if math.isfinite(max_value) else math.nan

            # Compute weighted averages for RMSD metrics
            if total_successes > 0:
                avg_rmsd_min_mean = base_values[2].item() / total_successes
                avg_rmsd_max_mean = base_values[3].item() / total_successes
                avg_rmsd_avg_mean = base_values[4].item() / total_successes
            else:
                avg_rmsd_min_mean = avg_rmsd_max_mean = avg_rmsd_avg_mean = float("nan")

            # Covmat aggregation - compute per-threshold means before flattening
            total_cov_molecules = int(base_values[7].item())
            total_mat_molecules = int(base_values[8].item())

            if total_cov_molecules > 0:
                cov_r_by_threshold = (cov_r_sum_tensor / total_cov_molecules).cpu().tolist()
                cov_p_by_threshold = (cov_p_sum_tensor / total_cov_molecules).cpu().tolist()
                cov_r_mean = float(np.mean(cov_r_by_threshold))
                cov_p_mean = float(np.mean(cov_p_by_threshold))
            else:
                cov_r_mean = cov_p_mean = None
                cov_r_by_threshold = cov_p_by_threshold = None

            if total_mat_molecules > 0:
                mat_r_mean = base_values[5].item() / total_mat_molecules
                mat_p_mean = base_values[6].item() / total_mat_molecules
            else:
                mat_r_mean = mat_p_mean = None

            # Reconstruct data structures for _log_validation_metrics
            min_rmsds = [avg_rmsd_min_mean] if total_successes > 0 else []
            max_rmsds = [avg_rmsd_max_mean] if total_successes > 0 else []
            avg_rmsds = [avg_rmsd_avg_mean] if total_successes > 0 else []

            failure_counts = {
                FAIL_NO_CLOSING_TAG: int(failure_tensor[0].item()),
                FAIL_EMPTY_CONFORMER: int(failure_tensor[1].item()),
                FAIL_PARSING_ERROR: int(failure_tensor[2].item()),
                FAIL_SMILES_MISMATCH: int(failure_tensor[3].item()),
                FAIL_RMSD_NAN: int(failure_tensor[4].item()),

            }

            # Collect limited sample failures (only rank 0 has data)
            sample_failures = gathered_samples[0][:5] if gathered_samples and gathered_samples[0] else []

            # Read failed generations from rank files
            all_failed_generations = []
            if self.config.validation.save_failed_generations:
                for r in range(world_size):
                    rank_file = self.output_dir / "numerical_validation" / f"failures_rank_{r}.txt"
                    if rank_file.exists():
                        try:
                            with open(rank_file, "r") as f:
                                content = f.read().strip()
                                if content:
                                    # Convert back to tuple format for compatibility
                                    all_failed_generations.append((f"rank_{r}_failures", content, "file_aggregated"))
                        except Exception:
                            pass

            def _finite_or_none(value: float | None) -> float | None:
                if value is None:
                    return None
                return value if math.isfinite(value) else None

            log_min_rmsds = []  # Already aggregated in global stats
            log_max_rmsds = []
            log_avg_rmsds = []
            global_logging_kwargs = {
                "global_rmsd_min_min": _finite_or_none(global_min_of_mins),
                "global_rmsd_min_max": _finite_or_none(global_max_of_mins),
                "global_rmsd_min_mean": _finite_or_none(avg_rmsd_min_mean),
                "global_rmsd_min_std": None,
                "global_rmsd_max_mean": _finite_or_none(avg_rmsd_max_mean),
                "global_rmsd_avg_mean": _finite_or_none(avg_rmsd_avg_mean),
                "global_successes": total_successes,
            }

        # Covmat metrics are computed above (both single-GPU and DDP cases)

        metrics = self._log_validation_metrics(
            log_min_rmsds,
            log_max_rmsds,
            log_avg_rmsds,
            failure_counts,
            step,
            sample_failures,
            cov_r_mean=cov_r_mean,
            cov_p_mean=cov_p_mean,
            mat_r_mean=mat_r_mean,
            mat_p_mean=mat_p_mean,
            cov_r_by_threshold=cov_r_by_threshold,
            cov_p_by_threshold=cov_p_by_threshold,
            **global_logging_kwargs,
        )

        if all_failed_generations and self.config.validation.save_failed_generations:
            self._save_failed_generations(all_failed_generations, step)

        # Log timing and statistics
        validation_end_time = time.time()
        validation_duration = validation_end_time - validation_start_time

        # Count total conformers generated (before sharding in DDP case)
        total_conformers_generated = total_prompts_before_sharding
        num_unique_smiles = len(valid_prompts)

        if rank == 0:
            logger.info(
                f"Numerical validation completed in {validation_duration:.2f}s: "
                f"{num_unique_smiles} unique SMILES, "
                f"{total_conformers_generated} conformers generated"
            )

        return metrics

