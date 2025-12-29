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
import time

import cloudpickle
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from transformers import StoppingCriteria, StoppingCriteriaList

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
FAIL_NO_GROUND_TRUTH = "no_ground_truth"


class ConformerEndStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that explicitly documents stopping at the [/CONFORMER] token.
    
    Note: Per-sequence stopping is handled by eos_token_id in model.generate().
    This stopping criteria provides explicit documentation of the stopping behavior
    and can be extended for additional stopping logic if needed in the future.
    
    The actual stopping is handled by setting eos_token_id=self._conformer_end_id,
    which stops each sequence independently when it encounters the [/CONFORMER] token.
    """
    
    def __init__(self, conformer_end_token_id: int):
        """
        Initialize the stopping criteria.
        
        Args:
            conformer_end_token_id: Token ID for [/CONFORMER]
        """
        super().__init__()
        self.conformer_end_token_id = conformer_end_token_id
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if generation should stop for the entire batch.
        
        Note: Per-sequence stopping is handled by eos_token_id. This method
        could be extended for batch-level stopping logic if needed.
        
        Args:
            input_ids: Current generated token IDs
            scores: Logits for next token prediction
            
        Returns:
            False - per-sequence stopping is handled by eos_token_id
        """
        # Per-sequence stopping is handled by eos_token_id parameter
        # This stopping criteria is kept for explicit documentation
        return False


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
        rng = np.random.default_rng()
        rng.shuffle(all_keys)

        prompts: List[str] = []
        for key in all_keys:
            # Keys are SMILES strings for which we have ground-truth conformers.
            prompts.append(str(key).strip())
            if len(prompts) >= self.config.validation.num_val_molecules:
                break

        logger.info(f"Loaded {len(prompts)} validation SMILES keys from validation_pickle")
        return prompts

    def _build_prompt_tensor(
        self, smiles: str, device: torch.device
    ) -> torch.Tensor:
        """Build a prompt tensor for conformer generation."""
        prompt_text = f"[SMILES]{smiles}[/SMILES][CONFORMER]"
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)


    def _extract_conformer_text(self, token_tensor: torch.Tensor) -> str:
        """Extract conformer text directly from token tensor."""
        tokens = (
            token_tensor[0].tolist()
            if token_tensor.dim() == 2
            else token_tensor.tolist()
        )
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)

        conformer = extract_between(decoded, "[CONFORMER]", "[/CONFORMER]")

        return conformer.strip() if conformer else ""

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
            f.write("=" * 80 + "\n\n")

            for i, (smiles, full_generated, fail_type) in enumerate(failed_generations, 1):
                f.write(f"--- Failure #{i} ---\n")
                f.write(f"SMILES: {smiles}\n\n")
                f.write(f"Full Generated String:\n{full_generated}\n\n")
                f.write(f"Fail Type: {fail_type}\n")
                f.write("-" * 80 + "\n\n")

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
        if global_successes is not None and global_rmsd_min_min is not None:
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

        # Add covmat-style metrics if available (matching evaluation pipeline)
        if cov_r_mean is not None:
            metrics["numerical_val/COV-R_mean"] = float(cov_r_mean)
        if cov_p_mean is not None:
            metrics["numerical_val/COV-P_mean"] = float(cov_p_mean)
        if mat_r_mean is not None:
            metrics["numerical_val/MAT-R_mean"] = float(mat_r_mean)
        if mat_p_mean is not None:
            metrics["numerical_val/MAT-P_mean"] = float(mat_p_mean)

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
        has_rmsd_metrics = (
            (global_successes is not None and global_rmsd_min_min is not None) or
            (not (global_successes is not None and global_rmsd_min_min is not None) and valid_min.size > 0)
        )
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
            logger.info(f"Sample failures (first {min(3, len(sample_failures))}):")
            for smiles, fail_type, details in sample_failures[:3]:
                logger.info(f"  [{fail_type}] {smiles[:50]}... : {details[:100]}")

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
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
            self._validation_prompts = self._load_validation_prompts()
        if not self._ground_truths:
            gt_path = get_data_path("validation_pickle")
            logger.info(f"Loading numerical validation ground truths from {gt_path}")

            with open(gt_path, "rb") as fh:
                gt_dict = cloudpickle.load(fh)

            for key in self._validation_prompts:
                confs = gt_dict.get(key)
                if confs:
                    self._ground_truths[key] = confs

        device = next(model.parameters()).device

        # Detect DDP / distributed context
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        valid_prompts = [
            smiles for smiles in self._validation_prompts
            if smiles in self._ground_truths
        ]
        skipped_no_gt = len(self._validation_prompts) - len(valid_prompts)

        if not valid_prompts:
            if rank == 0:
                logger.warning("Numerical validation skipped: no prompts with ground truths")
            return {}

        if rank == 0:
            logger.info(
                f"Starting numerical validation: {len(valid_prompts)} prompts, "
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
            FAIL_NO_GROUND_TRUTH: skipped_no_gt,
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

            # Shard molecules first, then expand prompts per rank for better load balancing
            if world_size > 1:
                sharded_smiles_list: List[str] = []
                for idx, smiles in enumerate(valid_prompts):
                    if idx % world_size == rank:
                        sharded_smiles_list.append(smiles)
                valid_prompts = sharded_smiles_list

            # Now expand prompts on each rank
            prepared_prompts: List[torch.Tensor] = []
            prepared_smiles: List[str] = []

            for smiles in valid_prompts:
                prompt = self._build_prompt_tensor(smiles, device)

                # Number of generations to request for this molecule
                if sampling_enabled:
                    num_ground_truths = len(self._ground_truths.get(smiles, []))
                    # Target ~2k conformers per molecule (k = num_ground_truths)
                    num_generations = max(2 * num_ground_truths, 1)
                else:
                    num_generations = 1

                for _ in range(num_generations):
                    prepared_prompts.append(prompt)
                    prepared_smiles.append(smiles)

            total_prompts_before_sharding = len(prepared_prompts)

            if not prepared_prompts:
                # Log for all ranks to ensure all are aware when validation is skipped
                logger.warning(
                    f"[rank {rank}] No valid prompts for batched generation (after sharding). "
                    f"Total prompts before sharding: {total_prompts_before_sharding}, "
                    f"world_size: {world_size}"
                )
            else:
                # Process prompts in batches
                for batch_start in range(0, len(prepared_prompts), validation_batch_size):
                    batch_end = min(batch_start + validation_batch_size, len(prepared_prompts))
                    batch_prompts = prepared_prompts[batch_start:batch_end]
                    batch_smiles = prepared_smiles[batch_start:batch_end]

                    if rank == 0:
                        logger.info(
                            f"[rank {rank}] Processing batch {batch_start // validation_batch_size + 1}: "
                            f"{len(batch_prompts)} prompts"
                        )

                    if not batch_prompts:
                        continue

                    # Pad prompts to same length for batching
                    max_prompt_len = max(p.shape[1] for p in batch_prompts)
                    padded_prompts = []

                    for prompt in batch_prompts:
                        pad_len = max_prompt_len - prompt.shape[1]
                        if pad_len > 0:
                            # Pad with pad_token_id
                            pad_tensor = torch.full(
                                (1, pad_len),
                                self._pad_id or 0,
                                device=device,
                                dtype=torch.long,
                            )
                            padded_prompt = torch.cat([prompt, pad_tensor], dim=1)
                        else:
                            padded_prompt = prompt

                        padded_prompts.append(padded_prompt)

                    # Stack into batch
                    input_ids = torch.cat(padded_prompts, dim=0)
                    attention_mask = (input_ids != (self._pad_id or 0)).long()

                    # Calculate max new tokens
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

                    # Generate batch
                    # Create explicit stopping criteria to stop at [/CONFORMER] token
                    stopping_criteria = StoppingCriteriaList([
                        ConformerEndStoppingCriteria(self._conformer_end_id)
                    ])
                    
                    generated_outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=sampling_configs[self.config.validation.sampling_config].do_sample,
                        top_p=sampling_configs[self.config.validation.sampling_config].top_p,
                        temperature=sampling_configs[self.config.validation.sampling_config].temperature,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self._pad_id,
                        eos_token_id=self._conformer_end_id,  # Stop at [/CONFORMER] token
                        stopping_criteria=stopping_criteria,  # Explicit stopping criteria for clarity
                    )
                    if rank == 0:
                        logger.info(f"Generated outputs shape: {generated_outputs.shape}")

                    # Process each generated sequence in the batch
                    for batch_idx, smiles in enumerate(batch_smiles):
                        # Extract the generated sequence for this sample
                        generated_seq = generated_outputs[batch_idx : batch_idx + 1]
                        # Remove the input prompt part
                        generated_tokens = generated_seq[0, max_prompt_len:]

                        # Decode full generated string for logging
                        tokens = generated_tokens.tolist()
                        full_decoded = self.tokenizer.decode(
                            tokens, skip_special_tokens=False
                        )

                        # Extract conformer text
                        conformer_text = self._extract_conformer_text(
                            generated_tokens.unsqueeze(0)
                        )

                        # Check for empty/missing conformer
                        if not conformer_text:
                            # Check if [/CONFORMER] is missing
                            if (
                                "[/CONFORMER]" not in full_decoded
                                and "[/CONFORMERS]" not in full_decoded
                            ):
                                failure_counts[FAIL_NO_CLOSING_TAG] += 1
                                all_failed_generations.append(
                                    (smiles, full_decoded, FAIL_NO_CLOSING_TAG)
                                )
                                if len(sample_failures) < 10:
                                    # Show what was generated after [CONFORMER]
                                    conf_start_idx = full_decoded.find("[CONFORMER]")
                                    if conf_start_idx >= 0:
                                        gen_part = full_decoded[
                                            conf_start_idx : conf_start_idx + 150
                                        ]
                                    else:
                                        gen_part = full_decoded[-150:]
                                    sample_failures.append(
                                        (
                                            smiles,
                                            FAIL_NO_CLOSING_TAG,
                                            f"generated: {gen_part}",
                                        )
                                    )
                            else:
                                failure_counts[FAIL_EMPTY_CONFORMER] += 1
                                all_failed_generations.append(
                                    (smiles, full_decoded, FAIL_EMPTY_CONFORMER)
                                )
                                if len(sample_failures) < 10:
                                    sample_failures.append(
                                        (
                                            smiles,
                                            FAIL_EMPTY_CONFORMER,
                                            f"decoded: {full_decoded[-100:]}",
                                        )
                                    )
                            continue

                        # Try to parse the conformer
                        try:
                            generated_mol = decode_cartesian_v2(conformer_text)
                        except Exception as e:
                            failure_counts[FAIL_PARSING_ERROR] += 1
                            all_failed_generations.append(
                                (smiles, full_decoded, FAIL_PARSING_ERROR)
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

                        # Check SMILES match
                        generated_smiles = strip_smiles(conformer_text)
                        if not same_molecular_graph(smiles, generated_smiles):
                            failure_counts[FAIL_SMILES_MISMATCH] += 1
                            all_failed_generations.append(
                                (smiles, full_decoded, FAIL_SMILES_MISMATCH)
                            )
                            if len(sample_failures) < 10:
                                sample_failures.append(
                                    (
                                        smiles,
                                        FAIL_SMILES_MISMATCH,
                                        f"got: {generated_smiles}",
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
                                (smiles, full_decoded, FAIL_RMSD_NAN)
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
                                f"{global_sample_idx}/{len(prepared_prompts)}, "
                                f"successes={len(min_rmsds)}, failures={total_failures}"
                            )

        finally:
            if was_training:
                model.train()

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

            # 2. Compute covmat metrics locally
            local_covmat = {}
            if per_smiles_rmsd_vectors:
                cov_r_list: List[np.ndarray] = []
                cov_p_list: List[np.ndarray] = []
                mat_r_list: List[float] = []
                mat_p_list: List[float] = []

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

                if cov_r_list:
                    local_covmat = {
                        "cov_r_sum": float(np.sum(np.vstack(cov_r_list))),
                        "cov_p_sum": float(np.sum(np.vstack(cov_p_list))),
                        "mat_r_sum": float(np.sum(mat_r_list)),
                        "mat_p_sum": float(np.sum(mat_p_list)),
                        "count": len(cov_r_list),
                    }

            # 3. Prepare minimal tensors for reduction (only essential scalars)
            local_stats = torch.tensor([
                # Basic counts
                len(valid_min),  # successes
                sum(failure_counts.values()),  # total failures
                # RMSD stats (send sums for proper weighted averaging)
                float(np.sum(valid_min)) if valid_min.size > 0 else 0.0,
                float(np.sum(valid_max)) if valid_max.size > 0 else 0.0,
                float(np.sum(valid_avg)) if valid_avg.size > 0 else 0.0,
                # Covmat (send sums for proper weighted averaging)
                local_covmat.get("cov_r_sum", 0.0),
                local_covmat.get("cov_p_sum", 0.0),
                local_covmat.get("mat_r_sum", 0.0),
                local_covmat.get("mat_p_sum", 0.0),
                local_covmat.get("count", 0),
            ], dtype=torch.float32, device=device)

            # 4. Reduce to rank 0 with SUM operation (works for counts and means)
            dist.reduce(local_stats, dst=0, op=dist.ReduceOp.SUM)

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
                failure_counts[FAIL_NO_GROUND_TRUTH],
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
                    for i, (smiles, gen_text, fail_type) in enumerate(all_failed_generations[:3]):  # Max 3 per rank
                        f.write(f"{i+1}. {smiles} | {fail_type} | {gen_text[:100]}...\n")

            # 9. Only rank 0 processes final results
            if rank != 0:
                return {}

            # Rank 0: extract aggregated results
            total_successes = int(local_stats[0].item())
            total_failures = int(local_stats[1].item())

            # Extract global min/max from reduced tensors
            global_min_of_mins = float(global_min_tensor.item()) if global_min_tensor.item() != float("inf") else float("nan")
            global_max_of_mins = float(global_max_tensor.item()) if global_max_tensor.item() != float("-inf") else float("nan")

            # Compute weighted averages for RMSD metrics
            total_successes = int(local_stats[0].item())
            if total_successes > 0:
                avg_rmsd_min_mean = local_stats[2].item() / total_successes
                avg_rmsd_max_mean = local_stats[3].item() / total_successes
                avg_rmsd_avg_mean = local_stats[4].item() / total_successes
            else:
                avg_rmsd_min_mean = avg_rmsd_max_mean = avg_rmsd_avg_mean = float("nan")

            # Covmat aggregation (weighted by molecule count)
            total_covmat_count = int(local_stats[9].item())
            if total_covmat_count > 0:
                cov_r_mean = local_stats[5].item() / total_covmat_count
                cov_p_mean = local_stats[6].item() / total_covmat_count
                mat_r_mean = local_stats[7].item() / total_covmat_count
                mat_p_mean = local_stats[8].item() / total_covmat_count
            else:
                cov_r_mean = cov_p_mean = mat_r_mean = mat_p_mean = None

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
                FAIL_NO_GROUND_TRUTH: int(failure_tensor[5].item()),
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

        # Covmat metrics are already computed and aggregated in DDP section above

        # Use pre-computed global statistics for logging (DDP case)
        metrics = self._log_validation_metrics(
            [], [], [],  # Empty arrays since we use pre-computed stats
            failure_counts,
            step,
            sample_failures,
            cov_r_mean=cov_r_mean,
            cov_p_mean=cov_p_mean,
            mat_r_mean=mat_r_mean,
            mat_p_mean=mat_p_mean,
            global_rmsd_min_min=global_min_of_mins if global_min_of_mins != float("nan") else None,
            global_rmsd_min_max=global_max_of_mins if global_max_of_mins != float("nan") else None,
            global_rmsd_min_mean=avg_rmsd_min_mean if not np.isnan(avg_rmsd_min_mean) else None,
            global_rmsd_min_std=None,
            global_rmsd_max_mean=avg_rmsd_max_mean if not np.isnan(avg_rmsd_max_mean) else None,
            global_rmsd_avg_mean=avg_rmsd_avg_mean if not np.isnan(avg_rmsd_avg_mean) else None,
            global_successes=total_successes,
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

