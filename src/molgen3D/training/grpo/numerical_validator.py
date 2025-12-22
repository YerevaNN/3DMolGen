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
    ) -> Dict[str, float]:
        """Log numerical validation metrics."""
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
        if valid_min.size > 0:
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
            # Get validation batch size from config
            validation_batch_size = getattr(self.config.validation, "validation_batch_size", 8)

            # Determine if sampling is enabled for validation
            gen_cfg = sampling_configs[self.config.validation.sampling_config]
            sampling_enabled = bool(getattr(gen_cfg, "do_sample", False))

            # Prepare all valid prompts.
            # If sampling is enabled, we generate approximately 2k conformers per molecule,
            # where k is the number of ground-truth conformers for that molecule.
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

            # Shard work across DDP ranks, if any
            total_prompts_before_sharding = len(prepared_prompts)
            if world_size > 1:
                sharded_prompts: List[torch.Tensor] = []
                sharded_smiles: List[str] = []
                for idx, (p, s) in enumerate(zip(prepared_prompts, prepared_smiles)):
                    if idx % world_size == rank:
                        sharded_prompts.append(p)
                        sharded_smiles.append(s)
                prepared_prompts = sharded_prompts
                prepared_smiles = sharded_smiles

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

        # Gather results across ranks if running under DDP.
        if world_size > 1:
            gathered: List[Dict] = [None] * world_size  # type: ignore[assignment]
            local_payload = {
                "min_rmsds": min_rmsds,
                "max_rmsds": max_rmsds,
                "avg_rmsds": avg_rmsds,
                "failure_counts": failure_counts,
                "per_smiles_rmsd_vectors": per_smiles_rmsd_vectors,
                "all_failed_generations": all_failed_generations,
                "sample_failures": sample_failures,
            }
            dist.all_gather_object(gathered, local_payload)

            # Only rank 0 will aggregate and return metrics; others return empty dict
            if rank != 0:
                return {}

            # Merge payloads from all ranks
            min_rmsds = []
            max_rmsds = []
            avg_rmsds = []
            per_smiles_rmsd_vectors = {}
            all_failed_generations = []
            sample_failures = []
            failure_counts = {
                FAIL_NO_CLOSING_TAG: 0,
                FAIL_EMPTY_CONFORMER: 0,
                FAIL_PARSING_ERROR: 0,
                FAIL_SMILES_MISMATCH: 0,
                FAIL_RMSD_NAN: 0,
                FAIL_NO_GROUND_TRUTH: skipped_no_gt,
            }

            for payload in gathered:
                if not payload:
                    continue
                min_rmsds.extend(payload.get("min_rmsds", []))
                max_rmsds.extend(payload.get("max_rmsds", []))
                avg_rmsds.extend(payload.get("avg_rmsds", []))
                for k, v_list in payload.get("per_smiles_rmsd_vectors", {}).items():
                    per_smiles_rmsd_vectors.setdefault(k, []).extend(v_list)
                all_failed_generations.extend(payload.get("all_failed_generations", []))
                # Keep only a small sample of failures for logging
                if len(sample_failures) < 10:
                    remaining = 10 - len(sample_failures)
                    sample_failures.extend(payload.get("sample_failures", [])[:remaining])
                # Sum failure counts
                for ft, cnt in payload.get("failure_counts", {}).items():
                    failure_counts[ft] = failure_counts.get(ft, 0) + cnt

        # Compute covmat-style metrics (COV-R, COV-P, MAT-R, MAT-P) using all
        # valid RMSD vectors, matching the evaluation pipeline behaviour.
        cov_r_mean = cov_p_mean = mat_r_mean = mat_p_mean = None
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
                    rmsd_matrix = np.stack(rmsd_vecs, axis=1)  # (n_true, n_gen)
                except ValueError:
                    # Skip molecules with inconsistent RMSD vectors
                    continue

                if np.isnan(rmsd_matrix).all():
                    continue

                cov_r, mat_r, cov_p, mat_p = covmat_metrics(rmsd_matrix, thresholds)
                cov_r_list.append(cov_r)
                cov_p_list.append(cov_p)
                mat_r_list.append(mat_r)
                mat_p_list.append(mat_p)

            if cov_r_list:
                cov_r_all = np.vstack(cov_r_list)
                cov_p_all = np.vstack(cov_p_list)
                mat_r_all = np.array(mat_r_list, dtype=float)
                mat_p_all = np.array(mat_p_list, dtype=float)

                cov_r_mean = float(np.mean(cov_r_all))
                cov_p_mean = float(np.mean(cov_p_all))
                mat_r_mean = float(np.mean(mat_r_all))
                mat_p_mean = float(np.mean(mat_p_all))

        metrics = self._log_validation_metrics(
            min_rmsds,
            max_rmsds,
            avg_rmsds,
            failure_counts,
            step,
            sample_failures,
            cov_r_mean=cov_r_mean,
            cov_p_mean=cov_p_mean,
            mat_r_mean=mat_r_mean,
            mat_p_mean=mat_p_mean,
        )

        if all_failed_generations and self.config.validation.save_failed_generations:
            self._save_failed_generations(all_failed_generations, step)

        return metrics

