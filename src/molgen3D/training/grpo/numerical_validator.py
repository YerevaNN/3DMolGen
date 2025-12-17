"""
Numerical validation for GRPO training.

This module implements numerical validation for GRPO runs, similar to the
pretraining numerical validator but adapted for the GRPO training setup.
It performs actual conformer generation during validation and computes
RMSD metrics against ground truth conformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.training.grpo.config import Config
from molgen3D.training.grpo.stats import RunStatistics
from molgen3D.training.grpo.utils import load_ground_truths, load_smiles_mapping
from molgen3D.utils.utils import get_best_rmsd

# Failure type constants
FAIL_NO_CLOSING_TAG = "no_closing_tag"
FAIL_EMPTY_CONFORMER = "empty_conformer"
FAIL_PARSING_ERROR = "parsing_error"
FAIL_SMILES_MISMATCH = "smiles_mismatch"
FAIL_RMSD_NAN = "rmsd_nan"
FAIL_NO_GROUND_TRUTH = "no_ground_truth"


class GRPONumericalValidator:
    """
    Numerical validator for GRPO training.

    Performs actual conformer generation during validation and computes RMSD
    metrics against ground truth conformers to validate model performance.
    """

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        stats: RunStatistics,
        output_dir: str
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

        # Load validation data
        self._smiles_mapping = None
        self._validation_prompts = None
        self._ground_truths = None

        # Token IDs for conformer extraction
        self._conformer_start_id = self.tokenizer.convert_tokens_to_ids("[CONFORMER]")
        self._conformer_end_id = self.tokenizer.convert_tokens_to_ids("[/CONFORMER]")
        self._eos_id = self.tokenizer.eos_token_id
        self._pad_id = self.tokenizer.pad_token_id

    def _load_validation_data(self) -> bool:
        """Load SMILES mapping and ground truths for validation."""
        try:
            # Load SMILES mapping if not already loaded
            if self._smiles_mapping is None:
                self._smiles_mapping = load_smiles_mapping(self.config.dataset.smiles_mapping_path)

            # Load validation prompts (sample from dataset)
            if self._validation_prompts is None:
                self._validation_prompts = self._load_validation_prompts()

            # Load ground truths
            if self._ground_truths is None:
                self._ground_truths = {}
                for smiles in self._validation_prompts:
                    ground_truths = load_ground_truths(smiles, num_gt=self.config.grpo.max_ground_truths)
                    if ground_truths:
                        self._ground_truths[smiles] = ground_truths

            return bool(self._validation_prompts and self._ground_truths)

        except Exception as e:
            logger.warning(f"Failed to load validation data: {e}")
            return False

    def _load_validation_prompts(self) -> List[str]:
        """Load a sample of SMILES strings for validation."""
        try:
            from datasets import Dataset
            dataset = Dataset.from_csv(self.config.dataset.dataset_path)

            # Sample prompts for validation
            num_samples = min(self.config.validation.num_val_molecules, len(dataset))
            indices = np.random.choice(len(dataset), num_samples, replace=False)

            prompts = []
            for idx in indices:
                row = dataset[int(idx)]
                # Extract SMILES from the prompt format
                smiles = extract_between(row['prompt'], "[SMILES]", "[/SMILES]")
                if smiles:
                    prompts.append(smiles.strip())

            logger.info(f"Loaded {len(prompts)} validation prompts")
            return prompts

        except Exception as e:
            logger.warning(f"Failed to load validation prompts: {e}")
            return []

    def _build_prompt_tensor(
        self, smiles: str, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Build a prompt tensor for conformer generation."""
        try:
            # Create prompt in the expected format
            prompt_text = f"[SMILES]{smiles}[/SMILES][CONFORMER]"
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            return torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)
        except Exception as e:
            logger.warning(f"Failed to build prompt tensor for {smiles}: {e}")
            return None


    def _extract_conformer_text(self, token_tensor: torch.Tensor) -> str:
        """Extract conformer text directly from token tensor."""
        tokens = token_tensor[0].tolist() if token_tensor.dim() == 2 else token_tensor.tolist()
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)

        # Try [CONFORMER] first (primary format), fall back to [CONFORMERS]
        conformer = extract_between(decoded, "[CONFORMER]", "[/CONFORMER]")
        if not conformer:
            conformer = extract_between(decoded, "[CONFORMERS]", "[/CONFORMERS]")
        return conformer.strip() if conformer else ""

    def _create_stopping_criteria(self, device: torch.device):
        """Create stopping criteria for generation."""
        end_token_ids = []
        if self._conformer_end_id is not None:
            end_token_ids.append(self._conformer_end_id)
        # Don't include EOS in stopping criteria - we want to generate complete conformers

        if end_token_ids:
            return StoppingCriteriaList([_ConformerStoppingCriteria(end_token_ids)])
        else:
            return None

    def _compute_rmsd_stats(
        self, generated_mol, ground_truths: List
    ) -> Tuple[float, float, float]:
        """Compute RMSD statistics against ground truths."""
        rmsds = []
        for gt in ground_truths:
            try:
                rmsd = float(get_best_rmsd(generated_mol, gt, use_alignmol=False))
                rmsds.append(rmsd)
            except Exception as e:
                logger.debug(f"RMSD calculation failed: {e}")
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

        # Add per-failure-type metrics
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
        except (ModuleNotFoundError, Exception) as e:
            logger.debug(f"Could not log to W&B: {e}")

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

        # Load validation data
        if not self._load_validation_data():
            logger.warning("Numerical validation skipped: no validation data available")
            return {}

        if not self._validation_prompts or not self._ground_truths:
            logger.warning("Numerical validation skipped: validation data not loaded")
            return {}

        # Check if required tokens are available
        if self._conformer_start_id is None or self._conformer_end_id is None:
            logger.warning("Numerical validation skipped: conformer tokens not found in tokenizer")
            return {}

        device = next(model.parameters()).device

        # Filter prompts to those with ground truths
        valid_prompts = [
            smiles for smiles in self._validation_prompts
            if smiles in self._ground_truths
        ]
        skipped_no_gt = len(self._validation_prompts) - len(valid_prompts)

        if not valid_prompts:
            logger.warning("Numerical validation skipped: no prompts with ground truths")
            return {}

        logger.info(f"Starting numerical validation: {len(valid_prompts)} prompts, step={step}")

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
        all_failed_generations: List[Tuple[str, str, str]] = []  # (smiles, full_generated, fail_type)

        was_training = model.training
        model.eval()

        try:
            # Get validation batch size from config
            validation_batch_size = getattr(self.config.validation, 'validation_batch_size', 8)

            # Prepare all valid prompts
            prepared_prompts = []
            prepared_smiles = []

            for i, smiles in enumerate(valid_prompts):
                prompt = self._build_prompt_tensor(smiles, device)
                if prompt is None:
                    failure_counts[FAIL_PARSING_ERROR] += 1
                    if len(sample_failures) < 10:
                        sample_failures.append((smiles, FAIL_PARSING_ERROR, "failed to build prompt"))
                    continue

                # Skip prompts that are too long to avoid wasting space in batching
                if prompt.shape[1] > 500:  # Skip prompts longer than 500 tokens
                    continue

                prepared_prompts.append(prompt)
                prepared_smiles.append(smiles)

            if not prepared_prompts:
                logger.warning("No valid prompts for batched generation")
            else:
                # Process prompts in batches
                for batch_start in range(0, len(prepared_prompts), validation_batch_size):
                    batch_end = min(batch_start + validation_batch_size, len(prepared_prompts))
                    batch_prompts = prepared_prompts[batch_start:batch_end]
                    batch_smiles = prepared_smiles[batch_start:batch_end]

                    logger.info(
                        f"Processing batch {batch_start // validation_batch_size + 1}: "
                        f"{len(batch_prompts)} prompts"
                    )

                    if not batch_prompts:
                        continue

                    # Pad prompts to same length for batching
                    max_prompt_len = max(p.shape[1] for p in batch_prompts)
                    padded_prompts = []
                    attention_masks = []

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
                            attention_mask = torch.cat(
                                [
                                    torch.ones(
                                        prompt.shape[1],
                                        device=device,
                                        dtype=torch.long,
                                    ),
                                    torch.zeros(
                                        pad_len,
                                        device=device,
                                        dtype=torch.long,
                                    ),
                                ],
                                dim=0,
                            )
                        else:
                            padded_prompt = prompt
                            attention_mask = torch.ones(
                                prompt.shape[1], device=device, dtype=torch.long
                            )

                        padded_prompts.append(padded_prompt)
                        attention_masks.append(attention_mask.unsqueeze(0))

                    # Stack into batch
                    input_ids = torch.cat(padded_prompts, dim=0)
                    attention_mask = torch.cat(attention_masks, dim=0)

                    # Calculate max new tokens
                    available = max(max_seq_len - max_prompt_len, 1)
                    max_new_tokens = min(
                        self.config.validation.max_conformer_tokens, available
                    )
                    logger.info(
                        f"Generation limits: max_seq_len={max_seq_len}, "
                        f"max_prompt_len={max_prompt_len}, "
                        f"available={available}, max_new_tokens={max_new_tokens}"
                    )

                    # Generate batch
                    with torch.no_grad():
                        # Set max_length to allow longer conformer generation
                        max_total_length = input_ids.shape[1] + 3500  # up to 3500 new tokens
                        generated_outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=3500,
                            max_length=max_total_length,
                            do_sample=False,  # Greedy decoding
                            pad_token_id=self._pad_id,
                            eos_token_id=None,  # Disable EOS stopping
                            stopping_criteria=None,  # No custom stopping criteria
                        )
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
                        min_val, max_val, avg_val = self._compute_rmsd_stats(
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

                        # Progress logging
                        global_sample_idx = batch_start + batch_idx + 1
                        if global_sample_idx % 5 == 0:
                            total_failures = sum(failure_counts.values())
                            logger.info(
                                f"Numerical validation progress: "
                                f"{global_sample_idx}/{len(prepared_prompts)}, "
                                f"successes={len(min_rmsds)}, failures={total_failures}"
                            )

        finally:
            if was_training:
                model.train()

        # Log metrics and save failed generations
        metrics = self._log_validation_metrics(
            min_rmsds, max_rmsds, avg_rmsds, failure_counts, step, sample_failures
        )

        if all_failed_generations and self.config.validation.save_failed_generations:
            self._save_failed_generations(all_failed_generations, step)

        return metrics


class _ConformerStoppingCriteria(StoppingCriteria):
    """Stopping criteria for conformer generation."""

    def __init__(self, end_token_ids):
        self.end_token_ids = end_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if any sequence has generated an end token in the last position
        last_tokens = input_ids[:, -1]
        return torch.any(torch.isin(last_tokens, torch.tensor(self.end_token_ids, device=input_ids.device)))