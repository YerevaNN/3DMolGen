# Standard library imports
import re
import os

# Third-party imports
import numpy as np
from loguru import logger
import wandb

# Local imports
from molgen3D.evaluation.utils import extract_between
from molgen3D.grpo.grpo_hf.utils import get_rmsd, load_ground_truths

# Global variables
_smiles_mapping = None
_geom_data_path = None

def get_rmsd_reward(ground_truths, generated_conformer, config):
    rmsd_value, min_index = get_rmsd(ground_truths, generated_conformer, align=False)
    if rmsd_value is None or np.isnan(rmsd_value):
        logger.info(f"\n None RMSD value for prompt: {ground_truths[min_index]} {generated_conformer}")
        rmsd_reward = 0.0
    else:
        rmsd_reward = 1.0 / (1.0 + (rmsd_value / config.grpo.rmsd_const))
    return rmsd_value, rmsd_reward, min_index
        
def get_match_reward(generated_smiles, canoncial_smiles, len_prompt):
    if generated_smiles == canoncial_smiles:
        return 1.0
    else:
        match_len = sum(
            1 for c1, c2 in zip(canoncial_smiles, generated_smiles) if c1 == c2
        )
        return match_len / len_prompt if len_prompt > 0 else 0.0
    
def reward_function(prompts, completions, stats, tokenizer, config):

    w_rmsd, w_match = config.grpo.reward_weight_rmsd, config.grpo.reward_weight_match
    norm = w_rmsd + w_match
    w_rmsd, w_match = w_rmsd / norm, w_match / norm

    rmsd_rewards, match_rewards, combined_rewards, rmsd_values = [], [], [], []

    tag_pattern = re.compile(r'<[^>]*>')
    
    ground_truth_cache = {}

    for prompt, completion in zip(prompts, completions):
        stats.processed_prompts += 1
        canoncial_smiles = extract_between(prompt, "[SMILES]", "[/SMILES]")
        len_prompt = len(canoncial_smiles)
        if canoncial_smiles in ground_truth_cache:
            ground_truths = ground_truth_cache[canoncial_smiles]
        else:
            ground_truths = load_ground_truths(canoncial_smiles, num_gt=config.grpo.max_ground_truths)
            ground_truth_cache[canoncial_smiles] = ground_truths
            if ground_truths:
                stats.distinct_prompts += 1
        rmsd_reward, match_reward, combined, rmsd_value, min_index = 0.0, 0.0, 0.0, float('nan'), -2
        generated_conformer = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
        generated_smiles = tag_pattern.sub('', generated_conformer) if generated_conformer else ""
        if not ground_truths:
            stats.failed_ground_truth += 1
        elif not generated_conformer:
            stats.failed_conformer_generation += 1
        elif generated_smiles != canoncial_smiles:
            stats.failed_matching_smiles += 1
            logger.info(f"\nGenerated SMILES does not match canonical prompt: {prompt}\nSMILES: {generated_smiles}")
        else:
            rmsd_value, rmsd_reward, min_index = get_rmsd_reward(ground_truths, generated_conformer, config)
            if np.isnan(rmsd_value):
                stats.failed_rmsd += 1
            else:
                stats.successful_generations += 1
                stats.add_rmsd(rmsd_value)
                rmsd_rewards.append(rmsd_reward)
                rmsd_values.append(rmsd_value)

            if wandb.run is not None:
                wandb.log({
                    "rmsd_value": rmsd_value,
                    "min_index": min_index,
                })

        match_reward = get_match_reward(generated_smiles, canoncial_smiles, len_prompt)
        match_rewards.append(match_reward)
        combined = w_rmsd * rmsd_reward + w_match * match_reward
        combined_rewards.append(combined)
            
        completion_tokens = len(tokenizer.encode(generated_conformer)) if generated_conformer else 0
        # logger.info(f"[PID {os.getpid()}] Prompt: {prompt}"
        #             # f"Completion: {completion}" +
        #             f"\nRewards-  RMSD reward: {rmsd_reward:.2f}, RMSD value: {rmsd_value}, index: {min_index}" +
        #             f" Match: {match_reward:.4f}, Combined: {combined:.4f}, Length completion: {completion_tokens} tokens\n")
        
    aggregate_stats = stats.update_stats()
    if wandb.run is not None:
        log_data = {
            "reward/rmsd": float(np.nanmean(rmsd_rewards)) if rmsd_rewards else 0.0,
            "reward/rmsd_std": float(np.nanstd(rmsd_rewards)) if rmsd_rewards else 0.0,
            "reward/match": float(np.nanmean(match_rewards)) if match_rewards else 0.0,
            "reward/combined": float(np.nanmean(combined_rewards)) if combined_rewards else 0.0,
            "stats/success_rate": aggregate_stats.get("success_rate", 0.0),
            "failure_rate/ground_truth": aggregate_stats.get("failed_ground_truth", 0.0),
            "failure_rate/conformer_generation": aggregate_stats.get("failed_conformer_generation", 0.0),
            "failure_rate/matching_smiles": aggregate_stats.get("failed_matching_smiles", 0.0),
            "failure_rate/rmsd": aggregate_stats.get("failed_rmsd", 0.0),
        }
        wandb.log(log_data)
    logger.info(f"[PID {os.getpid()}] {'='*40}\n")

    return combined_rewards 