# Standard library imports
import re
from itertools import takewhile

# Third-party imports
import numpy as np
from loguru import logger
import wandb

# Local imports
from molgen3D.evaluation.utils import extract_between
from molgen3D.grpo.grpo_hf.utils import get_rmsd, load_ground_truths

def calculate_single_rmsd_reward(processed, config, stats):
    """Calculate RMSD-based reward for a processed generation.
    
    Args:
        processed (dict): Dictionary containing processed generation data including:
            - generated_conformer: The generated conformer string
            - generated_smiles: The generated SMILES string
            - canoncial_smiles: The canonical SMILES string
            - ground_truth: The ground truth conformer
            - prompt: The input prompt
            - completion: The generated completion
        config: Configuration object containing reward parameters
        stats: Statistics object for tracking metrics
        
    Returns:
        tuple: (reward, rmsd_value) where:
            - reward: The calculated reward value (float)
            - rmsd_value: The actual RMSD value (float or None)
    """
    # Check if we have a valid generated conformer and matching SMILES
    if not processed["generated_conformer"] or processed["generated_smiles"] != processed["canoncial_smiles"]:
        logger.info(f"\n    Generated SMILES does not match canonical SMILES for prompt: {processed['prompt']}\n completion: {processed['completion']}")
        stats.failed_matching_smiles += 1
        return 0.0, None
        
    # Check if we have ground truth data
    if not processed["ground_truth"]:
        logger.info(f"\nNo ground truth available for prompt: {processed['prompt']}\n completion: {processed['completion']}")
        stats.failed_ground_truth += 1
        return 0.0, None
        
    # Calculate RMSD
    try:
        rmsd = get_rmsd(processed["ground_truth"], processed["generated_conformer"], align=False)
        
        # Validate RMSD value
        if rmsd is None or np.isnan(rmsd):
            logger.info(f"\nInvalid RMSD value for prompt: {processed['prompt']}\n completion: {processed['completion']}")
            stats.failed_rmsd += 1
            return 0.0, None
            
        # Calculate reward using the RMSD value
        reward = 1.0 / (1.0 + rmsd / config.grpo.rmsd_const)
        
        # Track successful RMSD calculation
        stats.add_rmsd(rmsd)
        return reward, rmsd
        
    except Exception as e:
        logger.info(f"\nError calculating RMSD: {e}")
        stats.failed_rmsd += 1
        return 0.0, None

def calculate_single_match_reward(processed, stats):
    """Calculate SMILES matching reward for a processed generation."""
    if processed["generated_conformer"]:
        if processed["generated_smiles"] == processed["canoncial_smiles"]:
            return 1.0
        else:
            match_len = sum(
                1 for c1, c2 in zip(processed["canoncial_smiles"], processed["generated_smiles"]) if c1 == c2
            )
            return match_len / processed["len_prompt"] if processed["len_prompt"] > 0 else 0.0
    else:
        logger.info(f"\nNo valid generated conformer found for prompt: {processed['prompt']}\n completion: {processed['completion']}")
        stats.failed_conformer_generation += 1
        return 0.0

def weighted_joint_reward_function(prompts, completions, stats, tokenizer, config):
    """
    Processes prompts and completions, computes RMSD and match rewards, combines them with config weights,
    logs to wandb, and updates stats. Returns a list of combined rewards.
    """
    w_rmsd = config.grpo.reward_weight_rmsd
    w_match = config.grpo.reward_weight_match
    norm = w_rmsd + w_match
    w_rmsd /= norm
    w_match /= norm

    rmsd_rewards = []
    match_rewards = []
    combined_rewards = []
    rmsd_values = []

    tag_pattern = re.compile(r'<[^>]*>')

    for prompt, completion in zip(prompts, completions):
        stats.processed_prompts += 1
        canoncial_smiles = extract_between(prompt, "[SMILES]", "[/SMILES]")
        len_prompt = len(canoncial_smiles)
        try:
            ground_truths = load_ground_truths(canoncial_smiles, num_gt=1)
            ground_truth = ground_truths[0] if ground_truths else None
            if ground_truth is None:
                stats.failed_ground_truth += 1
        except Exception as e:
            logger.exception(f"Error loading ground truth for {canoncial_smiles}: {e}")
            ground_truth = None
            stats.failed_ground_truth += 1
        generated_conformer = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
        if not generated_conformer:
            stats.failed_conformer_generation += 1
        generated_smiles = tag_pattern.sub('', generated_conformer)
        if generated_smiles != canoncial_smiles:
            stats.failed_matching_smiles += 1
            logger.info(f"\nGenerated SMILES does not match canonical prompt: {prompt}\nSMILES: {generated_smiles}")
        else:
            stats.successful_generations += 1
        if not generated_conformer or generated_smiles != canoncial_smiles or not ground_truth:
            rmsd_reward = 0.0
            rmsd_value = float('nan')
            if not generated_conformer:
                stats.failed_conformer_generation += 1
            if not ground_truth:
                stats.failed_ground_truth += 1
            if generated_smiles != canoncial_smiles:
                stats.failed_matching_smiles += 1
        else:
            try:
                rmsd_value = get_rmsd(ground_truth, generated_conformer, align=False)
                if rmsd_value is None or np.isnan(rmsd_value):
                    stats.failed_rmsd += 1
                    rmsd_reward = 0.0
                    rmsd_value = float('nan')
                else:
                    rmsd_reward = 1.0 / (1.0 + rmsd_value / config.grpo.rmsd_const)
                    stats.add_rmsd(rmsd_value)
            except Exception as e:
                logger.info(f"\nError calculating RMSD: {e}")
                stats.failed_rmsd += 1
                rmsd_reward = 0.0
                rmsd_value = float('nan')
        rmsd_rewards.append(rmsd_reward)
        rmsd_values.append(rmsd_value)
        if generated_conformer:
            if generated_smiles == canoncial_smiles:
                match_reward = 1.0
            else:
                match_len = sum(
                    1 for c1, c2 in zip(canoncial_smiles, generated_smiles) if c1 == c2
                )
                match_reward = match_len / len_prompt if len_prompt > 0 else 0.0
        else:
            match_reward = 0.0
            stats.failed_conformer_generation += 1
        match_rewards.append(match_reward)
        combined = w_rmsd * rmsd_reward + w_match * match_reward
        combined_rewards.append(combined)
        completion_tokens = len(tokenizer.encode(completion))
        logger.info(f"\nPrompt: {prompt}\nCompletion: {completion}" +
                    f"\nRewards-  RMSD reward: {rmsd_reward:.2f}, RMSD value: {rmsd_value}," +
                    f" Match: {match_reward:.4f}, Combined: {combined:.4f}, Length completion: {completion_tokens} tokens\n")
    if wandb.run is not None:
        wandb.log({
            "reward/rmsd": float(np.nanmean(rmsd_rewards)) if rmsd_rewards else 0.0,
            "reward/match": float(np.nanmean(match_rewards)) if match_rewards else 0.0,
            "reward/combined": float(np.nanmean(combined_rewards)) if combined_rewards else 0.0,
            "reward/rmsd_raw": float(np.nanmean(rmsd_values)) if rmsd_values else 0.0,
            "reward/rmsd_raw_std": float(np.nanstd(rmsd_values)) if rmsd_values else 0.0,
        })
    logger.info(f"{'='*40}\n")
    return combined_rewards 