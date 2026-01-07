#!/usr/bin/env python3
# Standard library imports
import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Ensure the snapshot/repo sources take precedence over any installed package.
SCRIPT_PATH = Path(__file__).resolve()
package_container = None
for parent in SCRIPT_PATH.parents:
    if (parent / "molgen3D").is_dir():
        package_container = parent
        break
if package_container and str(package_container) not in sys.path:
    sys.path.insert(0, str(package_container))

# Third-party imports
import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer
from trl.trainer import grpo_trainer as trl_grpo_module

# Local imports
from molgen3D.training.grpo.config import Config
from molgen3D.training.grpo.stats import RunStatistics
from molgen3D.training.grpo.utils import (
    load_smiles_mapping,
    setup_logging,
    save_config,
    get_torch_dtype
)
from molgen3D.training.grpo.rewards import reward_function
from molgen3D.training.grpo.numerical_validator import GRPONumericalValidator
from molgen3D.training.grpo.numerical_validation_callback import NumericalValidationCallback
from molgen3D.training.grpo.grpo_reward_v3 import reward_function as reward_function_v3


def initialize_random_seed(seed: int) -> None:
    """Seed all RNGs so the data order and sampling stays consistent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_completion_length_tracking():
    """Register a runtime hook so TRL always exposes completion lengths."""
    if getattr(trl_grpo_module, "_molgen3d_completion_length_hook", False):
        return

    original_generate = GRPOTrainer._generate

    def _generate_with_lengths(self, prompts):
        result = original_generate(self, prompts)
        _, completion_ids, tool_mask, *_ = result
        device = self.accelerator.device

        if tool_mask is not None:
            lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)

        trl_grpo_module.completion_lengths = lengths
        return result

    GRPOTrainer._generate = _generate_with_lengths
    trl_grpo_module._molgen3d_completion_length_hook = True


def main(config: Config, enable_wandb: bool = False, output_dir: str = None):
    initialize_random_seed(config.grpo.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.trainer.tokenizers_parallelism else "false"

    # Set up output and checkpoint directories if not already configured
    if output_dir is None and config.grpo.output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d-%H%M")
        output_base = Path(config.grpo.output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        output_dir = output_base / f"{timestamp}_{config.run.name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config.grpo.output_dir = str(output_dir)

        # Create checkpoint directory
        checkpoint_dir = os.path.join(config.grpo.checkpoint_base_dir, f"{timestamp}_{config.run.name}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        config.grpo.checkpoint_dir = checkpoint_dir

    actual_output_dir = output_dir or config.grpo.output_dir
    setup_logging(actual_output_dir, config.run.log_level)
    
    logger.info(f"Running GRPO")

    # Load SMILES mapping and set GEOM data path
    load_smiles_mapping(config.dataset.smiles_mapping_path)
    logger.info("Initialized data paths")

    # Initialize statistics object
    stats = RunStatistics(output_dir=actual_output_dir)
    

    training_args = TRLGRPOConfig(
        output_dir=config.grpo.checkpoint_dir,
        save_strategy=config.trainer.save_strategy,
        save_steps=config.trainer.save_steps,
        save_total_limit=config.trainer.save_total_limit,
        save_on_each_node=config.trainer.save_on_each_node,
        save_safetensors=config.trainer.save_safetensors,
        max_completion_length=config.generation.max_completion_length,
        learning_rate=config.grpo.learning_rate,
        lr_scheduler_type=config.grpo.scheduler,
        adam_beta1=config.grpo.adam_beta1,
        adam_beta2=config.grpo.adam_beta2,
        weight_decay=config.grpo.weight_decay,
        warmup_ratio=config.grpo.warmup_ratio,
        max_grad_norm=config.grpo.max_grad_norm,
        temperature=config.grpo.temperature,
        num_generations=config.grpo.num_generations,
        beta=config.grpo.beta,
        per_device_train_batch_size=config.grpo.per_device_batch_size,
        gradient_accumulation_steps=config.grpo.grad_acc_steps,
        log_on_each_node=config.trainer.log_on_each_node,
        report_to="wandb" if enable_wandb else "none",
        run_name=config.run.name,
        logging_steps=config.trainer.logging_steps,
        max_steps=config.grpo.max_steps,
        # num_train_epochs=config.grpo.num_epochs,
        use_liger_kernel=config.trainer.use_liger_loss,
        loss_type=config.trainer.loss_type,
        num_iterations=config.grpo.num_iterations,
        importance_sampling_level=config.grpo.importance_sampling_level,
        steps_per_generation=config.grpo.steps_per_generation,
        seed=config.grpo.seed
    )
    scale_rewards = getattr(config.grpo, "scale_rewards", None)
    if isinstance(scale_rewards, str):
        scale_rewards_value = scale_rewards.lower()
    else:
        scale_rewards_value = scale_rewards
    if scale_rewards_value is not None:
        setattr(training_args, "scale_rewards", scale_rewards_value)
        logger.info("Set TRL scale_rewards=%s", scale_rewards_value)

    # Convert string dtype to torch dtype
    torch_dtype = get_torch_dtype(config.trainer.torch_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.checkpoint_path,
        dtype=torch_dtype,
        attn_implementation=config.trainer.attn_implementation,
    )   
    # Verify Flash Attention is active
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Model attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")    

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_path,
        padding_side="left",
    )

    # Load dataset from text file and create prompt column
    with open(config.dataset.dataset_path, 'r', encoding='utf-8', errors='replace') as f:
        prompts = [
            line.strip()
            for line in f
            if line.strip() and len(line.strip()) <= 150
        ]
    dataset = Dataset.from_dict({"prompt": prompts})
    dataset = dataset.shuffle(seed=config.grpo.seed)
    logger.info(f"Loaded {len(dataset)} prompts from {config.dataset.dataset_path}")

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(config.model.pad_token)
    mol_end_token_id = tokenizer.convert_tokens_to_ids(config.model.mol_tags[1])
    
    # Explicitly set pad_token_id in model config to prevent warnings
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Set model pad_token_id to {model.config.pad_token_id}")

    reward_strategy = config.grpo.reward_strategy.lower()

    if reward_strategy == "v3":
        logger.info("Using GRPO reward function v3 (GEOM-Drugs aligned: quality + smooth coverage + hard matching)")
        def reward_func(prompts, completions, **kwargs):
            completion_entropies = kwargs.get("mean_token_entropy")
            completion_lengths = kwargs.get("completion_lengths")
            return reward_function_v3(
                prompts,
                completions,
                stats,
                tokenizer,
                config,
                completion_entropies=completion_entropies,
                completion_lengths=completion_lengths,
            )

    else:
        if reward_strategy != "legacy":
            raise ValueError(
                f"Unsupported reward_strategy '{reward_strategy}'. "
                "Supported values: 'v3', 'legacy'."
            )

        logger.info("Using legacy reward function")

        def reward_func(prompts, completions, **kwargs):
            return reward_function(prompts, completions, stats, tokenizer, config)

    ensure_completion_length_tracking()

    # Set DataLoader parameters from YAML config
    training_args.dataloader_num_workers = config.dataloader.num_workers
    training_args.dataloader_pin_memory = config.dataloader.pin_memory
    training_args.dataloader_persistent_workers = config.dataloader.persistent_workers
    training_args.dataloader_prefetch_factor = config.dataloader.prefetch_factor
    training_args.dataloader_drop_last = config.dataloader.drop_last

    numerical_callback = None
    if config.validation.enable_numerical_validation:
        numerical_validator = GRPONumericalValidator(
            config=config,
            tokenizer=tokenizer,
            stats=stats,
            output_dir=actual_output_dir,
        )
        logger.info("Numerical validation enabled")

        # Run validation every eval_steps if configured, otherwise every 1000 steps
        validation_interval = config.validation.eval_steps or 1000
        numerical_callback = NumericalValidationCallback(
            validator=numerical_validator,
            stats=stats,
            validation_steps=validation_interval,
            max_seq_len=config.generation.max_completion_length
            + config.validation.max_conformer_tokens,
        )
        logger.info(
            f"Numerical validation callback created (interval: {validation_interval} steps)"
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        callbacks=[numerical_callback] if numerical_callback is not None else None,
    )

    
    # Set epsilon parameters on trainer (not available in GRPOConfig)
    epsilon_low = float(config.grpo.epsilon_low)
    epsilon_high = float(config.grpo.epsilon_high)
    trainer.epsilon_low = epsilon_low
    trainer.epsilon_high = epsilon_high
    logger.info(f"Set epsilon_low={epsilon_low}, epsilon_high={epsilon_high} on GRPO trainer")
    
    trainer.train()

    stats.update_stats()

    model_dir = Path(config.grpo.checkpoint_dir) / "model"
    model_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(model_dir)
    logger.info(f"Saved model to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    main(config, args.wandb)
    
