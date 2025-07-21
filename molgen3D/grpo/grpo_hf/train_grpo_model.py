#!/usr/bin/env python3
# Standard library imports
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Third-party imports
import submitit
import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig as TRLGRPOConfig
from accelerate import InitProcessGroupKwargs

# Local imports
from molgen3D.grpo.grpo_hf.config import Config
from molgen3D.grpo.grpo_hf.grpo_trainer import GRPOTrainer
from molgen3D.grpo.grpo_hf.stats import RunStatistics
from molgen3D.grpo.grpo_hf.utils import (
    load_smiles_mapping,
    setup_logging
)
from molgen3D.grpo.grpo_hf.rewards import reward_function


def main(config: Config, enable_wandb: bool = False):
    logger.info(f"Running GRPO")
    setup_logging(config.grpo.output_dir, config.run.log_level)
    stats = RunStatistics(output_dir=config.grpo.output_dir)
    load_smiles_mapping(config.dataset.smiles_mapping_path)
    logger.info("Initialized data paths")
    

    training_args = TRLGRPOConfig(
        output_dir=config.grpo.output_dir,
        save_strategy="no",
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
        log_on_each_node=False,
        report_to="wandb" if enable_wandb else "none",
        run_name=config.run.name,
        logging_steps=1,
        max_steps=config.grpo.max_steps,
        # num_train_epochs=config.grpo.num_epochs,
        use_liger_loss=True,
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_path,
    )
    dataset = Dataset.from_csv(config.dataset.dataset_path)

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(config.model.pad_token)
    mol_end_token_id = tokenizer.convert_tokens_to_ids(config.model.mol_tags[1])

    def reward_func(prompts, completions, **kwargs):
        return reward_function(prompts, completions, stats, tokenizer, config)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    stats.update_stats()

    model_dir = Path(config.grpo.output_dir) / "model"
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
    
