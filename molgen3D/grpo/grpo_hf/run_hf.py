from typing import List, Dict, Any
import argparse
from collections import namedtuple
import os
from pathlib import Path
from ast import literal_eval
import yaml
import submitit
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM,  PreTrainedTokenizerBase
from datasets import Dataset, Features, Value
from trl import GRPOConfig as TRLGRPOConfig
from molgen3D.grpo.grpo_hf.grpo_trainer import GRPOTrainer
# from utils.mol import find_valid_mols

from loguru import logger
import re
from molgen3D.evaluation.utils import extract_between
from molgen3D.grpo.grpo_hf.config import Config


def main(config: Config, enable_wandb: bool = False):
    logger.info(f"Running GRPO")
    device = "cuda:0"

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
        num_train_epochs=config.grpo.num_epochs,
        num_generations=config.grpo.num_generations,
        beta=config.grpo.beta,
        per_device_train_batch_size=config.grpo.batch_size,
        gradient_accumulation_steps=config.grpo.grad_acc_steps,
        log_on_each_node=False,
        bf16=True,
        report_to="wandb" if enable_wandb else "none",
        run_name=config.run.name,
        logging_steps=1,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_path,
    )

    dataset = Dataset.from_csv(config.dataset.dataset_path)

    def reward_function(completions, **kwargs):
        tag_pattern = re.compile(r'<[^>]*>')
        rewards = []
        for prompt, completion in zip(kwargs["prompts"], completions):
            generated_conformer = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
    
            if generated_conformer:
                generated_smiles = tag_pattern.sub('', generated_conformer)
                rewards.append(1.0 - abs(len(extract_between(prompt, "[SMILES]", "[/SMILES]")) - len(generated_smiles)))
                print(f"Prompt: {prompt}\nCompletion: {generated_smiles}\nReward: {rewards[-1]}")
            else:
                rewards.append(0.0)
                print(f"Prompt: {prompt}\nCompletion: {generated_conformer}\nReward: {rewards[-1]}")
        print('end of batch ----------******************----------------------')
        return rewards

    tokenizer.pad_token_id = tokenizer.get_vocab()[config.model.pad_token]
    mol_end_token_id = tokenizer.get_vocab()[config.model.mol_tags[1]]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_function,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(cmd_args.config)

    # Setup submitit executor
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/grpo/job_%j")
    node = "a100"
    n_gpus = 1
    
    executor.update_parameters(
        name=config.run.name,
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 8,
        slurm_additional_parameters={"partition": node},
    )

    # Submit job to SLURM
    job = executor.submit(main, config=config, enable_wandb=cmd_args.wandb)

    # main(config=config, enable_wandb=cmd_args.wandb)  # Run the main function immediately for local testing