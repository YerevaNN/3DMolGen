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
from itertools import takewhile
from transformers import AutoTokenizer, AutoModelForCausalLM,  PreTrainedTokenizerBase
from datasets import Dataset, Features, Value
from trl import GRPOConfig as TRLGRPOConfig
from molgen3D.grpo.grpo_hf.grpo_trainer import GRPOTrainer
from molgen3D.grpo.grpo_hf.utils import load_ground_truths, get_rmsd
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
        rewards, prev_prompt = [], ''
        for prompt, completion in zip(kwargs["prompts"], completions):

            if prompt != prev_prompt or prev_prompt == '':
                canoncial_smiles = extract_between(prompt, "[SMILES]", "[/SMILES]")
                len_prompt = len(canoncial_smiles)
                try:
                    ground_truth = load_ground_truths(canoncial_smiles, num_gt=1)
                    ground_truth = ground_truth[0] if ground_truth else None
                except Exception as e:
                    print(f"Error loading ground truth for {canoncial_smiles}")
                    ground_truth = None
                prev_prompt = prompt

            generated_conformer = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
            generated_smiles = tag_pattern.sub('', generated_conformer)
            if generated_conformer:
                len_completion = len(generated_smiles)
                if generated_smiles == canoncial_smiles:
                    reward_match = 1.0
                    if ground_truth:
                        try:
                            rmsd = get_rmsd(ground_truth, generated_conformer, align=False)
                            reward_rmsd = 1.0 / (1.0 + rmsd / 2.0)
                        except Exception as e:
                            print(f"Error calculating RMSD for prompt: {prompt} completion: {completion}\n{e}")
                            reward_rmsd = 0.0
                    else:
                        print(f"No ground truth available for prompt: {prompt}")
                        reward_rmsd = 0.0
                else:
                    print(f"Generated SMILES does not match canonical SMILES for prompt")
                    reward_rmsd = 0.0
                    match_len = sum(1 for c1, c2 in takewhile(lambda pair: pair[0] == pair[1], zip(canoncial_smiles, generated_smiles)))
                    reward_match = match_len / len_prompt
            else:
                print(f"No valid generated conformer found for prompt")
                reward_rmsd = 0.0
                reward_match = 0.0
            rewards.append(reward_rmsd)
            print(f"Prompt: {prompt}\nCompletion: {generated_smiles}\nReward: {rewards[-1]}")
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

    # Save model and tokenizer after training
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


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
    n_gpus = 4
    
    executor.update_parameters(
        name=config.run.name,
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 16,
        slurm_additional_parameters={"partition": node},
    )

    # Submit job to SLURM
    job = executor.submit(main, config=config, enable_wandb=cmd_args.wandb)

    # main(config=config, enable_wandb=cmd_args.wandb)  # Run the main function immediately for local testing