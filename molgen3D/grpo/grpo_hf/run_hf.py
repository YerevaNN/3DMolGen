from typing import List
import argparse
from collections import namedtuple
import os
from pathlib import Path
from ast import literal_eval
import yaml

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM,  PreTrainedTokenizerBase
from datasets import Dataset, Features, Value
from trl import GRPOConfig
from molgen3D.grpo.grpo_hf.grpo_trainer import GRPOTrainer
# from utils.mol import find_valid_mols

from loguru import logger
import re
from molgen3D.evaluation.utils import extract_between


def search(args):
    logger.info(f"Running GRPO")
    device = "cuda:0"

    training_args = GRPOConfig(
        output_dir=args.grpo.output_dir,
        save_strategy="no",
        max_completion_length=args.generation.max_completion_length,
        # eos_token_id=args.processing.eos_token_id,
        learning_rate=args.grpo.learning_rate,
        lr_scheduler_type=args.grpo.scheduler,
        adam_beta1=args.grpo.adam_beta1,
        adam_beta2=args.grpo.adam_beta2,
        weight_decay=args.grpo.weight_decay,
        warmup_ratio=args.grpo.warmup_ratio,
        max_grad_norm=args.grpo.max_grad_norm,
        temperature=args.grpo.temperature,
        num_train_epochs=args.grpo.num_epochs,
        num_generations=args.grpo.num_generations,
        beta=args.grpo.beta,
        per_device_train_batch_size=args.grpo.batch_size,
        gradient_accumulation_steps=args.grpo.grad_acc_steps,
        log_on_each_node=False,
        bf16=True,
        report_to="wandb",
        run_name="my_grpo_experiment",
        logging_steps=1,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=None
        # attn_implementation="sdpa"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.tokenizer_path,
        # padding_side="left"
    )


    dataset = Dataset.from_csv(args.dataset.dataset_path)

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


    tokenizer.pad_token_id = tokenizer.get_vocab()[args.model.pad_token]
    mol_end_token_id = tokenizer.get_vocab()[args.model.mol_tags[1]]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_function,
        ],
        args=training_args,
        train_dataset=dataset,
        # mol_end_token_id=mol_end_token_id
    )
    trainer.train()


class ExperimentArgs:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    cmd_args = parser.parse_args()
    args_dict = yaml.safe_load(open(cmd_args.config, "r"))
    # print(args_dict)
    args = ExperimentArgs()

    # add the arguments to the args
    for k, v in args_dict.items():
        setattr(args, k, namedtuple(k.title(), v.keys())(**v))
    search(args)