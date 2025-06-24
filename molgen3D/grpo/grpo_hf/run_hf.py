# Standard library imports
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import submitit
import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig as TRLGRPOConfig

# Local imports
from molgen3D.grpo.grpo_hf.config import Config
from molgen3D.grpo.grpo_hf.grpo_trainer import GRPOTrainer
from molgen3D.grpo.grpo_hf.stats import RunStatistics
from molgen3D.grpo.grpo_hf.utils import (
    load_smiles_mapping,
    setup_logging,
    create_code_snapshot,
    save_config
)
from molgen3D.grpo.grpo_hf.rewards import reward_function

def main(config: Config, enable_wandb: bool = False, output_dir: str = None):
    # Setup logging if output_dir is provided
    setup_logging(output_dir)
    
    logger.info(f"Running GRPO")
    device = "cuda:0"

    # Load SMILES mapping and set GEOM data path
    load_smiles_mapping(config.dataset.smiles_mapping_path)
    logger.info("Initialized data paths")

    # Initialize statistics object
    stats = RunStatistics(output_dir=output_dir)

    training_args = TRLGRPOConfig(
        output_dir=output_dir,
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
        bf16=True,
        report_to="wandb" if enable_wandb else "none",
        run_name=config.run.name,
        logging_steps=1,
        # max_steps=config.grpo.max_steps,
        num_train_epochs=config.grpo.num_epochs,
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


    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(config.model.pad_token)
    mol_end_token_id = tokenizer.convert_tokens_to_ids(config.model.mol_tags[1])

    # Wrap the reward function to inject stats, tokenizer, and config
    def reward_func(prompts, completions, **kwargs):
        return reward_function(prompts, completions, stats, tokenizer, config)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()

    # After training
    stats.update_stats()

    # Save model and tokenizer
    model_dir = Path(output_dir) / "model"
    model_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(model_dir)
    logger.info(f"Saved model to {model_dir}")


def main_job(config: Config, enable_wandb: bool = False, output_dir: str = None, work_dir: str = None):
    # Convert output_dir to absolute path before changing working directory
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        
    if work_dir:
        os.chdir(work_dir)
        # Add the snapshot directory to Python path for imports
        sys.path.insert(0, work_dir)
        logger.info(f"Running from code snapshot: {work_dir}")
        logger.info(f"Logs and results will be saved to: {output_dir}")
    main(config, enable_wandb, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], help="Device type: local, a100, h100")
    parser.add_argument("--ngpus", type=int, help="Number of GPUs to use")
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(cmd_args.config)

    # Override config with CLI args if provided
    device_type = cmd_args.device if cmd_args.device is not None else config.device.device_type
    num_gpus = cmd_args.ngpus if cmd_args.ngpus is not None else config.device.num_gpus
    config.device.device_type = device_type
    config.device.num_gpus = num_gpus

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    output_dir = os.path.abspath(os.path.join(config.grpo.output_dir, f"{timestamp}_{config.run.name}"))
    os.makedirs(output_dir, exist_ok=True)

    # Save updated config to output directory
    save_config(config, output_dir)

    # Run locally or with SLURM based on device_type
    if device_type == "local":
        logger.info("Running locally (no SLURM)")
        main_job(config=config, enable_wandb=cmd_args.wandb, output_dir=output_dir, work_dir=None)
    else:
        # Create code snapshot
        snapshot_dir = os.path.join(output_dir, "code_snapshot")
        os.makedirs(snapshot_dir, exist_ok=True)
        create_code_snapshot(os.getcwd(), snapshot_dir)
        
        executor = submitit.AutoExecutor(folder="~/slurm_jobs/grpo/job_%j")
        node = device_type  # 'a100' or 'h100'
        n_gpus = num_gpus
        executor.update_parameters(
            name=config.run.name,
            timeout_min=24 * 24 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=n_gpus * 18,
            slurm_additional_parameters={"partition": node},
        )
        
        # Submit job to SLURM with the snapshot directory
        job = executor.submit(
            main_job,
            config=config,
            enable_wandb=cmd_args.wandb,
            output_dir=output_dir,
            work_dir=snapshot_dir
        )
        
        logger.info(f"Job submitted with ID: {job.job_id}")
        logger.info(f"Code snapshot created at: {snapshot_dir}")