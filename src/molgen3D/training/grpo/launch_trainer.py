import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repository's src directory is importable even when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import submitit
import yaml
from loguru import logger

from molgen3D.training.grpo.config import (
    Config,
    DataLoaderConfig,
    DatasetConfig,
    DeviceConfig,
    GenerationConfig,
    GRPOConfig,
    ModelConfig,
    ProcessingConfig,
    RunConfig,
    TrainerConfig,
)
from molgen3D.training.grpo.utils import create_code_snapshot
def ensure_snapshot_has_grpo_code(snapshot_dir: Path) -> None:
    """Verify that the GRPO training entry script exists inside the snapshot."""
    required_path = snapshot_dir / "molgen3D" / "training" / "grpo" / "train_grpo_model.py"
    if not required_path.exists():
        raise FileNotFoundError(
            f"Snapshot is missing GRPO training code at {required_path}. "
            "Please ensure create_code_snapshot completed successfully."
        )


def setup_launcher_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm}</green> | <level>{level: <4}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

def get_project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[4]


def build_launch_command(args, work_dir: str | None = None, accelerate_config_path: str | None = None, config_path: str | None = None) -> list[str]:
    python_executable = sys.executable
    cmd = [python_executable]
    script_path = Path("molgen3D/training/grpo/train_grpo_model.py")
    config_file_path = Path(config_path) if config_path else Path(args.config)
    if args.strategy != "single":
        cmd += ["-m", "accelerate.commands.launch", "--config_file", str(accelerate_config_path)]
    cmd += [str(script_path), "--config", str(config_file_path)]
    if args.wandb:
        cmd.append("--wandb")
    
    return cmd

def load_and_update_configs(args, project_root: Path):
    """Load and update configuration files with command line overrides."""
    # Load base config
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Override device settings from command line - ALWAYS override
    if "device" in config_data:
        config_data["device"]["device_type"] = args.device
        config_data["device"]["num_gpus"] = args.ngpus
        logger.info(f"Overriding device config: {args.device} with {args.ngpus} GPUs")
    
    # Write updated config back to file immediately
    with open(args.config, "w") as f:
        yaml.safe_dump(config_data, f)
    logger.info(f"Updated config file with CLI overrides: {args.device} with {args.ngpus} GPUs")

    accelerate_config_path = None
    if args.strategy != "single":
        accelerate_config_path = project_root / "src" / "molgen3D" / "config" / "grpo" / f"{args.strategy}.conf"
        if not accelerate_config_path.exists():
            raise FileNotFoundError(f"Accelerate config not found for strategy '{args.strategy}': {accelerate_config_path}")
        with open(accelerate_config_path, "r") as f:
            accelerate_config = yaml.safe_load(f)
        accelerate_config["num_processes"] = args.ngpus
        with open(accelerate_config_path, "w") as f:
            yaml.safe_dump(accelerate_config, f)
    
    return config_data, accelerate_config_path

def create_directories(config_data, args, project_root: Path):
    """Create output and checkpoint directories using updated config data."""
    # Create config object from the updated config_data (not from YAML file)
    config = Config(
        model=ModelConfig(**config_data['model']),
        generation=GenerationConfig(**config_data['generation']),
        processing=ProcessingConfig(**config_data['processing']),
        grpo=GRPOConfig(**config_data['grpo']),
        dataset=DatasetConfig(**config_data['dataset']),
        run=RunConfig(**config_data['run']),
        device=DeviceConfig(**config_data['device']),
        trainer=TrainerConfig(**config_data.get('trainer', {})),
        dataloader=DataLoaderConfig(**config_data.get('dataloader', {}))
    )
    
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    
    grpo_outputs_base = project_root / "outputs" / "grpo_outputs"
    grpo_outputs_base.mkdir(parents=True, exist_ok=True)
    config.grpo.output_base_dir = str(grpo_outputs_base)
    
    output_dir = grpo_outputs_base / f"{timestamp}_{config.run.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.grpo.output_dir = str(output_dir)
    
    # Create checkpoint directory (absolute path)
    checkpoint_dir = os.path.abspath(os.path.join(config.grpo.checkpoint_base_dir, f"{timestamp}_{config.run.name}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.grpo.checkpoint_dir = str(checkpoint_dir)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Using device: {config.device.device_type} with {config.device.num_gpus} GPUs")
    
    return config, output_dir


def copy_grpo_config(config_path: Path, snapshot_dir: Path) -> Path:
    """Copy the selected GRPO config into the snapshot directory."""
    destination_dir = snapshot_dir / "molgen3D" / "config" / "grpo"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / config_path.name
    shutil.copy2(config_path, destination_path)
    return destination_path

def setup_job_executor(device_type, num_gpus, run_name):
    """Setup the appropriate job executor based on device type."""
    job_folder_path = Path("outputs") / "slurm_jobs" / "grpo"
    job_folder_path.mkdir(parents=True, exist_ok=True)
    job_folder = str(job_folder_path / "job_%j")
    
    if device_type == "local":
        executor = submitit.LocalExecutor(folder=job_folder)
        logger.info("Using LocalExecutor for local execution")
    else:
        executor = submitit.AutoExecutor(folder=job_folder)
        logger.info(f"Using AutoExecutor for SLURM submission to {device_type} partition")
    
    # Configure job parameters - use GRES instead of gpus_per_node for specific GPU types
    if device_type in ["a100", "h100", "all"]:
        # Use GRES for specific GPU type requests
        executor.update_parameters(
            name=run_name,
            timeout_min=24 * 24 * 60,  # 24 hours
            nodes=1,
            mem_gb=80,
            cpus_per_task=num_gpus * 8,
            slurm_additional_parameters={
                "partition": device_type,
                "gres": f"gpu:{num_gpus}"
            },
        )
    else:
        # Use gpus_per_node for other cases
        executor.update_parameters(
            name=run_name,
            timeout_min=24 * 24 * 60,  # 24 hours
            gpus_per_node=num_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=num_gpus * 8,
            slurm_additional_parameters={"partition": device_type},
        )
    
    logger.info(f"Job configuration:")
    logger.info(f"  - Partition: {device_type}")
    logger.info(f"  - GPUs: {num_gpus}")
    if device_type in ["a100", "h100"]:
        logger.info(f"  - GRES: gpu:{device_type}:{num_gpus}")
    else:
        logger.info(f"  - GPUs per node: {num_gpus}")
    
    return executor

def update_config_in_place(config, config_file_path):
    """Update the YAML config file in place with runtime paths."""
    # Load the existing YAML file
    with open(config_file_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Update only the runtime paths, preserving the base paths
    if 'grpo' in config_data:
        config_data['grpo']['output_dir'] = config.grpo.output_dir
        config_data['grpo']['checkpoint_dir'] = config.grpo.checkpoint_dir
    
    # Write back to the same file
    with open(config_file_path, "w") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

def save_updated_config(config, config_file_path):
    """Save the updated configuration to file."""
    with open(config_file_path, "w") as f:
        yaml.safe_dump({
            'model': config.model.__dict__,
            'generation': config.generation.__dict__,
            'processing': config.processing.__dict__,
            'grpo': config.grpo.__dict__,
            'dataset': config.dataset.__dict__,
            'run': config.run.__dict__,
            'device': config.device.__dict__,
            'trainer': config.trainer.__dict__
        }, f)

def main():
    setup_launcher_logging()
    project_root = get_project_root()
    # Ensure snapshot jobs resolve data paths against the real repo root
    os.environ.setdefault("MOLGEN3D_REPO_ROOT", str(project_root))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch GRPO training job")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML configuration file")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100", "all"], 
                       default="local", help="Target device type for job submission")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--strategy", type=str, choices=["single", "ddp", "fsdp", "ds"], 
                       default="single", help="Training strategy")
    parser.add_argument("--nccl-debug", action="store_true", help="Enable NCCL debug logging")
    
    args = parser.parse_args()

    try:
        # Load and update configurations
        config_data, accelerate_config_path = load_and_update_configs(args, project_root)
        
        # Create directories and get config object
        config, output_dir = create_directories(config_data, args, project_root)
        
        # Create code snapshot
        create_code_snapshot(str(project_root), str(output_dir))
        ensure_snapshot_has_grpo_code(output_dir)
        logger.info(f"Code snapshot created in: {output_dir}")
        
        # Update the config file that was copied in the snapshot
        snapshot_config_path = copy_grpo_config(Path(args.config), output_dir)
        logger.info(f"Updating snapshot config: {snapshot_config_path}")
        
        # Update the copied config file in place with runtime paths
        update_config_in_place(config, str(snapshot_config_path))
        logger.info(f"Updated snapshot config in place with runtime paths")
        
        # Build launch command using snapshot config
        cmd = build_launch_command(args, work_dir=str(output_dir), accelerate_config_path=accelerate_config_path, config_path=str(snapshot_config_path))
        logger.info(f"Launch command: {' '.join(cmd)}")
        
        # Setup and submit job
        executor = setup_job_executor(args.device, config.device.num_gpus, config.run.name)
        job = executor.submit(submitit.helpers.CommandFunction(cmd, cwd=str(output_dir)))
        
        logger.info(f"Job submitted successfully!")
        logger.info(f"Job ID: {job.job_id}")
        logger.info(f"Target device: {args.device}")
        logger.info(f"Number of GPUs: {config.device.num_gpus}")
        logger.info(f"Training strategy: {args.strategy}")
        
    except Exception as e:
        logger.error(f"Failed to launch training: {str(e)}")
        raise

if __name__ == "__main__":
    main()