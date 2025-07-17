import argparse
from datetime import datetime
import os
import sys
from pathlib import Path
import submitit
from loguru import logger
from molgen3D.grpo.grpo_hf.config import Config
from molgen3D.grpo.grpo_hf.utils import create_code_snapshot
import yaml
import tempfile

def setup_launcher_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm}</green> | <level>{level: <4}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

def update_accelerate_config(strategy: str, ngpus: int, device: str) -> str:
    config_path = Path(f"molgen3D/grpo/grpo_hf/configs/{strategy}.conf")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["num_processes"] = ngpus
    config["main_process_gpu"] = 0
    config["gpu_ids"] = list(range(ngpus))
    if device in ["a100", "h100"]:
        config["machine_rank"] = 0
        config["main_training_function"] = "main"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".conf") as tmp:
        yaml.safe_dump(config, tmp)
        return tmp.name

def build_launch_command(args, work_dir: str | None = None, accelerate_config_path: str | None = None) -> list[str]:
    python_executable = sys.executable
    cmd = [python_executable]
    script_path = Path("molgen3D/grpo/grpo_hf/train_grpo_model.py")
    config_path = Path(args.config)
    if args.strategy != "single":
        cmd += ["-m", "accelerate.commands.launch", "--config_file", str(accelerate_config_path)]
    cmd += [str(script_path), "--config", str(config_path)]
    if args.wandb:
        cmd.append("--wandb")
    return cmd

def is_main_process() -> bool:
    env_vars = {
        "LOCAL_RANK": "0",
        "RANK": "0",
        "SLURM_PROCID": "0"
    }
    return all(os.environ.get(key, "0") == val for key, val in env_vars.items())

def main():
    setup_launcher_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], default="local", help="Device type")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--strategy", type=str, choices=["single", "ddp", "fsdp", "ds"], default="single", help="Training strategy")
    args = parser.parse_args()
    try:
        # Update general config file in-place
        with open(args.config, "r") as f:
            general_config_data = yaml.safe_load(f)
        if "device" in general_config_data:
            if args.device:
                general_config_data["device"]["device_type"] = args.device
            if args.ngpus:
                general_config_data["device"]["num_gpus"] = args.ngpus
        with open(args.config, "w") as f:
            yaml.safe_dump(general_config_data, f)

        # Update accelerate config file in-place
        accelerate_config_path = Path(f"molgen3D/grpo/grpo_hf/configs/{args.strategy}.conf")
        with open(accelerate_config_path, "r") as f:
            accelerate_config_data = yaml.safe_load(f)
        accelerate_config_data["num_processes"] = args.ngpus
        with open(accelerate_config_path, "w") as f:
            yaml.safe_dump(accelerate_config_data, f)

        config = Config.from_yaml(args.config)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
        output_dir = Path(os.path.abspath(os.path.join(config.grpo.output_dir, f"{timestamp}_{config.run.name}")))
        output_dir.mkdir(parents=True, exist_ok=True)
        config.grpo.output_dir = str(output_dir)
        logger.info(f"Output directory: {output_dir}")
        snapshot_dir = output_dir / "code_snapshot"
        create_code_snapshot(os.getcwd(), str(snapshot_dir))
        logger.info(f"Code snapshot created at: {snapshot_dir}")
        cmd = build_launch_command(args, work_dir=str(snapshot_dir), accelerate_config_path=accelerate_config_path)
        logger.info(f"Submitting command: {' '.join(cmd)}")
        if args.device == "local":
            executor = submitit.LocalExecutor(folder=str(Path.home() / "slurm_jobs/grpo/job_%j"))
        else:
            executor = submitit.AutoExecutor(folder=str(Path.home() / "slurm_jobs/grpo/job_%j"))
        executor.update_parameters(
            name=config.run.name,
            timeout_min=24 * 24 * 60,
            gpus_per_node=config.device.num_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=config.device.num_gpus * 22,
            slurm_additional_parameters={"partition": config.device.device_type,},
        )
        job = executor.submit(submitit.helpers.CommandFunction(cmd, cwd=str(snapshot_dir)))
        logger.info(f"Job submitted with ID: {job.job_id}")
    except Exception as e:
        logger.error(f"Failed to launch training: {str(e)}")
        raise

if __name__ == "__main__":
    main()