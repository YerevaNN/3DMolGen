#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime
import tempfile

DEFAULT_LOGS_BASE = Path("/home/fgeikyan/tokenizer/logs")
RUNS_BASE = Path(os.environ.get("FSQ_RUNS_DIR", os.getcwd()))


def find_run_log_dir(ckpt_dir_name: str) -> Path:
    preferred = RUNS_BASE / ckpt_dir_name
    if preferred.exists():
        return preferred
    return DEFAULT_LOGS_BASE / ckpt_dir_name

def find_free_gpus(num_gpus=4):
    import torch
    free_gpus = []
    for gpu_id in range(torch.cuda.device_count()):
        try:
            torch.tensor(0, device=torch.device("cuda", gpu_id))
            free_gpus.append(gpu_id)
            if len(free_gpus) == num_gpus:
                break
        except RuntimeError:
            continue
    return free_gpus


def load_saved_train_env(run_log_dir: Path) -> dict:
    env_path = run_log_dir / "train_config.env"
    if not env_path.is_file():
        return {}

    loaded = {}
    for line in env_path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key] = value
    return loaded


def build_eval_env(base_env: dict, *, gpu: int, ckpt_path: Path, eval_logs_dir: Path) -> dict:
    """
    Build environment for eval.py WITHOUT changing eval.py:
      - Sets CUDA_VISIBLE_DEVICES to one GPU (unless SLURM already set it)
      - Sets env vars required by train.py at import time:
          RUN_NAME, LOG_DIR, CKPT_DIR
    """
    ckpt_dir_name = ckpt_path.parent.name

    env = base_env.copy()
    env.update(load_saved_train_env(find_run_log_dir(ckpt_dir_name)))

    # Older runs may not have train_config.env saved. Infer DATA from the run/checkpoint name
    # so eval.py points at the correct val/test root instead of silently falling back.
    if "DATA" not in env:
        if ckpt_dir_name.startswith("full_20pct"):
            env["DATA"] = "full_20pct"
        elif ckpt_dir_name.startswith("full_5pct"):
            env["DATA"] = "full_5pct"
        elif ckpt_dir_name.startswith("full"):
            env["DATA"] = "full"
        elif "nablaDFT" in ckpt_dir_name or ckpt_dir_name.startswith("nabla"):
            env["DATA"] = "nablaDFT"

    # If SLURM allocated GPU, it already set CUDA_VISIBLE_DEVICES - don't override
    # Otherwise, set it to the GPU we found
    if "SLURM_JOB_ID" not in base_env:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Required by train.py (it raises if these are missing)
    env["RUN_NAME"] = ckpt_dir_name
    env["LOG_DIR"]  = str(eval_logs_dir)          # keep eval logs together
    env["CKPT_DIR"] = str(ckpt_path.parent)

    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--nabla", action="store_true",
                        help="Use nabla mode: evaluate on 4 GPUs with nabla datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Specific dataset to evaluate on (only used with --nabla)")
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"],
                        help="If set, only run eval on this split (val or test); "
                             "otherwise eval.py will run both splits.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite final CSV if it already exists (only meaningful if you remove timestamping).")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    ckpt_dir_name = ckpt_path.parent.name
    run_log_dir = find_run_log_dir(ckpt_dir_name)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Put ALL eval-related outputs here
    eval_logs_dir = run_log_dir / "eval_logs"
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Final merged CSV + submit logs all go to eval_logs_dir
    final_csv = eval_logs_dir / f"{ckpt_path.stem}_{timestamp}_eval.csv"
    launcher_log = eval_logs_dir / f"{ckpt_path.stem}_{timestamp}_submit_eval.log"

    def lprint(*a, **k):
        print(*a, **k)
        with open(launcher_log, "a") as lf:
            print(*a, file=lf, **k)

    # NOTE: with timestamped final_csv, this almost never triggers.
    if final_csv.exists() and not args.overwrite:
        lprint(f"⏭️  Skipping (final CSV already exists): {final_csv}")
        sys.exit(0)

    base_env = os.environ.copy()

    if args.nabla:
        if args.dataset:
            num_gpus_needed = 1
            tasks = [
                {"gpu": 0, "dataset": args.dataset},
            ]
        else:
            num_gpus_needed = 4
            tasks = [
                {"gpu": 0, "dataset": "nablaDFT.csv"},
                {"gpu": 1, "dataset": "nablaDFT_conformations.csv"},
                {"gpu": 2, "dataset": "nablaDFT_scaffolds.csv"},
                {"gpu": 3, "dataset": "nablaDFT_structures.csv"},
            ]

        free_gpus = find_free_gpus(num_gpus_needed)
        if len(free_gpus) < num_gpus_needed:
            lprint(f"Error: Need {num_gpus_needed} free GPU(s), but only found {len(free_gpus)}: {free_gpus}")
            sys.exit(1)

        for i, t in enumerate(tasks):
            t["gpu"] = free_gpus[i]

        lprint(f"Launching {len(tasks)} evaluation task(s) for checkpoint: {ckpt_path.name}")
        lprint(f"Using GPUs: {[t['gpu'] for t in tasks]}")
        lprint(f"Eval logs dir: {eval_logs_dir}")
        lprint(f"Final merged CSV: {final_csv}")

        # Temp CSVs go here and get auto-deleted
        with tempfile.TemporaryDirectory(prefix=f"evaltmp_{ckpt_path.stem}_{timestamp}_") as tmpdir:
            tmpdir = Path(tmpdir)

            processes = []
            for t in tasks:
                gpu = t["gpu"]
                dataset = t["dataset"]
                dataset_stem = Path(dataset).stem

                temp_csv = tmpdir / f"{dataset_stem}.csv"

                cmd = [
                    "python3", "eval.py",
                    "--ckpt", str(ckpt_path),
                    "--nabla",
                    "--dataset", dataset,
                    "--output", str(temp_csv),
                    "--num_workers", str(args.num_workers),
                ]

                if args.split is not None:
                    cmd.extend(["--split", args.split])

                env = build_eval_env(base_env, gpu=gpu, ckpt_path=ckpt_path, eval_logs_dir=eval_logs_dir)

                # Capture stdout/stderr into eval_logs_dir too
                task_log = eval_logs_dir / f"{ckpt_path.stem}_{timestamp}_{dataset_stem}.stdout.log"
                lprint(f"  [GPU {gpu}] Evaluating {dataset} -> {temp_csv}")
                lprint(f"         stdout/stderr: {task_log}")

                lf = open(task_log, "w")
                p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
                processes.append((p, dataset, temp_csv, lf))

            all_success = True
            partial_csvs = []

            for p, dataset, temp_csv, lf in processes:
                rc = p.wait()
                lf.close()
                if rc == 0 and temp_csv.exists():
                    lprint(f"  [OK] {dataset} complete.")
                    partial_csvs.append(temp_csv)
                else:
                    lprint(f"  [ERROR] {dataset} failed (rc={rc}). See logs in: {eval_logs_dir}")
                    all_success = False

            if not all_success:
                lprint("\n⚠️  Some evaluations failed. No merge produced.")
                sys.exit(1)

            # Merge partial CSVs into final_csv
            lprint(f"\nMerging {len(partial_csvs)} partial CSVs -> {final_csv}")
            rows_written = 0
            with open(final_csv, "w", newline="") as f_out:
                writer = None
                for part in partial_csvs:
                    with open(part, "r", newline="") as f_in:
                        reader = csv.DictReader(f_in)
                        if reader.fieldnames is None:
                            continue
                        if writer is None:
                            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
                            writer.writeheader()
                        for row in reader:
                            writer.writerow(row)
                            rows_written += 1

            if rows_written == 0:
                lprint("⚠️  Merge produced 0 rows (unexpected). Check per-task logs.")
                sys.exit(1)

            lprint(f"✅ Merged {rows_written} rows into {final_csv}")

    else:
        # In SLURM, respect the GPU allocation exactly as provided by SLURM.
        # If not in SLURM, find a free GPU and set CUDA_VISIBLE_DEVICES accordingly.
        if "SLURM_JOB_ID" in os.environ:
            gpu = None
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            lprint(f"🚀 Evaluating {ckpt_path.name} on SLURM-allocated GPU(s): {visible}")
        else:
            free_gpus = find_free_gpus(1)
            if not free_gpus:
                lprint("Error: No free GPUs available")
                sys.exit(1)
            gpu = free_gpus[0]
            lprint(f"🚀 Evaluating {ckpt_path.name} on GPU {gpu}")

        lprint(f"Eval logs dir: {eval_logs_dir}")
        lprint(f"Final CSV: {final_csv}")

        cmd = [
            "python3", "eval.py",
            "--ckpt", str(ckpt_path),
            "--output", str(final_csv),
            "--num_workers", str(args.num_workers),
        ]

        if args.split is not None:
            cmd.extend(["--split", args.split])

        env = build_eval_env(base_env, gpu=gpu, ckpt_path=ckpt_path, eval_logs_dir=eval_logs_dir)

        task_log = eval_logs_dir / f"{ckpt_path.stem}_{timestamp}.stdout.log"
        lprint(f"stdout/stderr: {task_log}")

        with open(task_log, "w") as lf:
            rc = subprocess.call(cmd, env=env, stdout=lf, stderr=lf)

        if rc != 0:
            lprint(f"❌ Evaluation failed (exit code {rc}). See logs in: {eval_logs_dir}")
            sys.exit(1)

        lprint(f"✅ Evaluation complete. Results: {final_csv}")

    lprint("\nAll evaluation tasks finished.")

if __name__ == "__main__":
    main()
