#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import random
import torch

TRAIN_SCRIPT = "train.py"
EVAL_SCRIPT = "submit_eval.py"

CKPTS_BASE = "/mnt/weka/fgeikyan/fsq/new_checkpoints/"


def lr_to_str(lr: float) -> str:
    s = str(float(lr))
    return s.replace("-", "m").replace("+", "p").replace(".", "_")


def resolve_eval_ckpt(ckpt_dir: Path):
    ckpt_paths = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_paths:
        return None

    best = None
    best_key = None
    for path in ckpt_paths:
        try:
            ckpt = torch.load(path, map_location="cpu")
            key = (int(ckpt.get("global_step", -1)), int(ckpt.get("epoch", -1)), path.stat().st_mtime)
        except Exception:
            key = (-1, -1, path.stat().st_mtime)
        if best_key is None or key > best_key:
            best = path
            best_key = key
    return best


def next_train_log_path(log_dir: Path) -> Path:
    primary = log_dir / "train.log"
    if not primary.exists():
        return primary

    idx = 1
    while True:
        candidate = log_dir / f"train{idx}.log"
        if not candidate.exists():
            return candidate
        idx += 1


def launch_post_train_eval(script_dir: str, ckpt_path: Path, data_name: str):
    if data_name == "nablaDFT":
        eval_args = ["--nabla", "--dataset", "nablaDFT.csv", "--split", "val"]
    else:
        eval_args = ["--split", "val"]

    cmd = [sys.executable, os.path.join(script_dir, EVAL_SCRIPT), "--ckpt", str(ckpt_path), *eval_args]
    print(f"[EVAL] running post-train eval for {ckpt_path} with data={data_name}", flush=True)
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="Submit training job")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["full_5pct", "full_20pct", "full", "nablaDFT"],
        help="Dataset to use",
    )
    parser.add_argument("--levels-size", dest="levels_size", type=int, required=True)
    parser.add_argument("--d-model", dest="d_model", type=int, required=True)
    parser.add_argument("--batch-base", dest="batch_base", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-ckpt", dest="resume_ckpt", type=str)
    parser.add_argument("--run-name", dest="run_name", type=str)
    args = parser.parse_args()
    use_1gpu = os.environ["USE_1GPU"].lower() in ("1", "true", "yes", "y")
    ckpt_every_percent = os.environ.get("CKPT_EVERY_PERCENT", "10")
    stop_after_epoch_fraction = os.environ.get("STOP_AFTER_EPOCH_FRACTION", "")
    lightning_profiler = os.environ["LIGHTNING_PROFILER"]
    atoms_to_decoder = os.environ["ATOMS_TO_DECODER"]
    atoms_to_encoder = os.environ["ATOMS_TO_ENCODER"]
    nproc_per_node = 1 if use_1gpu else 8

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(script_dir, TRAIN_SCRIPT)

    if not os.path.exists(train_py_path):
        print(f"[ERROR] Missing {TRAIN_SCRIPT} at: {train_py_path}")
        sys.exit(1)

    runs_base = os.environ.get("FSQ_RUNS_DIR", os.getcwd())
    os.makedirs(runs_base, exist_ok=True)

    resume_ckpt = Path(args.resume_ckpt).expanduser().resolve() if args.resume_ckpt else None
    if resume_ckpt is not None and not resume_ckpt.is_file():
        print(f"[ERROR] Resume checkpoint not found: {resume_ckpt}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vocab_size = pow(2, args.levels_size)
    lr_str = lr_to_str(args.lr)
    selective_decay = os.environ.get("SELECTIVE_DECAY")
    if selective_decay is None or selective_decay == "":
        selective_decay = "false" if args.data == "nablaDFT" else "true"
    selective_decay_bool = selective_decay.lower() in ("1", "true", "yes", "y")
    selective_decay_tag = f"seldec{1 if selective_decay_bool else 0}"

    if args.run_name:
        run_name = args.run_name
    elif resume_ckpt is not None:
        run_name = f"{resume_ckpt.parent.name}_{selective_decay_tag}_to_e{args.epochs}_{timestamp}"
    else:
        run_name = f"{args.data}_d{args.d_model}_v{vocab_size}_b{args.batch_base}_lr{lr_str}_{selective_decay_tag}_e{args.epochs}_{timestamp}"

    RUN_NAME = run_name
    LOG_DIR = os.path.join(runs_base, RUN_NAME)
    CKPT_DIR = os.path.join(CKPTS_BASE, RUN_NAME)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    shutil.copy2(train_py_path, os.path.join(LOG_DIR, "train.py"))
    with open(os.path.join(LOG_DIR, "train_config.env"), "w") as f:
        f.write(f"DATA={args.data}\n")
        f.write(f"TRAIN_LEVELS_SIZE={args.levels_size}\n")
        f.write(f"TRAIN_D_MODEL={args.d_model}\n")
        f.write(f"TRAIN_BATCH_BASE={args.batch_base}\n")
        f.write(f"TRAIN_EPOCHS={args.epochs}\n")
        f.write(f"TRAIN_LR={args.lr}\n")
        f.write(f"TRAIN_SEED={args.seed}\n")
        f.write(f"SELECTIVE_DECAY={'true' if selective_decay_bool else 'false'}\n")
        f.write(f"CKPT_EVERY_PERCENT={ckpt_every_percent}\n")
        if stop_after_epoch_fraction:
            f.write(f"STOP_AFTER_EPOCH_FRACTION={stop_after_epoch_fraction}\n")
        if resume_ckpt is not None:
            f.write(f"RESUME_CKPT_PATH={resume_ckpt}\n")
        f.write(f"ATOMS_TO_DECODER={atoms_to_decoder}\n")
        f.write(f"ATOMS_TO_ENCODER={atoms_to_encoder}\n")
    with open(os.path.join(LOG_DIR, "run_name.txt"), "w") as f:
        f.write(RUN_NAME + "\n")
    with open(os.path.join(LOG_DIR, "paths.txt"), "w") as f:
        f.write(f"RUN_NAME={RUN_NAME}\nLOG_DIR={LOG_DIR}\nCKPT_DIR={CKPT_DIR}\n")
        if resume_ckpt is not None:
            f.write(f"RESUME_CKPT_PATH={resume_ckpt}\n")

    log_file = str(next_train_log_path(Path(LOG_DIR)))
    default_port = 15000 + random.randint(0, 20000)
    master_port = int(os.environ.get("MASTER_PORT", str(default_port)))

    cmd_parts = [
        "export PYTHONNOUSERSITE=1 && ",
        "export OMP_NUM_THREADS=1 && ",
        "export MKL_NUM_THREADS=1 && ",
        "export OPENBLAS_NUM_THREADS=1 && ",
        "export NUMEXPR_NUM_THREADS=1 && ",
        "export VECLIB_MAXIMUM_THREADS=1 && ",
        "export OMP_PROC_BIND=FALSE && ",
        "export OMP_PLACES=threads && ",
        f"export MASTER_PORT='{master_port}' && ",
        f"export USE_1GPU='{'true' if use_1gpu else 'false'}' && ",
        f"export RUN_NAME='{RUN_NAME}' && ",
        f"export LOG_DIR='{LOG_DIR}' && ",
        f"export CKPT_DIR='{CKPT_DIR}' && ",
        f"export DATA='{args.data}' && ",
        f"export TRAIN_LEVELS_SIZE='{args.levels_size}' && ",
        f"export TRAIN_D_MODEL='{args.d_model}' && ",
        f"export TRAIN_BATCH_BASE='{args.batch_base}' && ",
        f"export TRAIN_EPOCHS='{args.epochs}' && ",
        f"export TRAIN_LR='{args.lr}' && ",
        f"export TRAIN_SEED='{args.seed}' && ",
        f"export SELECTIVE_DECAY='{'true' if selective_decay_bool else 'false'}' && ",
        f"export CKPT_EVERY_PERCENT='{ckpt_every_percent}' && ",
    ]
    if resume_ckpt is not None:
        cmd_parts.append(f"export RESUME_CKPT_PATH='{resume_ckpt}' && ")
    if stop_after_epoch_fraction:
        cmd_parts.append(f"export STOP_AFTER_EPOCH_FRACTION='{stop_after_epoch_fraction}' && ")
    cmd_parts.extend(
        [
            f"export LIGHTNING_PROFILER='{lightning_profiler}' && ",
            f"export ATOMS_TO_DECODER='{atoms_to_decoder}' && ",
            f"export ATOMS_TO_ENCODER='{atoms_to_encoder}' && ",
            f"python -m torch.distributed.run --nproc_per_node={nproc_per_node} '{train_py_path}' ",
            f"2>&1 | tee '{log_file}'",
        ]
    )
    cmd_str = "".join(cmd_parts)

    print(f"[RUN_NAME] {RUN_NAME}")
    print(f"[RUNS_BASE] {runs_base}")
    print(f"[LOG_DIR ] {LOG_DIR}")
    print(f"[CKPT_DIR] {CKPT_DIR}")
    print(f"[DATA    ] {args.data}")
    print(f"[LEVELS_SIZE] {args.levels_size}")
    print(f"[D_MODEL ] {args.d_model}")
    print(f"[BATCH_BASE] {args.batch_base}")
    print(f"[EPOCHS  ] {args.epochs}")
    print(f"[LR      ] {args.lr}")
    print(f"[SEED    ] {args.seed}")
    print(f"[SELECTIVE_DECAY] {selective_decay_bool}")
    print(f"[CKPT_EVERY_PERCENT] {ckpt_every_percent}")
    if stop_after_epoch_fraction:
        print(f"[STOP_AFTER_EPOCH_FRACTION] {stop_after_epoch_fraction}")
    if resume_ckpt is not None:
        print(f"[RESUME_CKPT] {resume_ckpt}")
    print(f"[USE_1GPU] {use_1gpu}")
    print(f"[PROFILER] {lightning_profiler}")
    print(f"[ATOMS_TO_DECODER] {atoms_to_decoder}")
    print(f"[ATOMS_TO_ENCODER] {atoms_to_encoder}")
    print(f"[NPROC   ] {nproc_per_node}")
    print(f"[MASTER_PORT] {master_port}")
    print(f"[LOGFILE ] {log_file}")
    print(f"[CMD     ] {cmd_str}", flush=True)

    rc = subprocess.run(["bash", "-c", cmd_str]).returncode
    if rc != 0:
        sys.exit(rc)

    ckpt_path = resolve_eval_ckpt(Path(CKPT_DIR))
    if ckpt_path is None:
        print(f"[EVAL] no checkpoint found in {CKPT_DIR}; skipping eval submission", flush=True)
        sys.exit(1)

    print(f"[EVAL] selected checkpoint for post-train eval: {ckpt_path}", flush=True)
    eval_rc = launch_post_train_eval(script_dir, ckpt_path, args.data)
    sys.exit(eval_rc)


if __name__ == "__main__":
    main()
