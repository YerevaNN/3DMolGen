
#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import subprocess
import itertools
import math
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_NABLA_DATA_ROOT = Path("/mnt/weka/fgeikyan/fsq/TokenizerData")
DEFAULT_NABLA_VAL_DIR = Path("/mnt/weka/fgeikyan/fsq/nabla_val")
DEFAULT_FULL_DATA_DIR = Path("/mnt/weka/fgeikyan/fsq")
DEFAULT_LOGS_BASE = Path("/home/fgeikyan/tokenizer/logs")
RUNS_BASE = Path(os.environ.get("FSQ_RUNS_DIR", os.getcwd()))
BATCH_SIZE = None


def find_run_log_dir(ckpt_dir_name: str) -> Path:
    preferred = RUNS_BASE / ckpt_dir_name
    if preferred.exists():
        return preferred
    return DEFAULT_LOGS_BASE / ckpt_dir_name


def get_data_dir(split_name: str, *, nabla: bool) -> Path:
    data_name = os.environ.get("DATA", "nablaDFT")
    if nabla:
        if split_name == "val":
            return DEFAULT_NABLA_VAL_DIR
        return DEFAULT_NABLA_DATA_ROOT / split_name
    if data_name in {"full_5pct", "full_20pct", "full"}:
        return DEFAULT_FULL_DATA_DIR / split_name
    return DEFAULT_NABLA_DATA_ROOT / split_name

# -----------------------------------------------------------------------------
# Import project code from logs folder (train.py that was used for training)
# -----------------------------------------------------------------------------
def setup_train_import(ckpt_path: Path):
    ckpt_dir_name = ckpt_path.parent.name
    run_dir = find_run_log_dir(ckpt_dir_name)
    train_py_path = run_dir / "train.py"
    
    if not train_py_path.exists():
        print(f"Warning: {train_py_path} not found, falling back to original train.py", flush=True)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    else:
        sys.path.insert(0, str(run_dir))
        print(f"Using train.py from: {train_py_path}", flush=True)
    
    try:
        from train import MolModel, MAX_TOKENS, V, VOCAB, BATCH_SIZE as TRAIN_BATCH_SIZE
        from utils import tokenize_and_encode
        return MolModel, MAX_TOKENS, V, VOCAB, tokenize_and_encode, TRAIN_BATCH_SIZE
    except ImportError as e:
        print(f"Error: Missing project files: {e}", flush=True)
        sys.exit(1)

# Atom mask over vocab ids (will be initialized after import)
IS_ATOM_NP = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def count_csv_samples(csv_path: Path) -> int:
    """Fast-ish row count using wc -l, minus header."""
    try:
        result = subprocess.run(["wc", "-l", str(csv_path)], capture_output=True, text=True, check=True)
        return max(0, int(result.stdout.strip().split()[0]) - 1)
    except Exception as e:
        print(f"Error counting samples: {e}", flush=True)
        return 0


def get_csv_sample_count(csv_path: Path) -> int:
    sidecar_path = csv_path.with_suffix(".nrows.txt")
    if sidecar_path.exists():
        try:
            return int(sidecar_path.read_text().strip())
        except Exception as e:
            print(f"Error reading sample count from {sidecar_path}: {e}", flush=True)
    return count_csv_samples(csv_path)


class MolDataset(IterableDataset):
    def __init__(self, path: Path, num_samples: int):
        super().__init__()
        self.path = Path(path)
        self.num_samples = int(num_samples)

    def __len__(self):
        if BATCH_SIZE is None:
            raise RuntimeError("BATCH_SIZE must be set before creating MolDataset")
        return math.ceil(self.num_samples / BATCH_SIZE) if self.num_samples > 0 else 0

    def __iter__(self):
        if BATCH_SIZE is None or MAX_TOKENS is None or V is None:
            raise RuntimeError("BATCH_SIZE, MAX_TOKENS, and V must be set before using MolDataset")
        info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (info.num_workers, info.id) if info else (1, 0)

        if worker_id == 0:
            print(f"[MolDataset] {self.path} samples={self.num_samples}", flush=True)

        f_batch = np.zeros((BATCH_SIZE, MAX_TOKENS, V + 3), dtype=np.float32)
        m_batch = np.zeros((BATCH_SIZE, MAX_TOKENS), dtype=np.float32)
        curr = 0

        with open(self.path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                text_col = header.index("enriched_text")
            except ValueError:
                print(f"Error finding enriched_text column: {e}", flush=True)
                return

            for row in itertools.islice(reader, worker_id, None, num_workers):
                try:
                    toks, tok_ids = tokenize_and_encode(row[text_col])
                    n = min(toks.shape[0], MAX_TOKENS)

                    mask_float = IS_ATOM_NP[tok_ids[:n]]
                    mask_bool = mask_float.astype(bool)

                    if mask_bool.any():
                        coords = toks[:n, V:]
                        coords[mask_bool] -= coords[mask_bool].mean(axis=0)
                        # toks[:n][mask_bool, V:] -= toks[:n][mask_bool, V:].mean(axis=0)

                    f_batch[curr, :n] = toks[:n]
                    m_batch[curr, :n] = mask_float
                    curr += 1

                    if curr == BATCH_SIZE:
                        yield torch.from_numpy(f_batch.copy()), torch.from_numpy(m_batch.copy())
                        f_batch.fill(0.0)
                        m_batch.fill(0.0)
                        curr = 0
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue

            if curr > 0:
                yield torch.from_numpy(f_batch.copy()), torch.from_numpy(m_batch.copy())


def make_loader(csv_path: Path, num_workers: int) -> DataLoader:
    num_samples = get_csv_sample_count(csv_path)
    ds = MolDataset(csv_path, num_samples=num_samples)
    return DataLoader(
        ds,
        batch_size=None,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


class EvalProgressBar(TQDMProgressBar):
    def __init__(self, total_batches):
        super().__init__()
        self.total_batches = total_batches

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.total = self.total_batches
        return bar


def save_token_rmse_plot(rows, plot_path: Path, title: str):
    if not rows:
        return

    xs = np.array([r["n_tokens"] for r in rows], dtype=np.int32)
    ys = np.array([r["rmse"] for r in rows], dtype=np.float32)

    n_bins = min(16, max(6, int(np.sqrt(len(rows)))))
    edges = np.linspace(xs.min(), xs.max() + 1, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    mean_rmse = []
    counts = []
    labels = []
    for lo, hi, mid in zip(edges[:-1], edges[1:], mids):
        if hi == edges[-1]:
            mask = (xs >= lo) & (xs <= hi)
        else:
            mask = (xs >= lo) & (xs < hi)
        if not np.any(mask):
            continue
        mean_rmse.append(float(ys[mask].mean()))
        counts.append(int(mask.sum()))
        labels.append(float(mid))

    if not labels:
        return

    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax2 = ax1.twinx()

    bar_width = max(1.0, (edges[1] - edges[0]) * 0.78)
    ax2.bar(labels, counts, width=bar_width, color="#d7dee8", edgecolor="none", alpha=0.85, zorder=1)
    ax1.plot(labels, mean_rmse, color="#163a5f", linewidth=2.6, marker="o", markersize=4.5, zorder=3)

    ax1.set_xlabel("Number of tokens")
    ax1.set_ylabel("Average RMSD")
    ax2.set_ylabel("Count")

    ax1.set_title("")
    ax1.grid(True, axis="y", color="#cfcfcf", linewidth=0.8, alpha=0.6)
    ax1.grid(False, axis="x")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax1.tick_params(axis="both", labelsize=10)
    ax2.tick_params(axis="y", labelsize=10)
    ax1.yaxis.label.set_size(11)
    ax2.yaxis.label.set_size(11)
    ax1.xaxis.label.set_size(11)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_eval(model, split_name: str, args):
    split_path = get_data_dir(split_name, nabla=args.nabla)
    if not split_path.exists():
        return {}, {}

    if args.nabla:
        if not args.dataset:
            return {}, {}
        file_path = split_path / args.dataset
        files = [file_path] if file_path.exists() else []
    else:
        files = sorted(split_path.glob("*.csv"))
        if args.dataset:
            files = [f for f in files if f.name == args.dataset or f.stem == args.dataset]

    if not files:
        return {}, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    metrics = {}
    per_dataset_rows = {}
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for f in files:
        name = f.stem
        loader = make_loader(f, args.num_workers)
        rows = []
        sum_rmse = 0.0
        count = 0

        for x, m in loader:
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True).bool()
            target = x[..., -3:]
            with torch.no_grad():
                with autocast_ctx:
                    recon, _ = model(x)

            token_presence = (x[..., :V].sum(dim=-1) > 0)
            for b in range(x.size(0)):
                m_b = m[b]
                if not m_b.any():
                    continue
                mse = (recon[b][m_b] - target[b][m_b]).square().sum(dim=-1).mean()
                rmse = float(torch.sqrt(mse).detach().float().cpu())
                n_atoms = int(m_b.sum().item())
                n_tokens = int(token_presence[b].sum().item())
                rows.append({"dataset": name, "split": split_name, "n_tokens": n_tokens, "n_atoms": n_atoms, "rmse": rmse})
                sum_rmse += rmse
                count += 1

        if count > 0:
            metrics[name] = {"rmse": round(sum_rmse / count, 6), "count": count}
            per_dataset_rows[name] = rows

    return metrics, per_dataset_rows


def default_eval_logs_dir(ckpt: Path) -> Path:
    """Always keep eval outputs inside the training run folder."""
    ckpt_dir_name = ckpt.parent.name
    d = find_run_log_dir(ckpt_dir_name) / "eval_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_output_csv(ckpt: Path, args) -> Path:
    """
    No more ./evals folder. Default CSV goes to:
      /home/chem-project/fsq/logs/<run>/eval_logs/<ckpt>_eval.csv
    In --nabla mode with --dataset, use a dataset-specific suffix to avoid clashes.
    """
    out_dir = default_eval_logs_dir(ckpt)
    if args.nabla and args.dataset:
        return out_dir / f"{ckpt.stem}_{Path(args.dataset).stem}_eval.csv"
    return out_dir / f"{ckpt.stem}_eval.csv"


def default_log_file(ckpt: Path, args) -> Path:
    """
    Logs also go to eval_logs.
    In --nabla mode with --dataset, make it unique per process to avoid clobbering.
    """
    out_dir = default_eval_logs_dir(ckpt)
    if args.nabla and args.dataset:
        return out_dir / f"{ckpt.stem}_{Path(args.dataset).stem}_eval.log"
    return out_dir / f"{ckpt.stem}_eval.log"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, dest="checkpoint_path", required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path. Defaults to logs/<run>/eval_logs/<ckpt>_eval.csv "
                             "(or <ckpt>_<dataset>_eval.csv in --nabla mode).")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to specific dataset file (by name or stem). "
                             "In --nabla mode, this should be the exact file name (e.g. nablaDFT.csv).")
    parser.add_argument("--nabla", action="store_true",
                        help="Use nablaDFT-specific file filtering (val: nablaDFT.csv, test: conformations/scaffolds/structures)")
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"],
                        help="Limit evaluation to a specific split (e.g., 'val' or 'test'). If not specified, evaluates both.")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override MAX_TOKENS from train.py (useful for datasets like geom_xl that need longer sequences)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_path)
    if not ckpt.is_file():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        sys.exit(1)

    # Import train.py from logs folder
    MolModel, MAX_TOKENS_ORIG, V, VOCAB, tokenize_and_encode, TRAIN_BATCH_SIZE = setup_train_import(ckpt)
    
    # Override MAX_TOKENS if specified (e.g., for geom_xl which needs 240)
    # For nabla, automatically use 54 (consistent with training) unless explicitly overridden
    MAX_TOKENS = MAX_TOKENS_ORIG
    if args.max_tokens is not None:
        MAX_TOKENS = args.max_tokens
        print(f"Overriding MAX_TOKENS to {MAX_TOKENS} (from train.py: {MAX_TOKENS_ORIG})", flush=True)
    elif args.nabla:
        MAX_TOKENS = 54
        if MAX_TOKENS != MAX_TOKENS_ORIG:
            print(f"Using MAX_TOKENS=54 for nablaDFT (from train.py: {MAX_TOKENS_ORIG})", flush=True)
    
    # Keep eval batch size identical to training batch size unless explicitly changed in code.
    BATCH_SIZE = TRAIN_BATCH_SIZE
    print(f"Using eval batch size: {BATCH_SIZE} (same as training batch size)", flush=True)
    print(f"Using MAX_TOKENS: {MAX_TOKENS}", flush=True)
    
    # Initialize atom mask after importing VOCAB
    IS_ATOM_NP = np.array([v.startswith('[') for v in VOCAB], dtype=np.float32)

    # Output CSV (default: logs/<run>/eval_logs/..., never ./evals)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = default_output_csv(ckpt, args)

    # Log file (always logs/<run>/eval_logs/..., and unique in nabla mode)
    log_file = default_log_file(ckpt, args)

    print(f"\n🚀 Loading: {ckpt.name}", flush=True)
    print(f"Writing CSV: {output_path}", flush=True)
    print(f"Logging to : {log_file}", flush=True)

    model = MolModel.load_from_checkpoint(str(ckpt))

    # Write both to terminal and log file
    with open(log_file, "w") as log_f, open(output_path, "w", newline="") as out_f:
        def log_print(*a, **k):
            print(*a, **k)
            print(*a, file=log_f, **k)

        log_print(f"Evaluation for checkpoint: {ckpt.name}")
        log_print(f"Checkpoint directory: {ckpt.parent.name}")
        log_print("=" * 80)

        writer = csv.DictWriter(out_f, fieldnames=["Checkpoint", "Dataset", "Split", "RMSE", "Count"])
        writer.writeheader()

        splits_to_evaluate = [args.split] if args.split else ["val", "test"]
        for split in splits_to_evaluate:
            results, per_dataset_rows = run_eval(model, split, args)
            if results:
                for name, payload in results.items():
                    rmse = payload["rmse"]
                    count = payload["count"]
                    writer.writerow({"Checkpoint": ckpt.stem, "Dataset": name, "Split": split, "RMSE": rmse, "Count": count})
                    log_print(f"    [{split}] {name:.<30} RMSE: {rmse:.6f} over {count} molecules")

                    stem = output_path.stem
                    detail_csv = output_path.with_name(f"{stem}_{name}_per_molecule.csv")
                    plot_png = output_path.with_name(f"{stem}_{name}_tokens_vs_rmse.png")
                    with open(detail_csv, "w", newline="") as detail_f:
                        detail_writer = csv.DictWriter(detail_f, fieldnames=["dataset", "split", "n_tokens", "n_atoms", "rmse"])
                        detail_writer.writeheader()
                        detail_writer.writerows(per_dataset_rows[name])
                    save_token_rmse_plot(per_dataset_rows[name], plot_png, f"{name} ({split}) token count vs RMSE")
                    log_print(f"        details: {detail_csv}")
                    log_print(f"        plot   : {plot_png}")
            out_f.flush()
            log_f.flush()

        log_print(f"\n✅ Done. Results: {output_path}")
        log_print(f"Log file: {log_file}")
