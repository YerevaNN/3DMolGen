
#!/usr/bin/env python3
import math, os, sys, csv, cProfile, pstats, subprocess, torch, numpy as np, itertools, lightning as L
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.strategies import DDPStrategy
from datetime import datetime

from fsq import FSQ
from utils import tokenize_and_encode, V, VOCAB, build_fsq_string, parse_fsq_text, tokens_to_vocab_onehot, format_enriched_from_tokens_and_coords, CustomProgressBar, MFUCallback


# Atom mask over vocab ids
IS_ATOM_NP = np.array([v.startswith('[') for v in VOCAB], dtype=np.float32)  # [V]
IS_ATOM_TORCH = torch.from_numpy(IS_ATOM_NP)  # used only for optional checks/logging


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "y")


def _print_startup_config(*, rank: int, world: int, local_world: int):
    if rank != 0:
        return

    print("=== TRAIN CONFIG ===", flush=True)
    print(f"[CONFIG] DATA={DATA}", flush=True)
    print(f"[CONFIG] LEVELS_SIZE={LEVELS_SIZE}", flush=True)
    print(f"[CONFIG] LEVELS={LEVELS}", flush=True)
    print(f"[CONFIG] D_MODEL={D_MODEL}", flush=True)
    print(f"[CONFIG] N_LAYERS={N_LAYERS}", flush=True)
    print(f"[CONFIG] BATCH_BASE={BATCH_BASE}", flush=True)
    print(f"[CONFIG] BATCH_SIZE={BATCH_SIZE}", flush=True)
    print(f"[CONFIG] EPOCHS={EPOCHS}", flush=True)
    print(f"[CONFIG] CKPT_EVERY_PERCENT={CKPT_EVERY_PERCENT}", flush=True)
    print(f"[CONFIG] STOP_AFTER_EPOCH_FRACTION={STOP_AFTER_EPOCH_FRACTION}", flush=True)
    print(f"[CONFIG] SEED={SEED}", flush=True)
    print(f"[CONFIG] LR={LR}", flush=True)
    print(f"[CONFIG] SELECTIVE_DECAY={SELECTIVE_DECAY}", flush=True)
    print(f"[CONFIG] MAX_TOKENS={MAX_TOKENS}", flush=True)
    print(f"[CONFIG] NUM_WORKERS={NUM_WORKERS}", flush=True)
    print(f"[CONFIG] USE_1GPU={USE_1GPU}", flush=True)
    print(f"[CONFIG] NUM_DEVICES={NUM_DEVICES}", flush=True)
    print(f"[CONFIG] ATOMS_TO_DECODER={atoms_to_decoder}", flush=True)
    print(f"[CONFIG] ATOMS_TO_ENCODER={atoms_to_encoder}", flush=True)
    print(f"[CONFIG] WORLD_SIZE={world}", flush=True)
    print(f"[CONFIG] LOCAL_WORLD_SIZE={local_world}", flush=True)
    print(f"[CONFIG] PACKED_DIR={PACKED_DIR}", flush=True)
    print(f"[CONFIG] VAL_DIR={VAL_DIR}", flush=True)
    print(f"[CONFIG] EPOCH_FILE={EPOCH_FILE}", flush=True)
    print(f"[CONFIG] RUN_NAME={RUN_NAME}", flush=True)
    print(f"[CONFIG] LOG_DIR={LOG_DIR}", flush=True)
    print(f"[CONFIG] CKPT_DIR={CKPT_DIR}", flush=True)
    print("=== END TRAIN CONFIG ===", flush=True)


# =============================================================================
# Hyperparameters
# =============================================================================
LEVELS_SIZE = _get_env_int("TRAIN_LEVELS_SIZE", 12)
LEVELS = [2] * LEVELS_SIZE
D_MODEL = _get_env_int("TRAIN_D_MODEL", 1024)
BATCH_BASE = _get_env_int("TRAIN_BATCH_BASE", 64)
EPOCHS = _get_env_int("TRAIN_EPOCHS", 1)
SEED = _get_env_int("TRAIN_SEED", 42)
LR = _get_env_float("TRAIN_LR", 1e-4)
DATA = os.environ.get("DATA", "full")
SELECTIVE_DECAY = _get_env_bool("SELECTIVE_DECAY", DATA != "nablaDFT")
CKPT_EVERY_PERCENT = _get_env_float("CKPT_EVERY_PERCENT", 10.0 if DATA == "full" else 100.0)
STOP_AFTER_EPOCH_FRACTION = _get_env_float("STOP_AFTER_EPOCH_FRACTION", 0.0)
N_LAYERS = D_MODEL // 128
BATCH_SIZE = BATCH_BASE * (1024//D_MODEL) * (8//N_LAYERS)
NUM_WORKERS = 3
STARTING_EPOCH = 0
NABLA_PACKED_DIR = "/mnt/weka/fgeikyan/fsq/shuffle_index_nabla"
NABLA_VAL_DIR = "/mnt/weka/fgeikyan/fsq/nabla_val"
FULL_PACKED_DIR = "/dev/shm/shuffle_index_merged_train"
FULL_5PCT_PACKED_DIR = "/dev/shm/shuffle_index_merged_train_5pct"
FULL_20PCT_PACKED_DIR = "/dev/shm/shuffle_index_merged_train"
FULL_VAL_DIR = "/mnt/weka/fgeikyan/fsq/val"
MAX_TOKENS = 54 if DATA == "nablaDFT" else 218
USE_1GPU = os.environ.get("USE_1GPU", "false").lower() in ("1", "true", "yes", "y")
NUM_DEVICES = 1 if USE_1GPU else 8
atoms_to_decoder = _get_env_bool("ATOMS_TO_DECODER", True)
atoms_to_encoder = _get_env_bool("ATOMS_TO_ENCODER", True)
RESUME_CKPT_PATH = os.environ.get("RESUME_CKPT_PATH") or None
if DATA == "nablaDFT":
    PACKED_DIR = NABLA_PACKED_DIR
    VAL_DIR = NABLA_VAL_DIR
elif DATA == "full_5pct":
    PACKED_DIR = FULL_5PCT_PACKED_DIR
    VAL_DIR = FULL_VAL_DIR
elif DATA == "full_20pct":
    PACKED_DIR = FULL_20PCT_PACKED_DIR
    VAL_DIR = FULL_VAL_DIR
elif DATA == "full":
    PACKED_DIR = FULL_PACKED_DIR
    VAL_DIR = FULL_VAL_DIR
else:
    raise ValueError(f"Unsupported DATA={DATA!r}. Expected 'full_5pct', 'full_20pct', 'full', or 'nablaDFT'.")
# LIGHTNING_PROFILER = os.environ.get("LIGHTNING_PROFILER", "advanced")
LIGHTNING_PROFILER = os.environ.get("LIGHTNING_PROFILER", "none")

EPOCH_FILE = f"/dev/shm/current_epoch_{os.environ.get('SLURM_JOB_ID','local')}.txt"
# =============================================================================
# >>> FINAL: CKPT_DIR / LOG_DIR come from submit.py (single source of truth)
# =============================================================================
RUN_NAME = os.environ.get("RUN_NAME")
LOG_DIR  = os.environ.get("LOG_DIR")
CKPT_DIR = os.environ.get("CKPT_DIR")

# Only enforce env vars when actually running training (not on import)
def _check_env_vars():
    missing = [k for k, v in [("RUN_NAME", RUN_NAME), ("LOG_DIR", LOG_DIR), ("CKPT_DIR", CKPT_DIR)] if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Launch via submit.py so logs/ckpts match.")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_dir = CKPT_DIR
    print(f"[RUN_NAME] {RUN_NAME}", flush=True)
    print(f"[LOG_DIR ] {LOG_DIR}", flush=True)
    print(f"[CKPT_DIR] {ckpt_dir}", flush=True)
    return ckpt_dir

ckpt_dir = None  # Will be set in __main__
# =============================================================================
# <<< END CKPT_DIR / LOG_DIR
# =============================================================================

# =============================================================================
# Implementation
# =============================================================================

import mmap


def count_csv_samples(csv_path: Path) -> int:
    try:
        result = subprocess.run(["wc", "-l", str(csv_path)], capture_output=True, text=True, check=True)
        return max(0, int(result.stdout.strip().split()[0]) - 1)
    except Exception as e:
        print(f"[VAL] Error counting samples in {csv_path}: {e}", flush=True)
        return 0


def get_csv_sample_count(csv_path: Path) -> int:
    sidecar_path = csv_path.with_suffix(".nrows.txt")
    if sidecar_path.exists():
        try:
            return int(sidecar_path.read_text().strip())
        except Exception as e:
            print(f"[VAL] Error reading sample count from {sidecar_path}: {e}", flush=True)
    return count_csv_samples(csv_path)


def read_key_value_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _batch_tensors(features, atom_mask):
    # Use all vocabulary positions: punctuation is a real token, too.
    present = np.flatnonzero(features[:, :, :V].any(axis=(0, 2)))
    length = int(present[-1]) + 1 if present.size else 1
    length = min(features.shape[1], ((length + 7) // 8) * 8)
    return (torch.from_numpy(features[:, :length].copy()),
            torch.from_numpy(atom_mask[:, :length].copy()))


class CSVMolDataset(IterableDataset):
    def __init__(self, path: str | Path, num_samples: int):
        super().__init__()
        self.path = Path(path)
        self.num_samples = int(num_samples)

    def __len__(self):
        return math.ceil(self.num_samples / BATCH_SIZE) if self.num_samples > 0 else 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (info.num_workers, info.id) if info else (1, 0)

        if worker_id == 0:
            print(f"[VAL] dataset={self.path} samples={self.num_samples}", flush=True)

        f_batch = np.zeros((BATCH_SIZE, MAX_TOKENS, V + 3), dtype=np.float32)
        m_batch = np.zeros((BATCH_SIZE, MAX_TOKENS), dtype=np.float32)
        curr = 0

        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                text_col = header.index("enriched_text")
            except ValueError:
                raise RuntimeError(f"CSV header in {self.path} does not contain 'enriched_text'")

            for row in itertools.islice(reader, worker_id, None, num_workers):
                try:
                    toks, tok_ids = tokenize_and_encode(row[text_col])
                    n = toks.shape[0]
                    if n > MAX_TOKENS:
                        print(f"[VAL] Skipping long molecule in {self.path.name}", flush=True)
                        continue

                    mask_float = IS_ATOM_NP[tok_ids]
                    mask_bool = mask_float.astype(bool)
                    if mask_bool.any():
                        toks[mask_bool, V:] -= toks[mask_bool, V:].mean(axis=0)

                    f_batch[curr, :n] = toks
                    m_batch[curr, :n] = mask_float
                    curr += 1

                    if curr == BATCH_SIZE:
                        yield _batch_tensors(f_batch, m_batch)
                        f_batch.fill(0.0)
                        m_batch.fill(0.0)
                        curr = 0
                except Exception as e:
                    print(f"[VAL] Error processing row from {self.path.name}: {e}", flush=True)
                    continue

            if curr > 0:
                yield _batch_tensors(f_batch, m_batch)


def make_val_loaders(val_dir: str | None, num_workers: int):
    if not val_dir:
        return None

    val_path = Path(val_dir)
    if not val_path.exists():
        print(f"[VAL] Validation directory not found: {val_path}", flush=True)
        return None

    files = sorted(val_path.glob("*.csv"))
    if not files:
        print(f"[VAL] No CSV files found in {val_path}", flush=True)
        return None

    loaders = []
    for csv_path in files:
        num_samples = get_csv_sample_count(csv_path)
        ds = CSVMolDataset(csv_path, num_samples=num_samples)
        loader = DataLoader(
            ds,
            batch_size=None,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        loader.dataset_name = csv_path.stem
        loaders.append(loader)

    print(f"[VAL] Using {len(loaders)} validation datasets from {val_path}", flush=True)
    return loaders

class MolDataset(IterableDataset):
    """
    Uses:
      part_{rank}/text.bin  (packed enriched_text bytes)
      part_{rank}/offs.u64  (offsets into text.bin)
      part_{rank}/lens.u32  (lengths)
      part_{rank}/perm_eXX.npy (shuffle order per epoch)
    """
    def __init__(self, packed_dir, epoch_file, nrows=0):
        super().__init__()
        self.packed_dir = Path(packed_dir)
        self.epoch_file = Path(epoch_file)
        self.nrows = int(nrows)

    def __len__(self):
        return (self.nrows + BATCH_SIZE - 1) // BATCH_SIZE

    def _get_epoch(self) -> int:
        try:
            return int(self.epoch_file.read_text().strip())
        except Exception:
            return 0

    def _get_resume_state(self):
        try:
            resume_data_epoch = int(os.environ.get("RESUME_DATA_EPOCH", "-1"))
        except Exception:
            resume_data_epoch = -1
        try:
            resume_batches_this_epoch = int(os.environ.get("RESUME_BATCHES_THIS_EPOCH", "0"))
        except Exception:
            resume_batches_this_epoch = 0
        return resume_data_epoch, max(0, resume_batches_this_epoch)

    def __iter__(self):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        part_dir = self.packed_dir / f"part_{rank}"

        text_path = part_dir / "text.bin"
        offs_path = part_dir / "offs.u64"
        lens_path = part_dir / "lens.u32"
        nrows = int((part_dir / "nrows.txt").read_text().strip())

        epoch = self._get_epoch()
        perm_path = part_dir / f"perm_e{epoch:02d}.npy"

        offs = np.memmap(offs_path, mode="r", dtype=np.uint64, shape=(nrows,))
        lens = np.memmap(lens_path, mode="r", dtype=np.uint32, shape=(nrows,))
        perm = np.load(perm_path, mmap_mode="r")  # uint32

        info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
        num_batches = (nrows + BATCH_SIZE - 1) // BATCH_SIZE
        resume_data_epoch, resume_batches_this_epoch = self._get_resume_state()
        start_batch_idx = resume_batches_this_epoch if epoch == resume_data_epoch else 0
        start_batch_idx = min(start_batch_idx, num_batches)
        if worker_id == 0 and rank == 0:
            print(f"[DATA] epoch={epoch} perm_file={perm_path}", flush=True)
            print(f"[DATA] perm_head={perm[:3].tolist()}", flush=True)
            print(
                f"[DATA] resume_data_epoch={resume_data_epoch} "
                f"resume_batches_this_epoch={resume_batches_this_epoch} "
                f"start_batch_idx={start_batch_idx} num_batches={num_batches}",
                flush=True,
            )

        profile_dataset = os.environ.get("PROFILE_DATASET", "0").lower() in ("1", "true", "yes", "y")
        profile_dataset_samples = int(os.environ.get("PROFILE_DATASET_SAMPLES", "5000"))
        profile_dataset_dir = Path(os.environ.get("PROFILE_DATASET_DIR", os.environ.get("LOG_DIR", "."))) / "profiling"
        profiler = None
        profiled_samples = 0
        profiler_stopped = False
        if profile_dataset:
            profile_dataset_dir.mkdir(parents=True, exist_ok=True)
            profiler = cProfile.Profile()
            profiler.enable()

        f_batch = np.zeros((BATCH_SIZE, MAX_TOKENS, V + 3), dtype=np.float32)
        m_batch = np.zeros((BATCH_SIZE, MAX_TOKENS), dtype=np.float32)
        curr = 0
        # worker_preproc_time = 0.0
        # emitted_batches = 0
        # emitted_samples = 0

        with open(text_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for batch_idx in range(start_batch_idx + worker_id, num_batches, num_workers):
                f_batch.fill(0.0)
                m_batch.fill(0.0)
                curr = 0

                batch_start = batch_idx * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, nrows)

                for pos in range(batch_start, batch_end):
                    ridx = int(perm[pos])
                    off = int(offs[ridx])
                    ln = int(lens[ridx])

                    try:
                        s = mm[off:off+ln].decode("utf-8", errors="strict")
                        toks, tok_ids = tokenize_and_encode(s)

                        n = toks.shape[0]
                        if n > MAX_TOKENS:
                            print(f"WOW, Skipping long molecule: {s}")
                            continue

                        mask_float = IS_ATOM_NP[tok_ids]
                        mask_bool = mask_float.astype(bool)
                        if mask_bool.any():
                            toks[mask_bool, V:] -= toks[mask_bool, V:].mean(axis=0)

                        f_batch[curr, :n] = toks
                        m_batch[curr, :n] = mask_float
                        curr += 1
                        if profiler is not None and not profiler_stopped:
                            profiled_samples += 1
                            if profiled_samples >= profile_dataset_samples:
                                profiler.disable()
                                profiler_stopped = True
                    except Exception as e:
                        print(f"WOW, Error processing row: {e}")
                        continue

                yield _batch_tensors(f_batch, m_batch)

            mm.close()

        # print(
        #     f"[DATA PREPROC] rank={rank} worker={worker_id}/{num_workers} pid={os.getpid()} "
        #     f"batches={emitted_batches} samples={emitted_samples} seconds={worker_preproc_time:.6f}",
        #     flush=True,
        # )

        if profiler is not None:
            if not profiler_stopped:
                profiler.disable()
            profile_path = profile_dataset_dir / (
                f"moldataset_rank{rank}_worker{worker_id}_pid{os.getpid()}_epoch{epoch:02d}.prof"
            )
            profiler.dump_stats(str(profile_path))
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            print(
                f"[DATA PROFILE] wrote={profile_path} profiled_samples={profiled_samples} "
                f"total_calls={stats.total_calls}",
                flush=True,
            )
        
class EpochFileCallback(L.Callback):
    def __init__(self, epoch_file: str, steps_per_epoch: int):
        super().__init__()
        self.epoch_file = Path(epoch_file)
        self.steps_per_epoch = max(1, int(steps_per_epoch))

    def _epoch_to_write(self, trainer) -> int:
        global_step = int(getattr(trainer, "global_step", 0))
        if global_step <= 0:
            return int(getattr(trainer, "current_epoch", 0))
        return global_step // self.steps_per_epoch

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.epoch_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[EpochFileCallback] Initialized epoch file to: {STARTING_EPOCH}")
            self.epoch_file.write_text(str(STARTING_EPOCH))
        trainer.strategy.barrier()

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            epoch_to_write = self._epoch_to_write(trainer)
            self.epoch_file.write_text(str(epoch_to_write))
            print(f"[EpochFileCallback] Updated {self.epoch_file} to epoch {epoch_to_write}")
        trainer.strategy.barrier()

class PrintResumeState(L.Callback):
    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        try:
            opt = trainer.optimizers[0] if trainer.optimizers else None
            lr = opt.param_groups[0]["lr"] if opt is not None else None

            # Lightning stores schedulers in lr_scheduler_configs
            if getattr(trainer, "lr_scheduler_configs", None):
                sched = trainer.lr_scheduler_configs[0].scheduler
                last_epoch = getattr(sched, "last_epoch", None)
                sched_name = type(sched).__name__
            else:
                sched = None
                last_epoch = None
                sched_name = None

            print(
                "[RESUME CHECK]",
                "global_step=", trainer.global_step,
                "current_epoch=", trainer.current_epoch,
                "scheduler=", sched_name,
                "sched.last_epoch=", last_epoch,
                "lr=", lr,
                flush=True
            )
        except Exception as e:
            print(f"[RESUME CHECK] Error: {e}", flush=True)


class ProgressStateCallback(L.Callback):
    def __init__(self, log_dir: str, ckpt_dir: str, steps_per_epoch: int, checkpoint_every_n_steps: int):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.steps_per_epoch = int(steps_per_epoch)
        self.checkpoint_every_n_steps = int(checkpoint_every_n_steps)
        self.progress_path = self.log_dir / "progress.txt"
        self.run_state_path = self.log_dir / "run_state.txt"

    def _write_progress(self, trainer):
        approx_epoch = float(trainer.global_step) / max(1, self.steps_per_epoch)
        resume_data_epoch = trainer.global_step // max(1, self.steps_per_epoch)
        resume_batches_this_epoch = trainer.global_step % max(1, self.steps_per_epoch)
        self.progress_path.write_text(
            "\n".join(
                [
                    f"global_step={trainer.global_step}",
                    f"current_epoch={trainer.current_epoch}",
                    f"approx_epoch={approx_epoch:.6f}",
                    f"resume_data_epoch={resume_data_epoch}",
                    f"resume_batches_this_epoch={resume_batches_this_epoch}",
                    f"steps_per_epoch={self.steps_per_epoch}",
                    f"checkpoint_every_n_steps={self.checkpoint_every_n_steps}",
                    f"latest_checkpoint_hint={self.ckpt_dir / 'last.ckpt'}",
                ]
            )
            + "\n"
        )

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        total_steps = trainer.max_steps if trainer.max_steps is not None else trainer.estimated_stepping_batches
        resume_data_epoch = trainer.global_step // max(1, self.steps_per_epoch)
        resume_batches_this_epoch = trainer.global_step % max(1, self.steps_per_epoch)
        self.run_state_path.write_text(
            "\n".join(
                [
                    f"data={DATA}",
                    f"epochs={EPOCHS}",
                    f"resume_data_epoch={resume_data_epoch}",
                    f"resume_batches_this_epoch={resume_batches_this_epoch}",
                    f"steps_per_epoch={self.steps_per_epoch}",
                    f"total_steps={total_steps}",
                    f"checkpoint_every_n_steps={self.checkpoint_every_n_steps}",
                    f"ckpt_every_percent={CKPT_EVERY_PERCENT}",
                    f"resume_ckpt_path={RESUME_CKPT_PATH or ''}",
                ]
            )
            + "\n"
        )
        self._write_progress(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if trainer.global_step > 0 and trainer.global_step % self.checkpoint_every_n_steps == 0:
            self._write_progress(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._write_progress(trainer)

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._write_progress(trainer)

class MolModel(L.LightningModule):
    def __init__(self, total_steps, d_model, n_layers, levels):
        super().__init__()
        self.save_hyperparameters()
        self.tok_emb = nn.Linear(V + 3, d_model)

        # Cache for sinusoidal PE; not saved in checkpoints (prevents shape mismatches when MAX_TOKENS changes)
        self.register_buffer("pos_emb_cache", torch.empty(1, 0, d_model), persistent=False)

        def block():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=d_model // 64,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=True,
                bias=False,
            )
            return nn.TransformerEncoder(layer, n_layers)

        self.enc, self.dec = block(), block()
        self.pre_q = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, len(levels)),
            nn.Tanh(),
        )
        self.quant = FSQ(
            levels,
            return_indices=True,
            preserve_symmetry=True,
            keep_num_codebooks_dim=False,
        )
        self.post_q = nn.Sequential(
            nn.Linear(V + len(levels), 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
        self.out = nn.Linear(d_model, 3)

        self.register_buffer("util", torch.zeros(math.prod(levels), dtype=torch.long))
        self.util_every = 500
        self.grad_norm_every = int(os.environ.get("GRAD_NORM_EVERY", "50"))
        self._util_buf = []


    def _get_pos_emb(self, n: int, device, dtype):
        pe = self.pos_emb_cache
        d_model = self.hparams.d_model
        if pe.size(1) >= n and pe.device == device and pe.dtype == dtype:
            return pe[:, :n]
        p = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        v = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        new_pe = torch.zeros(1, n, d_model, device=device, dtype=torch.float32)
        new_pe[0, :, 0::2] = torch.sin(p * v)
        new_pe[0, :, 1::2] = torch.cos(p * v)
        new_pe = new_pe.to(dtype=dtype)
        self.pos_emb_cache = new_pe
        return new_pe
            
    def encode_text(self, enriched_smiles: str) -> str:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        feats_np, tok_ids = tokenize_and_encode(enriched_smiles)
        mask_float = IS_ATOM_NP[tok_ids]
        mask_bool = mask_float.astype(bool)
        if mask_bool.any():
            feats_np[mask_bool, V:] -= feats_np[mask_bool, V:].mean(axis=0)
        x = torch.from_numpy(feats_np).unsqueeze(0).to(self._device)
        x = x.to(dtype=next(self.parameters()).dtype)
        with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if not atoms_to_encoder:
                x = x.clone()
                x[:, :, :V] = 0.0
            emb = self.tok_emb(x)
            pe = self._get_pos_emb(x.size(1), x.device, emb.dtype)
            z = self.enc(emb + pe)
            _, indices = self.quant(self.pre_q(z))
        indices_np = indices[0].detach().cpu().numpy()
        return build_fsq_string(enriched_smiles, indices_np)

    def _project_coordinates(self, h):
        # Keep the small XYZ projection in FP32 under mixed-precision training.
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.out(h.float())

    def decode_text(self, fsq_text: str, precision: int = 4) -> str:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        tokens, codes_np = parse_fsq_text(fsq_text)
        T = len(tokens)
        xv_np = tokens_to_vocab_onehot(tokens)
        indices = torch.from_numpy(codes_np).unsqueeze(0).to(self._device)
        xv = torch.from_numpy(xv_np).unsqueeze(0).to(self._device)
        with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            indices = indices.to(dtype=torch.int32, device=self._device)
            quant = self.quant.indices_to_codes(indices).to(dtype=next(self.dec.parameters()).dtype)
            xv = xv.to(dtype=quant.dtype, device=quant.device)
            if not atoms_to_decoder:
                xv = torch.zeros_like(xv)
            d_in = self.post_q(torch.cat([xv, quant], dim=-1))
            pe = self._get_pos_emb(T, d_in.device, d_in.dtype)
            h = self.dec(d_in + pe, src_key_padding_mask=None)
            recon = self._project_coordinates(h)[0].detach().cpu().numpy()

        return format_enriched_from_tokens_and_coords(tokens, recon, precision=precision)


    def forward(self, x):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            pad_mask = (x.abs().sum(dim=-1) == 0)
            if not atoms_to_encoder:
                x = x.clone()
                x[:, :, :V] = 0.0
            emb = self.tok_emb(x)
            pe = self._get_pos_emb(x.size(1), x.device, emb.dtype)
            z = self.enc(emb + pe, src_key_padding_mask=pad_mask)
            quant, indices = self.quant(self.pre_q(z))
            xv = x[..., :V].to(dtype=quant.dtype)
            if not atoms_to_decoder:
                xv = torch.zeros_like(xv)
            d_in = self.post_q(torch.cat([xv, quant], dim=-1))
            h = self.dec(d_in + pe, src_key_padding_mask=pad_mask)
            recon = self._project_coordinates(h)
            return recon, indices


    def training_step(self, batch, batch_idx):
        x, m = batch
        recon, indices = self(x)

        target = x[..., -3:]
        mask = m.unsqueeze(-1)  # [B,T,1], float
        diff2 = (recon - target).square() * mask
        denom = mask.sum().clamp_min(1.0) * 3.0
        loss = diff2.sum() / denom

        self._util_buf.append(indices.detach().reshape(-1))
        if (batch_idx + 1) % self.util_every == 0:
            all_codes = torch.cat(self._util_buf, dim=0)
            self.util += torch.bincount(all_codes, minlength=self.util.numel())
            self._util_buf.clear()

        # Do NOT sync loss here; honest global loss is logged in MFUCallback periodically.
        self.log("train/loss_local", loss, prog_bar=False, sync_dist=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.grad_norm_every <= 0 or (self.global_step + 1) % self.grad_norm_every != 0:
            return
        grads = [
            p.grad.detach().float().norm(2)
            for p in self.parameters()
            if p.grad is not None
        ]
        if grads:
            grad_norm = torch.norm(torch.stack(grads), 2)
            self.log("train/grad_norm_local", grad_norm, prog_bar=False, sync_dist=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, m = batch
        recon, _ = self(x)
        mask = m.bool()
        target = x[..., -3:]

        # Calculate per-molecule RMSDs to match train_on_nabladft logic
        rmsds = []
        for b in range(x.size(0)):
            m_b = mask[b]
            if not m_b.any():
                continue
            # mse = mean squared error over atoms in this molecule
            mse = (recon[b][m_b] - target[b][m_b]).square().sum(dim=-1).mean()
            rmsds.append(torch.sqrt(mse))

        if not rmsds:
            return None

        avg_rmse = torch.stack(rmsds).mean()

        name = "val"
        if self.trainer and self.trainer.val_dataloaders:
            loaders = self.trainer.val_dataloaders
            if isinstance(loaders, list) and dataloader_idx < len(loaders):
                name = getattr(loaders[dataloader_idx], "dataset_name", f"val_{dataloader_idx}")

        self.log(f"val/{name}_rmse", avg_rmse, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
        return avg_rmse

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def on_train_epoch_end(self):
        if self._util_buf:
            all_codes = torch.cat(self._util_buf, dim=0)
            self.util += torch.bincount(all_codes, minlength=self.util.numel())
            self._util_buf.clear()

        if self.trainer.is_global_zero:
            self._print_util("train", self.util)
        self.util.zero_()

    def _print_util(self, title: str, util_tensor: torch.Tensor):
        levels = self.hparams.levels
        counts_nd = util_tensor.view(*levels)
        print(f"Code utilization ({title}):")
        for d, L_size in enumerate(levels):
            axes = tuple(i for i in range(len(levels)) if i != d)
            freq = (counts_nd.sum(dim=axes) / (counts_nd.sum() + 1e-6)).tolist()
            nonzero = sum(f > 0 for f in freq)
            print(f"  Dim {d}: {[round(float(f), 6) for f in freq]} non-zero {nonzero}/{L_size}")

    def configure_optimizers(self):
        decay, no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if SELECTIVE_DECAY and (param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()):
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups = [{"params": decay, "weight_decay": 0.01}]
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0})

        opt = torch.optim.AdamW(param_groups, lr=LR)

        t = self.hparams.total_steps
        warmup = max(1, int(0.1 * t))
        decay_start = int(0.9 * t)

        def wsd_lambda(s):
            if s < warmup:
                return s / warmup
            if s < decay_start:
                return 1.0
            tail = max(1, t - decay_start)
            progress = min(1.0, (s - decay_start) / tail)
            return 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

if __name__ == "__main__":
    ckpt_dir = _check_env_vars()

    print("=== ENV DDP ===")
    print("RANK", os.environ.get("RANK"))
    print("LOCAL_RANK", os.environ.get("LOCAL_RANK"))
    print("WORLD_SIZE", os.environ.get("WORLD_SIZE"))
    print("LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE"))
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("=== END ENV ===", flush=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        torch.cuda.empty_cache()
        torch.cuda.set_device(local_rank)
    except Exception as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cudaerror" in error_str:
            raise RuntimeError(f"CUDA OOM on GPU {local_rank} during initialization. This may indicate GPU {local_rank} has leftover memory from a previous process. Try clearing GPU memory or restarting the job.")
        if "system not yet initialized" in error_str or "802" in error_str or "cudagetdevicecount" in error_str:
            raise RuntimeError("CUDA initialization failed. This may indicate no GPUs are available or CUDA driver/runtime issue.")
        raise
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this job (no GPU allocated or driver/runtime issue).")

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    world = int(os.environ.get("WORLD_SIZE", 1))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() if torch.cuda.is_available() else 1))
    print(f"world: {world}, local_world: {local_world}")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    _print_startup_config(rank=rank, world=world, local_world=local_world)
    nrows_path = Path(PACKED_DIR, f"part_{rank}", "nrows.txt")
    if not nrows_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {nrows_path}\n"
            f"This usually means the configured packed data directory is missing or incomplete.\n"
            f"Check if {PACKED_DIR}/part_{rank}/ exists and contains the required files.\n"
            f"Source data should exist at the dataset-specific source configured in submit_train.sh."
        )
    nrows_rank = int(nrows_path.read_text().strip())
    steps_per_epoch = (nrows_rank + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    checkpoint_every_n_steps = max(1, math.ceil(steps_per_epoch * (CKPT_EVERY_PERCENT / 100.0)))
    stop_after_steps = 0
    if STOP_AFTER_EPOCH_FRACTION > 0:
        stop_after_steps = max(1, math.ceil(steps_per_epoch * STOP_AFTER_EPOCH_FRACTION))
    print(f"steps_per_epoch: {steps_per_epoch}, total_steps: {total_steps}")
    print(f"checkpoint_every_n_steps: {checkpoint_every_n_steps}")
    if stop_after_steps > 0:
        print(
            f"stop_after_epoch_fraction: {STOP_AFTER_EPOCH_FRACTION}, "
            f"stop_after_steps: {stop_after_steps}",
            flush=True,
        )
    model = MolModel(
        total_steps=total_steps,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        levels=LEVELS,
    )

    jid = os.environ.get("SLURM_JOB_ID", "local")

    ckpt_path = RESUME_CKPT_PATH
    resume_global_step = 0
    resume_data_epoch = 0
    resume_batches_this_epoch = 0
    target_max_steps = -1
    if ckpt_path is not None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            resume_global_step = int(ckpt["global_step"])
            checkpoint_epoch = ckpt.get("epoch", None)
            resume_data_epoch = resume_global_step // max(1, steps_per_epoch)
            resume_batches_this_epoch = resume_global_step % max(1, steps_per_epoch)
            progress_state = read_key_value_file(Path(LOG_DIR) / "progress.txt")
            progress_global_step = progress_state.get("global_step")
            if progress_global_step is not None and int(progress_global_step) == resume_global_step:
                resume_data_epoch = int(progress_state.get("resume_data_epoch", resume_data_epoch))
                resume_batches_this_epoch = int(
                    progress_state.get("resume_batches_this_epoch", resume_batches_this_epoch)
                )
            STARTING_EPOCH = resume_data_epoch
            os.environ["RESUME_GLOBAL_STEP"] = str(resume_global_step)
            os.environ["RESUME_DATA_EPOCH"] = str(resume_data_epoch)
            os.environ["RESUME_BATCHES_THIS_EPOCH"] = str(resume_batches_this_epoch)
            print(f"global_step: {resume_global_step}")
            print(f"resume_data_epoch: {resume_data_epoch}")
            print(f"resume_batches_this_epoch: {resume_batches_this_epoch}")
            print(f"starting_epoch: {STARTING_EPOCH}")
            if rank == 0:
                print(f"[RESUME] Using checkpoint: {ckpt_path}", flush=True)
                print(
                    f"[RESUME] Global step: {resume_global_step}, checkpoint_epoch: {checkpoint_epoch}, "
                    f"resume_data_epoch: {resume_data_epoch}, resume_batches_this_epoch: {resume_batches_this_epoch}, "
                    f"starting_epoch: {STARTING_EPOCH}",
                    flush=True,
                )
        except (IndexError, ValueError, KeyError) as e:
            if rank == 0:
                print(f"[RESUME] Warning: Could not parse checkpoint: {e}", flush=True)

    if stop_after_steps > 0:
        target_max_steps = resume_global_step + stop_after_steps
        print(f"target_max_steps: {target_max_steps}")

    loader_kwargs = dict(
        batch_size=None,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        in_order=True,
    )
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4

    print(loader_kwargs)
    train_loader = DataLoader(MolDataset(PACKED_DIR, epoch_file=EPOCH_FILE, nrows=nrows_rank), num_workers=NUM_WORKERS, **loader_kwargs)
    val_loaders = make_val_loaders(VAL_DIR, NUM_WORKERS)
    trainer_strategy = "auto" if NUM_DEVICES == 1 else DDPStrategy(find_unused_parameters=False, bucket_cap_mb=200)
    print(f"[TRAINER] devices={NUM_DEVICES} strategy={trainer_strategy}", flush=True)
    print(f"[TRAINER] profiler={LIGHTNING_PROFILER}", flush=True)

    trainer_logger = AimLogger(experiment=jid)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        max_steps=target_max_steps,
        accelerator="gpu",
        devices=NUM_DEVICES,
        num_nodes=1,
        strategy=trainer_strategy,
        precision="bf16-mixed",
        limit_train_batches=steps_per_epoch,
        logger=trainer_logger,
        # OPTIONAL: keep Lightning misc outputs rooted under your LOG_DIR
        default_root_dir=LOG_DIR,
        callbacks=[
            LearningRateMonitor("step"),
            CustomProgressBar(manual_total=steps_per_epoch),
            PrintResumeState(),
            MFUCallback(max_tokens=MAX_TOKENS),
            EpochFileCallback(EPOCH_FILE, steps_per_epoch),
            ProgressStateCallback(LOG_DIR, ckpt_dir, steps_per_epoch, checkpoint_every_n_steps),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="fsq-{epoch:02d}-{step:09d}",
                save_top_k=-1,
                every_n_train_steps=checkpoint_every_n_steps,
                save_last=False,
            ),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="fsq-epochend-{epoch:02d}-{step:09d}",
                save_top_k=-1,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
                save_last=True,
            ),
        ],
        log_every_n_steps=50,
        gradient_clip_val=0.1,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        profiler=None if LIGHTNING_PROFILER.lower() in ("", "none", "null", "false", "0") else LIGHTNING_PROFILER,
    )
    if ckpt_path is not None:
        print(f"Training from checkpoint: {ckpt_path}")
        trainer.fit(model, train_loader, val_loaders, ckpt_path=ckpt_path)
    else:
        print("Training from scratch")
        trainer.fit(model, train_loader, val_loaders)
        # if trainer.is_global_zero and trainer.profiler is not None:
        #     print("\n" + "="*20 + " PROFILER SUMMARY " + "="*20)
        #     print(trainer.profiler.summary())

    # print("[DATA PREPROC] per-worker preprocessing times are reported above.", flush=True)
