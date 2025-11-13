# pretraining/data/dataloaders/jsonl_tagged_packed_simple.py
from __future__ import annotations
import io, os, json, glob, random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
from loguru import logger

from molgen3D.training.pretraining.dataprocessing.text_processing import build_unit, is_valid_unit, ChunkPacker
from molgen3D.training.pretraining.dataprocessing.utils import expand_paths, build_line_index, read_line_at

class JsonlTaggedPackedDataset(IterableDataset):
    """
    JSONL -> unit string -> HF tokenize(no specials) -> pack to 2048.
    - BOS at start of each chunk
    - EOS between units
    - no unit splitting; over-long units dropped
    Resume works with torchdata.StatefulDataLoader (if available).
    """
    def __init__(
        self,
        train_path: Union[str, Sequence[str]],
        tokenizer_path: str,
        seq_len: int = 2048,
        min_emb_len: int = 16,
        shuffle_lines: bool = True,
        infinite: bool = True,
        seed: int = 1234,
    ):
        super().__init__()
        self.files = expand_paths(train_path)
        self.idxs = [build_line_index(p) for p in self.files]

        # world/rank from env (torchrun/Titan sets these)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))

        # tokenizer (HF only for simplicity)
        self.tk = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.bos_id = int(self.tk.bos_token_id)
        self.eos_id = int(self.tk.eos_token_id)
        if self.bos_id is None or self.eos_id is None:
            raise RuntimeError("Tokenizer must define bos_token_id and eos_token_id.")

        self.seq_len = int(seq_len)
        self.min_emb_len = int(min_emb_len)
        self.shuffle_lines = bool(shuffle_lines)
        self.infinite = bool(infinite)

        self._epoch = 0
        self._rng = random.Random(seed + self.rank * 97)
        self._packer = ChunkPacker(self.seq_len, self.bos_id, self.eos_id)
        self._start_k = 0     # resume cursor within worker's pair list
        self._pairs_total = 0

    def _pairs_for_epoch(self) -> List[Tuple[int,int]]:
        pairs: List[Tuple[int,int]] = []
        base = 0
        for fi, idx in enumerate(self.idxs):
            n = len(idx)
            order = list(range(n))
            if self.shuffle_lines:
                self._rng.shuffle(order)
            for li in order:
                g = base + li
                if (g % self.world_size) == self.rank:
                    pairs.append((fi, li))
            base += n
        return pairs

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1

        while True:
            all_pairs = self._pairs_for_epoch()
            worker_pairs = [p for i, p in enumerate(all_pairs) if (i % nworkers) == wid]
            self._pairs_total = len(worker_pairs)
            k = self._start_k

            fps = [open(p, "rb") for p in self.files]
            try:
                while k < self._pairs_total:
                    fi, li = worker_pairs[k]
                    k += 1
                    self._start_k = k

                    raw = read_line_at(fps[fi], int(self.idxs[fi][li]))
                    if not raw: continue
                    try:
                        obj = json.loads(raw)
                        canon = (obj.get("canonical_smiles") or "").strip()
                        emb   = (obj.get("embedded_smiles") or "").strip()
                        if not is_valid_unit(canon, emb, min_emb_len=self.min_emb_len):
                            continue
                        unit = build_unit(canon, emb)
                    except Exception:
                        continue

                    # encode WITHOUT specials, then prepend EOS token
                    toks = self.tk.encode(unit, add_special_tokens=False)
                    toks = [self.eos_id] + toks  # EOS before each unit

                    # pack: try to add unit (will add as much as possible, may cut unit)
                    unit_completed = self._packer.try_add_unit(toks)

                    # If unit was cut off and we have pending content, yield completed blocks
                    if not unit_completed and self._packer.pending_unit:
                        for inp, lab in self._packer.yield_blocks():
                            yield inp, lab

                    # drain any full blocks
                    for inp, lab in self._packer.yield_blocks():
                        yield inp, lab
            finally:
                for f in fps: f.close()

            if not self.infinite:
                break
            self._epoch += 1
            # deterministic but different shuffle each epoch per rank
            self._rng.seed(1337 + self._epoch * 100003 + self.rank * 97)
            self._start_k = 0

    # ------ resume for StatefulDataLoader ------
    def state_dict(self) -> Dict:
        return {
            "epoch": self._epoch,
            "rng_state": self._rng.getstate(),
            "start_k": self._start_k,
            "pairs_total": self._pairs_total,
            "packer": self._packer.state_dict(),
        }

    def load_state_dict(self, s: Dict):
        self._epoch = int(s.get("epoch", 0))
        self._rng.setstate(s.get("rng_state", random.Random().getstate()))
        self._start_k = int(s.get("start_k", 0))
        self._pairs_total = int(s.get("pairs_total", 0))
        self._packer.load_state_dict(s.get("packer", {}))

# ---------- Titan factory ----------
def build_dataloader(
    train_path: Union[str, Sequence[str]],
    tokenizer_path: str,
    seq_len: int,
    batch_size: int,
    *,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_lines: bool = True,
    infinite: bool = True,
    seed: Optional[int] = None,
):
    ds = JsonlTaggedPackedDataset(
        train_path=train_path,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        shuffle_lines=shuffle_lines,
        infinite=infinite,
        seed=seed if seed is not None else 1234,
    )
    return StatefulDataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
