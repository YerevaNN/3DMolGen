"""Example-level batching with a shared, arm-independent batch composition.

The primary comparison is matched by molecule-conformer examples, so batch
*membership* must not depend on sequence length. The order of examples is one
seeded permutation of the global example index, identical for both arms; a
global batch is a fixed slice of that permutation. Only the split of a global
batch into micro-batches uses length bucketing, and only to cut padding -- both
arms still see exactly the same examples in exactly the same optimizer step.

Examples live in a uint16 token blob with a u64 offset table, memory-mapped so
that dataloader workers share pages instead of copying a multi-gigabyte list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Token ids are passed in by the caller; this module stays vocabulary-agnostic.

LABEL_IGNORE = -100


def pack_binary(
    example_files: Sequence[str | Path],
    destination: str | Path,
    token_to_id: Dict[str, int],
    unk_id: int,
    stem: str,
) -> Dict[str, object]:
    """Convert JSONL example shards into a memory-mappable token blob.

    Shards are consumed in sorted order, so the row index in the blob is the
    stable global example index and is identical across arms.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if len(token_to_id) > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocabulary too large for uint16 packing")

    tokens_path = destination / f"{stem}.tokens.u16"
    offsets_path = destination / f"{stem}.offsets.u64"
    ids_path = destination / f"{stem}.example_ids.txt"

    offsets: List[int] = [0]
    unknown = 0
    written = 0
    with open(tokens_path, "wb") as blob, open(ids_path, "w") as ids:
        for shard in sorted(Path(path) for path in example_files):
            with open(shard) as handle:
                for line in handle:
                    record = json.loads(line)
                    pieces = record["tokens"].split(" ")
                    row = np.fromiter(
                        (token_to_id.get(piece, unk_id) for piece in pieces),
                        dtype=np.uint16,
                        count=len(pieces),
                    )
                    unknown += int((row == unk_id).sum())
                    blob.write(row.tobytes())
                    offsets.append(offsets[-1] + row.size)
                    ids.write(f"{record['example_id']}\n")
                    written += 1

    np.asarray(offsets, dtype=np.uint64).tofile(offsets_path)
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    report = {
        "stem": stem,
        "examples": written,
        "tokens": int(lengths.sum()),
        "unknown_tokens": unknown,
        "length_mean": float(lengths.mean()) if written else 0.0,
        "length_max": int(lengths.max()) if written else 0,
        "length_histogram": {
            str(int(bucket)): int(count)
            for bucket, count in zip(*np.unique(lengths // 64 * 64, return_counts=True))
        },
    }
    (destination / f"{stem}.pack_report.json").write_text(json.dumps(report, indent=1))
    return report


@dataclass
class PackedExamples:
    """A memory-mapped view of one arm's packed examples."""

    directory: Path
    stem: str

    def __post_init__(self) -> None:
        self.offsets = np.fromfile(
            self.directory / f"{self.stem}.offsets.u64", dtype=np.uint64
        ).astype(np.int64)
        self.tokens = np.memmap(
            self.directory / f"{self.stem}.tokens.u16", dtype=np.uint16, mode="r"
        )
        self.lengths = np.diff(self.offsets)

    def __len__(self) -> int:
        return int(self.lengths.size)

    def __getitem__(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.tokens[start:end], dtype=np.int64)

    def example_ids(self) -> List[str]:
        path = self.directory / f"{self.stem}.example_ids.txt"
        return path.read_text().splitlines()


def epoch_order(n_examples: int, seed: int, epoch: int) -> np.ndarray:
    """Return the shared example order for one epoch.

    Depends only on ``(n_examples, seed, epoch)``, never on sequence lengths, so
    both arms traverse identical examples in identical order.
    """
    generator = np.random.default_rng([seed, epoch])
    return generator.permutation(n_examples)


@dataclass
class MicroBatch:
    """One forward/backward unit."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    geometry_mask: torch.Tensor
    example_indices: np.ndarray

    @property
    def enabled_tokens(self) -> int:
        return int((self.labels != LABEL_IGNORE).sum().item())

    @property
    def predicted_tokens(self) -> int:
        """Enabled tokens the loss actually scores.

        Next-token loss drops the label at position 0 of every row, so this is
        ``enabled_tokens`` minus one per row. Normalizing by this -- and not by
        ``enabled_tokens`` -- is what makes every scored token carry weight
        exactly 1.0.
        """
        return self.enabled_tokens - self.labels.size(0)


def _collate(
    packed: PackedExamples,
    indices: Sequence[int],
    pad_id: int,
    xyz_id: int,
) -> MicroBatch:
    rows = [packed[int(index)] for index in indices]
    width = max(row.size for row in rows)
    batch = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for position, row in enumerate(rows):
        batch[position, : row.size] = row
        mask[position, : row.size] = 1

    input_ids = torch.from_numpy(batch)
    attention_mask = torch.from_numpy(mask)
    labels = input_ids.masked_fill(attention_mask == 0, LABEL_IGNORE)

    # Everything strictly after <XYZ> is the geometry suffix; the flag lets the
    # trainer report suffix loss separately from total loss.
    after_xyz = (input_ids == xyz_id).cumsum(dim=1) > 0
    geometry = after_xyz & (input_ids != xyz_id) & (attention_mask == 1)

    return MicroBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        geometry_mask=geometry,
        example_indices=np.asarray(indices),
    )


def iter_global_batches(
    packed: PackedExamples,
    order: np.ndarray,
    global_batch_size: int,
    micro_batch_size: int,
    pad_id: int,
    xyz_id: int,
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = True,
) -> Iterator[Tuple[int, List[MicroBatch]]]:
    """Yield ``(step, micro_batches_for_this_rank)``.

    The global batch is a fixed slice of ``order``. It is split across ranks by
    stride, then each rank sorts its own share by length before cutting
    micro-batches -- padding shrinks, membership does not move.
    """
    if global_batch_size % world_size:
        raise ValueError(
            f"global batch {global_batch_size} must divide evenly across "
            f"{world_size} ranks, otherwise ranks disagree on micro-batch count"
        )
    total_steps = len(order) // global_batch_size
    if not drop_last and len(order) % global_batch_size:
        total_steps += 1

    for step in range(total_steps):
        window = order[step * global_batch_size : (step + 1) * global_batch_size]
        if window.size == 0:
            continue
        mine = window[rank::world_size]
        if mine.size == 0:
            yield step, []
            continue
        ordered = mine[np.argsort(packed.lengths[mine], kind="stable")]
        micro = [
            _collate(packed, ordered[start : start + micro_batch_size], pad_id, xyz_id)
            for start in range(0, ordered.size, micro_batch_size)
        ]
        yield step, micro


def validation_batches(
    packed: PackedExamples,
    micro_batch_size: int,
    pad_id: int,
    xyz_id: int,
    limit: Optional[int] = None,
    seed: int = 17,
) -> Iterator[MicroBatch]:
    """Deterministic, length-sorted validation batches."""
    count = len(packed)
    indices = np.arange(count)
    if limit is not None and limit < count:
        indices = np.random.default_rng(seed).choice(count, size=limit, replace=False)
        indices.sort()
    indices = indices[np.argsort(packed.lengths[indices], kind="stable")]
    for start in range(0, indices.size, micro_batch_size):
        yield _collate(packed, indices[start : start + micro_batch_size], pad_id, xyz_id)
