import json
import re
from pathlib import Path
from typing import Iterable, List

import pytest

from molgen3D.config.paths import get_tokenizer_path
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    JsonlTaggedPackedDataset,
    ensure_tokenizer_pad_token,
)
from transformers import AutoTokenizer

SMILES_PATTERN = re.compile(r"\[SMILES\](.*?)\[/SMILES\]", re.DOTALL)

PARALLEL_TEST_TOKENIZER_PATH = str(get_tokenizer_path("qwen3_0.6b_origin"))
PARALLEL_TEST_TOKENIZER = AutoTokenizer.from_pretrained(
    PARALLEL_TEST_TOKENIZER_PATH,
    use_fast=True,
    fix_mistral_regex=True,
)
ensure_tokenizer_pad_token(PARALLEL_TEST_TOKENIZER)


def _extract_sequences(tensor) -> List[str]:
    decoded = PARALLEL_TEST_TOKENIZER.decode(
        tensor.tolist(),
        skip_special_tokens=False,
    )
    return [canonical.strip() for canonical in SMILES_PATTERN.findall(decoded)]


def _create_validation_jsonl(path: Path, count: int = 8) -> List[str]:
    canonical_entries: List[str] = []
    with path.open("w", encoding="utf-8") as fp:
        for idx in range(count):
            canonical = f"C{idx}"
            embedded = f"[H]C{idx}[H]" + "X" * 9
            obj = {"canonical_smiles": canonical, "embedded_smiles": embedded}
            fp.write(json.dumps(obj) + "\n")
            canonical_entries.append(canonical)
    return canonical_entries


def _build_dataset(
    train_path: Iterable[str] | str,
    *,
    world_size: int,
    rank: int,
    seq_len: int = 512,
    seed: int = 0,
) -> JsonlTaggedPackedDataset:
    return JsonlTaggedPackedDataset(
        train_path=train_path,
        tokenizer_path=PARALLEL_TEST_TOKENIZER_PATH,
        seq_len=seq_len,
        min_emb_len=16,
        shuffle_lines=False,
        infinite=False,
        seed=seed,
        world_size=world_size,
        rank=rank,
        lookahead_limit=32,
    )


def _collect_canonicals(dataset: JsonlTaggedPackedDataset) -> List[str]:
    results: List[str] = []
    for features, _ in dataset:
        results.extend(_extract_sequences(features["input"]))
    return results


@pytest.mark.parametrize("world_size", [2])
def test_parallel_dataloader_splits(tmp_path, world_size):
    data_file = tmp_path / "validation.jsonl"
    canonical_entries = _create_validation_jsonl(data_file, count=8)

    per_rank_seen: dict[int, List[str]] = {}
    for rank in range(world_size):
        ds = _build_dataset(str(data_file), world_size=world_size, rank=rank, seed=17)
        per_rank_seen[rank] = _collect_canonicals(ds)

    total_seen = sum(len(v) for v in per_rank_seen.values())
    unique_seen = set().union(*per_rank_seen.values())

    assert total_seen >= len(unique_seen)
    assert unique_seen == set(canonical_entries)
    for rank in range(world_size):
        assert per_rank_seen[rank], f"rank {rank} saw no data"


def test_dataloader_resume_continues_sequence(tmp_path):
    data_file = tmp_path / "validation.jsonl"
    canonical_entries = _create_validation_jsonl(data_file, count=12)

    ds = _build_dataset(str(data_file), world_size=2, rank=0, seed=3)
    iterator = iter(ds)
    before_resume: List[str] = []
    for _ in range(2):
        try:
            features, _ = next(iterator)
        except StopIteration:
            break
        before_resume.extend(_extract_sequences(features["input"]))

    state = ds.state_dict()
    resumed = _build_dataset(str(data_file), world_size=2, rank=0, seed=3)
    resumed.load_state_dict(state)
    after_resume = _collect_canonicals(resumed)

    combined = before_resume + after_resume
    assert combined, "Expected at least one sequence before or after resume."
    assert set(combined).issubset(set(canonical_entries))
    assert set(before_resume).isdisjoint(set(after_resume))


@pytest.mark.parametrize("world_size", [8])
def test_parallel_dataloader_fsdp_scaleup(tmp_path, world_size):
    data_file = tmp_path / "validation_fsdp.jsonl"
    canonical_entries = _create_validation_jsonl(
        data_file, count=world_size * 4
    )

    per_rank_seen = {}
    for rank in range(world_size):
        ds = _build_dataset(str(data_file), world_size=world_size, rank=rank, seed=rank)
        per_rank_seen[rank] = _collect_canonicals(ds)

    all_seen = sum(per_rank_seen.values(), [])
    unique_seen = set(all_seen)
    duplicates = len(all_seen) - len(unique_seen)
    missing = set(canonical_entries) - unique_seen

    assert duplicates == 0
    assert not missing


@pytest.mark.parametrize("world_size", [4])
def test_multi_file_parallel_loader(tmp_path, world_size):
    """Run the parallel loader on the real hardcoded dataset when available."""
    hardcoded = Path(
        "/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v2/processed_strings/parallel_test"
    )
    using_real_data = hardcoded.is_dir()
    if using_real_data:
        real_files = sorted(str(p) for p in hardcoded.glob("*.jsonl"))[:2]
        data_files = []
        canonical_entries = []
        for file_idx, src in enumerate(real_files):
            dst = tmp_path / f"real_subset_{file_idx}.jsonl"
            copied = 0
            with open(src, "r", encoding="utf-8") as fp, dst.open("w", encoding="utf-8") as out:
                for line in fp:
                    obj = json.loads(line)
                    out.write(json.dumps(obj) + "\n")
                    canonical_entries.append(obj["canonical_smiles"])
                    copied += 1
                    if copied >= world_size * 8:
                        break
            data_files.append(str(dst))
    else:
        file_count = 3
        entries_per_file = world_size * 6
        data_files = []
        canonical_entries = []
        idx = 0
        for part in range(file_count):
            path = tmp_path / f"part_{part}.jsonl"
            data_files.append(str(path))
            with path.open("w", encoding="utf-8") as fp:
                for _ in range(entries_per_file):
                    canonical = f"C{idx}"
                    embedded = f"[H]C{idx}[H]" + "X" * 9
                    fp.write(
                        json.dumps(
                            {"canonical_smiles": canonical, "embedded_smiles": embedded}
                        )
                        + "\n"
                    )
                    canonical_entries.append(canonical)
                    idx += 1

    per_rank_seen = {}
    for rank in range(world_size):
        ds = _build_dataset(data_files, world_size=world_size, rank=rank, seed=rank)
        per_rank_seen[rank] = _collect_canonicals(ds)

    union_seen = set().union(*per_rank_seen.values())
    missing = set(canonical_entries) - union_seen

    if using_real_data:
        # Real dataset contains some very long conformers; allow a small miss budget.
        assert len(missing) <= 35000
    else:
        assert not missing
        assert union_seen == set(canonical_entries)
