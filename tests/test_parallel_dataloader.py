import json
import re
from pathlib import Path
from typing import Iterable, List

import pytest

from molgen3D.config.paths import get_tokenizer_path
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_dataloader,
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


def _create_validation_jsonl(path: Path, count: int = 8) -> List[str]:
    canonical_entries: List[str] = []
    with path.open("w", encoding="utf-8") as fp:
        for idx in range(count):
            canonical = f"C{idx}"
            # Create embedded SMILES that fit within seq_len constraints and min_emb_len
            # For seq_len=64, max_unit_tokens=62, and min_emb_len=16
            embedded = f"[H]C{idx}[H]" + "X" * 9  # Make it at least 16 chars but not too long
            obj = {
                "canonical_smiles": canonical,
                "embedded_smiles": embedded,
            }
            fp.write(json.dumps(obj) + "\n")
            canonical_entries.append(canonical)
    return canonical_entries


def _extract_canonical_sequences(tensor, tokenizer=None) -> List[str]:
    flattened = []
    data = tensor.tolist()
    if not data:
        return []
    if isinstance(data[0], list):
        for row in data:
            flattened.extend(row)
    else:
        flattened = data
    if tokenizer is not None:
        decoded = tokenizer.decode(flattened, skip_special_tokens=False)
    else:
        decoded = "".join(chr(int(tok)) for tok in flattened)
    return [canonical.strip() for canonical in SMILES_PATTERN.findall(decoded)]


def _build_test_loader(
    data_file: Path,
    *,
    world_size: int,
    rank: int,
    batch_size: int = 1,
    seed: int = 0,
) -> Iterable:
    return build_dataloader(
        train_path=str(data_file),
        tokenizer_path=PARALLEL_TEST_TOKENIZER_PATH,
        seq_len=64,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle_lines=False,
        infinite=False,
        seed=seed,
        min_emb_len=16,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=None,
        world_size=world_size,
        rank=rank,
    )


def _create_validation_jsonls(
    tmp_dir: Path, file_count: int, entries_per_file: int
) -> tuple[list[str], List[str]]:
    canonical_entries: List[str] = []
    paths: list[str] = []
    idx = 0
    for file_idx in range(file_count):
        path = tmp_dir / f"validation_part_{file_idx}.jsonl"
        paths.append(str(path))
        with path.open("w", encoding="utf-8") as fp:
            for _ in range(entries_per_file):
                canonical = f"C{idx}"
                # Create embedded SMILES that fit within seq_len constraints and min_emb_len
                embedded = f"[H]C{idx}[H]" + "X" * 9  # Make it at least 16 chars but not too long
                obj = {
                    "canonical_smiles": canonical,
                    "embedded_smiles": embedded,
                }
                fp.write(json.dumps(obj) + "\n")
                canonical_entries.append(canonical)
                idx += 1
    return paths, canonical_entries


def _read_canonical_entries_from_files(paths: list[str]) -> List[str]:
    canonical_entries: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                canonical = obj.get("canonical_smiles")
                embedded = obj.get("embedded_smiles")
                if canonical and embedded:
                    canonical_entries.append(canonical.strip())
    return canonical_entries


STATS_REPORT: List[dict] = []


@pytest.fixture(scope="session", autouse=True)
def print_test_stats():
    yield
    if not STATS_REPORT:
        return
    print("\n=== Parallel dataloader test stats summary ===")
    for entry in STATS_REPORT:
        print(f"{entry['label']}:")
        print(f"  world_size={entry['world_size']}")
        print(f"  file_count={entry['file_count']}")
        print(f"  expected_total={entry['expected_total']}")
        print(f"  total_loaded={entry['total_loaded']}")
        print(f"  unique_loaded={entry['unique_loaded']}")
        print(f"  duplicates={entry['duplicates']}")
        print(f"  missing={entry['missing']}")
        for line in entry.get("per_rank_summary", []):
            print(f"    {line}")


@pytest.mark.parametrize("world_size", [2])
def test_parallel_dataloader_splits(tmp_path, world_size):
    data_file = tmp_path / "validation.jsonl"
    canonical_entries = _create_validation_jsonl(data_file, count=8)

    per_rank_seen: dict[int, List[str]] = {}
    for rank in range(world_size):
        loader = _build_test_loader(data_file, world_size=world_size, rank=rank)
        seen_for_rank: List[str] = []
        for batch in loader:
            inputs_dict, _ = batch
            seen_for_rank.extend(
                _extract_canonical_sequences(
                    inputs_dict["input"],
                    tokenizer=PARALLEL_TEST_TOKENIZER,
                )
            )
        per_rank_seen[rank] = seen_for_rank

    total_seen = sum(len(v) for v in per_rank_seen.values())
    unique_seen = set().union(*per_rank_seen.values())
    duplicates = total_seen - len(unique_seen)

    print(
        "Parallel loader stats:",
        f"total_known={len(canonical_entries)}",
        f"total_loaded={total_seen}",
        f"unique_loaded={len(unique_seen)}",
        f"repeated={duplicates}",
    )

    assert duplicates == 0
    assert len(unique_seen) >= world_size
    assert unique_seen.issubset(set(canonical_entries))
    for rank in range(world_size):
        assert per_rank_seen[rank], f"rank {rank} saw no data"
        assert set(per_rank_seen[rank]).issubset(set(canonical_entries))


def test_dataloader_resume_continues_sequence(tmp_path):
    data_file = tmp_path / "validation.jsonl"
    canonical_entries = _create_validation_jsonl(data_file, count=9)

    world_size = 2
    rank = 0
    loader = _build_test_loader(
        data_file, world_size=world_size, rank=rank, seed=17, batch_size=1
    )
    iterator = iter(loader)
    pre_resume: List[str] = []
    for _ in range(3):
        inputs_dict, _ = next(iterator)
        pre_resume.extend(
            _extract_canonical_sequences(
                inputs_dict["input"],
                tokenizer=PARALLEL_TEST_TOKENIZER,
            )
        )

    state = loader.state_dict()

    resumed_loader = _build_test_loader(
        data_file, world_size=world_size, rank=rank, seed=17, batch_size=1
    )
    resumed_loader.load_state_dict(state)
    over_resume: List[str] = []
    for batch in resumed_loader:
        inputs, _ = batch
        over_resume.extend(
            _extract_canonical_sequences(
                inputs["input"],
                tokenizer=PARALLEL_TEST_TOKENIZER,
            )
        )

    combined = pre_resume + over_resume
    assert len(set(over_resume)) == len(over_resume)
    assert set(combined).issubset(set(canonical_entries))
    expected_rank = canonical_entries[rank::world_size]
    print(
        "Resume loader stats:",
        f"total_expected={len(expected_rank)}",
        f"before_resume={len(pre_resume)}",
        f"after_resume={len(over_resume)}",
    )


@pytest.mark.parametrize("world_size", [8])
def test_parallel_dataloader_fsdp_scaleup(tmp_path, world_size):
    data_file = tmp_path / "validation_fsdp.jsonl"
    # choose multiple of world_size to avoid partial slices
    canonical_entries = _create_validation_jsonl(data_file, count=world_size * 4)
    file_count = 1

    per_rank_seen: dict[int, List[str]] = {}
    for rank in range(world_size):
        loader = _build_test_loader(data_file, world_size=world_size, rank=rank)
        seen: List[str] = []
        for batch in loader:
            inputs_dict, _ = batch
            seen.extend(
                _extract_canonical_sequences(
                    inputs_dict["input"],
                    tokenizer=PARALLEL_TEST_TOKENIZER,
                )
            )
        per_rank_seen[rank] = seen

    all_seen = []
    for rank in range(world_size):
        all_seen.extend(per_rank_seen[rank])

    unique_seen = set(all_seen)
    total_expected = len(canonical_entries)
    total_loaded = len(all_seen)
    missed = set(canonical_entries) - unique_seen
    duplicates = total_loaded - len(unique_seen)

    print("FSDP 8-rank report:")
    print(f"  expected_samples={total_expected}")
    print(f"  total_loaded={total_loaded}")
    print(f"  unique_loaded={len(unique_seen)}")
    print(f"  duplicates={duplicates}")
    print(f"  missing={len(missed)}, missing_list={sorted(missed)}")

    STATS_REPORT.append(
        {
            "label": "FSDP 8-rank",
            "world_size": world_size,
            "file_count": file_count,
            "expected_total": total_expected,
            "total_loaded": total_loaded,
            "unique_loaded": len(unique_seen),
            "duplicates": duplicates,
            "missing": len(missed),
        }
    )

    assert duplicates == 0
    assert len(unique_seen) >= world_size
    assert unique_seen.issubset(set(canonical_entries))
    for rank in range(world_size):
        assert per_rank_seen[rank], f"rank {rank} saw no data"
        assert set(per_rank_seen[rank]).issubset(set(canonical_entries))


@pytest.mark.parametrize("world_size", [4])
def test_multi_file_parallel_loader(tmp_path, world_size):
    HARD_CODED_PARALLEL_PATH = (
        Path("/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v2/processed_strings/parallel_test")
    )
    using_real_data = HARD_CODED_PARALLEL_PATH.is_dir()
    if HARD_CODED_PARALLEL_PATH.is_dir():
        data_files = sorted(str(p) for p in HARD_CODED_PARALLEL_PATH.glob("*.jsonl"))
        canonical_entries = _read_canonical_entries_from_files(data_files)
    else:
        file_count = 3
        entries_per_file = world_size * 5
        data_files, canonical_entries = _create_validation_jsonls(
            tmp_path, file_count=file_count, entries_per_file=entries_per_file
        )

    per_rank_seen = {}
    for rank in range(world_size):
        loader = build_dataloader(
            train_path=data_files,
            tokenizer_path=PARALLEL_TEST_TOKENIZER_PATH,
            seq_len=512,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            shuffle_lines=False,
            infinite=False,
            seed=rank,
            min_emb_len=16,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=None,
            world_size=world_size,
            rank=rank,
        )
        seen: List[str] = []
        for batch in loader:
            inputs_dict, _ = batch
            seen.extend(
                _extract_canonical_sequences(
                    inputs_dict["input"],
                    tokenizer=PARALLEL_TEST_TOKENIZER,
                )
            )
        per_rank_seen[rank] = seen

    all_seen = []
    for rank in range(world_size):
        all_seen.extend(per_rank_seen[rank])

    unique_seen = set(all_seen)
    total_expected = len(canonical_entries)
    total_loaded = len(all_seen)
    duplicates = total_loaded - len(unique_seen)
    missing = set(canonical_entries) - unique_seen

    file_count = len(data_files)
    expected_set = set(canonical_entries)
    canonical_set = sorted(expected_set)
    canon_total = len(canonical_set)
    seen_union: set[str] = set()
    per_rank_summary: list[str] = []
    for rank in range(world_size):
        seen_list = per_rank_seen[rank]
        rank_set = set(seen_list)
        rank_duplicates = len(seen_list) - len(rank_set)
        seen_union.update(rank_set)
        per_rank_summary.append(
            f"rank {rank}: loaded={len(seen_list)} unique={len(rank_set)} duplicates={rank_duplicates}"
        )
    print("Multi-file loader report:")
    print(f"  world_size={world_size}")
    print(f"  file_count={file_count}")
    print(f"  canonical_unique={canon_total}")
    print(f"  unique_loaded={len(unique_seen)}")
    print(f"  total_loaded={total_loaded}")
    print(f"  duplicates={duplicates}")
    print(f"  missing={len(missing)}")
    for line in per_rank_summary:
        print(f"    {line}")

    STATS_REPORT.append(
        {
            "label": "Multi-file loader",
            "world_size": world_size,
            "file_count": file_count,
            "expected_total": total_expected,
            "total_loaded": total_loaded,
            "unique_loaded": len(unique_seen),
            "duplicates": duplicates,
            "missing": len(missing),
            "per_rank_summary": per_rank_summary,
        }
    )

    if not using_real_data:
        assert duplicates == 0
        assert len(missing) == 0
        assert seen_union == set(canonical_set)
    else:
        assert len(missing) <= 35000  # Allow more missing entries for real data with long 3D conformers
