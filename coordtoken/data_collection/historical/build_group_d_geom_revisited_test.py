#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


COORD_PATTERN = re.compile(r"<[^>]+>")
DEFAULT_REVISITED_PICKLE = Path("/auto/home/vover/revisited.pickle")
DEFAULT_OUTPUT_ROOT = Path("/nfs/h100/raid/chem/3D_big_data/grp_d")
DEFAULT_DATASET_NAME = "geom_revisited_test"

TEST_KEYS: tuple[str, ...] = ("test", "test_set", "test_data", "test_split")
TEXT_KEYS: tuple[str, ...] = ("enriched_text", "text", "sample_text")
TOP_LEVEL_SMILES_KEY = "__top_level_smiles__"
SMILES_KEYS: tuple[str, ...] = (
    "smiles",
    "smi",
    "canonical_smiles",
    "canonical_smi",
    "base_smiles",
    "corrected_smi",
    "geom_smiles",
    "geom_smiles_c",
    TOP_LEVEL_SMILES_KEY,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def alias_candidates(path: Path) -> list[Path]:
    candidates = [path]
    s = str(path)

    if s.startswith("/home/"):
        candidates.append(Path("/auto") / s.lstrip("/"))
    elif s.startswith("/auto/home/"):
        candidates.append(Path(s[len("/auto") :]))

    if s.startswith("/raid/"):
        candidates.append(Path("/nfs/h100") / s.lstrip("/"))
    elif s.startswith("/nfs/h100/raid/"):
        candidates.append(Path(s[len("/nfs/h100") :]))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_existing_path(path: Path) -> Path:
    for candidate in alias_candidates(path):
        if candidate.exists():
            return candidate
    return path


def resolve_output_path(path: Path) -> Path:
    for candidate in alias_candidates(path):
        ancestor = candidate
        while ancestor != ancestor.parent and not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.exists():
            return candidate
    return path


def extract_smiles(enriched_text: str) -> str:
    return COORD_PATTERN.sub("", enriched_text)


def is_dataframe_like(obj: Any) -> bool:
    return hasattr(obj, "to_dict") and hasattr(obj, "columns")


def find_string_field(obj: Any, candidate_keys: Sequence[str], max_depth: int = 3) -> str | None:
    if max_depth < 0 or isinstance(obj, str):
        return None

    if isinstance(obj, dict):
        for key in candidate_keys:
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
        for value in obj.values():
            if isinstance(value, (dict, list, tuple)):
                result = find_string_field(value, candidate_keys, max_depth=max_depth - 1)
                if result:
                    return result

    if isinstance(obj, (list, tuple)):
        for value in obj[:10]:
            if isinstance(value, (dict, list, tuple)):
                result = find_string_field(value, candidate_keys, max_depth=max_depth - 1)
                if result:
                    return result

    return None


def looks_like_columnar_dict(obj: dict[Any, Any]) -> bool:
    if not obj:
        return False
    values = list(obj.values())
    if not all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in values):
        return False
    lengths = {len(value) for value in values}
    return len(lengths) == 1


def columnar_dict_to_records(obj: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    size = len(next(iter(obj.values())))
    return [{key: value[idx] for key, value in obj.items()} for idx in range(size)]


def select_test_payload(obj: Any) -> tuple[Any, str]:
    if isinstance(obj, dict):
        for key in TEST_KEYS:
            if key in obj:
                return obj[key], key
    return obj, "<all>"


def normalize_records(obj: Any) -> list[Any]:
    if is_dataframe_like(obj):
        return obj.to_dict(orient="records")

    if isinstance(obj, dict):
        if looks_like_columnar_dict(obj):
            return columnar_dict_to_records(obj)
        if all(isinstance(value, dict) for value in obj.values()):
            records: list[dict[str, Any]] = []
            for key, value in obj.items():
                record = dict(value)
                if isinstance(key, str) and TOP_LEVEL_SMILES_KEY not in record:
                    record[TOP_LEVEL_SMILES_KEY] = key
                records.append(record)
            return records
        if all(not isinstance(value, (list, tuple, dict)) for value in obj.values()):
            return [obj]
        for key in TEST_KEYS:
            if key in obj:
                return normalize_records(obj[key])
        raise ValueError(
            "Could not normalize revisited payload into records. "
            f"Top-level keys: {list(obj.keys())[:20]}"
        )

    if isinstance(obj, (list, tuple)):
        return list(obj)

    raise ValueError(f"Unsupported revisited payload type: {type(obj)}")


def encode_mol_to_enriched(mol: Any) -> str:
    from train.utils import encode_cartesian_v2

    return encode_cartesian_v2(mol, precision=4)


def extract_reference_entries(record: Any) -> list[tuple[str | None, str | None]]:
    if isinstance(record, str):
        if "<" in record and ">" in record:
            enriched_text = record
            smiles = extract_smiles(enriched_text)
            return [(smiles or None, enriched_text)]
        return [(record, None)]

    if isinstance(record, dict):
        confs = record.get("confs")
        if isinstance(confs, (list, tuple)) and confs:
            fallback_smiles = find_string_field(record, SMILES_KEYS)
            entries: list[tuple[str | None, str | None]] = []
            for mol in confs:
                enriched_text = encode_mol_to_enriched(mol)
                entry_smiles = extract_smiles(enriched_text) or fallback_smiles
                entries.append((entry_smiles or None, enriched_text))
            return entries

    enriched_text = find_string_field(record, TEXT_KEYS)
    smiles = find_string_field(record, SMILES_KEYS)
    if enriched_text:
        smiles = extract_smiles(enriched_text) or smiles

    if smiles == "":
        smiles = None
    if enriched_text == "":
        enriched_text = None
    return [(smiles, enriched_text)]


def load_revisited_rows(pickle_path: Path) -> tuple[list[dict[str, str]], str]:
    resolved_path = resolve_existing_path(pickle_path)
    if not resolved_path.exists():
        raise ValueError(f"Revisited pickle not found: {pickle_path}")

    with resolved_path.open("rb") as f:
        raw = pickle.load(f)

    payload, payload_name = select_test_payload(raw)
    records = normalize_records(payload)
    log(f"Loaded revisited pickle from {resolved_path}")
    log(f"Using payload as Group D test source: {payload_name}")
    log(f"Top-level records to inspect: {len(records):,}")

    output_rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    unique_smiles: set[str] = set()

    for record_idx, record in enumerate(
        tqdm(records, desc="Converting revisited records", mininterval=5),
        start=1,
    ):
        conf_id = 0
        for smiles, enriched_text in extract_reference_entries(record):
            if not enriched_text or enriched_text in seen_texts:
                continue
            conf_id += 1
            seen_texts.add(enriched_text)
            if smiles:
                unique_smiles.add(smiles)
            output_rows.append(
                {
                    "name": f"geom_rev_{record_idx}_{conf_id}",
                    "enriched_text": enriched_text,
                }
            )

    if not output_rows:
        raise ValueError("No enriched_text rows could be extracted from the revisited pickle.")

    log(f"Unique Group D enriched_text rows: {len(output_rows):,}")
    log(f"Unique Group D base SMILES: {len(unique_smiles):,}")
    return output_rows, payload_name


def save_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "enriched_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary(summary_csv: Path, dataset_name: str, row_count: int, payload_name: str) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "source_payload", "rows"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset_name,
                "source_payload": payload_name,
                "rows": row_count,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the geom revisited pickle into flat enriched_text CSV format and save it "
            "as a Group D test dataset."
        )
    )
    parser.add_argument(
        "--revisited-pickle",
        type=Path,
        default=DEFAULT_REVISITED_PICKLE,
        help="Path to the revisited pickle file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Group D root directory. The CSV will be written under test/.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Output CSV stem for the Group D test dataset.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path. Defaults to <output-root>/grp_d_summary.csv.",
    )
    args = parser.parse_args()

    output_root = resolve_output_path(args.output_root)
    output_csv = output_root / "test" / f"{args.dataset_name}.csv"
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = output_root / "grp_d_summary.csv"
    else:
        summary_csv = resolve_output_path(summary_csv)

    rows, payload_name = load_revisited_rows(args.revisited_pickle)
    save_rows(rows, output_csv)
    save_summary(summary_csv, args.dataset_name, len(rows), payload_name)

    log(f"Saved Group D test CSV to {output_csv}")
    log(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
