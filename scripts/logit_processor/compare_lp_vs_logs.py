#!/usr/bin/env python
"""
Compare SMILES failures reported in generation logs against a logit-processed run.

The script extracts all SMILES mentioned in log lines that contain
"smiles fails parsing" or "smiles mismatch", then cross-references them with the
passes and failures stored in an LP JSON result file. The final summary is
emitted as JSON and designed to highlight:

* SMILES the LP run correctly failed (caught mismatches from the log)
* SMILES the LP run still passed even though the log recorded a mismatch
* SMILES that failed in the LP run but never appeared in the log failures
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence

LOG_FAILURE_PATTERNS = {
    "smiles fails parsing": "parse_failure",
    "smiles mismatch": "mismatch",
}

LOG_TIMESTAMP_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)")
CANONICAL_RE = re.compile(r"canonical_smiles='([^']*)'")
GENERATED_RE = re.compile(r"generated_smiles='([^']*)'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SMILES failures from generation logs with LP results."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(
            "/auto/home/menuab/code/3DMolGen/outputs/gen_results/"
            "20251211_124743_m600_qwen_pre_2e_distinct/logs.txt"
        ),
        help="Path to the original generation log file.",
    )
    parser.add_argument(
        "--lp-json",
        type=Path,
        default=Path(
            "outputs/smoke/qwen_lp_top_p_sampling1_v39_qwen3_sampling_1000.json"
        ),
        help="Path to the LP run JSON result (must contain passes/failures arrays).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scripts/logit_processor/lp_vs_logs_summary.json",
        ),
        help="Path where the comparison JSON should be written.",
    )
    return parser.parse_args()


def parse_log_failures(log_path: Path) -> List[Dict[str, Any]]:
    """Return structured failure entries parsed from the log."""
    entries: List[Dict[str, Any]] = []
    pending_entry: Dict[str, Any] | None = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip("\n")
            timestamp_match = LOG_TIMESTAMP_RE.match(stripped)

            if timestamp_match:
                pending_entry = None  # reset before potentially starting a new entry
                normalized_line = stripped.lower()
                matched_reason = None
                for fragment, reason in LOG_FAILURE_PATTERNS.items():
                    if fragment in normalized_line:
                        matched_reason = reason
                        break

                if matched_reason:
                    entries.append(
                        {
                            "timestamp": timestamp_match.group(1),
                            "reason": matched_reason,
                            "canonical_smiles": None,
                            "generated_smiles": None,
                        }
                    )
                    pending_entry = entries[-1]
                continue

            if not pending_entry:
                continue

            canonical_match = CANONICAL_RE.search(stripped)
            if canonical_match:
                pending_entry["canonical_smiles"] = canonical_match.group(1)
                continue

            generated_match = GENERATED_RE.search(stripped)
            if generated_match:
                pending_entry["generated_smiles"] = generated_match.group(1)
                continue

            if stripped.startswith("generated_conformer"):
                pending_entry = None

    return entries


def load_lp_json(lp_json_path: Path) -> Dict[str, Any]:
    with lp_json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_lp_index(
    entries: Sequence[Dict[str, Any]], status: str
) -> Dict[str, DefaultDict[str, List[Dict[str, Any]]]]:
    """Index LP entries by prompt/generated SMILES for quick lookup."""
    index: Dict[str, DefaultDict[str, List[Dict[str, Any]]]] = {
        "prompt": defaultdict(list),
        "generated": defaultdict(list),
    }
    for entry in entries:
        normalized = {
            "prompt_smiles": entry.get("prompt_smiles"),
            "generated_smiles": entry.get("generated_smiles"),
            "smiles_exact_match": entry.get("smiles_exact_match"),
            "parse_success": entry.get("parse_success"),
            "issues": entry.get("issues"),
            "parse_error": entry.get("parse_error"),
            "status": status,
        }
        prompt = normalized["prompt_smiles"]
        generated = normalized["generated_smiles"]

        if prompt:
            index["prompt"][prompt].append(normalized)
        if generated and generated != prompt:
            index["generated"][generated].append(normalized)
    return index


def flatten_lp_index(
    index: Dict[str, DefaultDict[str, List[Dict[str, Any]]]], source: str
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for smiles, records in index[source].items():
        for record in records:
            flattened.append(
                {
                    "smiles": smiles,
                    "lp_source": source,
                    "status": record["status"],
                    "smiles_exact_match": record.get("smiles_exact_match"),
                    "parse_success": record.get("parse_success"),
                    "issues": record.get("issues"),
                    "parse_error": record.get("parse_error"),
                }
            )
    return flattened


def expand_log_smiles(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each log entry into per-SMILES records (canonical + generated)."""
    expanded: Dict[tuple[str, str], Dict[str, Any]] = {}
    for entry in entries:
        for field in ("canonical_smiles", "generated_smiles"):
            smiles = entry.get(field)
            if not smiles:
                continue
            key = (smiles, field)
            expanded.setdefault(
                key,
                {
                    "smiles": smiles,
                    "log_source": field.replace("_smiles", ""),
                    "log_reason": entry["reason"],
                    "timestamp": entry["timestamp"],
                },
            )
    return list(expanded.values())


def find_matches(
    log_records: Sequence[Dict[str, Any]],
    lp_index: Dict[str, DefaultDict[str, List[Dict[str, Any]]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate intersection summaries."""
    caught: List[Dict[str, Any]] = []
    missed: List[Dict[str, Any]] = []
    lp_pass_index = lp_index["pass"]
    lp_fail_index = lp_index["fail"]

    def gather(
        table: Dict[str, DefaultDict[str, List[Dict[str, Any]]]],
        smiles: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for source in ("prompt", "generated"):
            results.extend(table[source].get(smiles, []))
        return results

    for log_entry in log_records:
        log_smiles = log_entry["smiles"]
        fail_matches = gather(lp_fail_index, log_smiles)
        pass_matches = gather(lp_pass_index, log_smiles)

        if fail_matches:
            for match in fail_matches:
                caught.append(
                    {
                        **log_entry,
                        "lp_source": (
                            "prompt"
                            if match.get("prompt_smiles") == log_smiles
                            else "generated"
                        ),
                        "lp_status": "fail",
                        "smiles_exact_match": match.get("smiles_exact_match"),
                        "parse_success": match.get("parse_success"),
                        "issues": match.get("issues"),
                        "parse_error": match.get("parse_error"),
                    }
                )
        elif pass_matches:
            for match in pass_matches:
                missed.append(
                    {
                        **log_entry,
                        "lp_source": (
                            "prompt"
                            if match.get("prompt_smiles") == log_smiles
                            else "generated"
                        ),
                        "lp_status": "pass",
                        "smiles_exact_match": match.get("smiles_exact_match"),
                        "parse_success": match.get("parse_success"),
                    }
                )

    return {"caught": caught, "missed": missed}


def main() -> None:
    args = parse_args()

    log_entries = parse_log_failures(args.log_file)
    log_smiles_records = expand_log_smiles(log_entries)
    log_smiles_set = {record["smiles"] for record in log_smiles_records}
    log_reason_counter = Counter(entry["reason"] for entry in log_entries)

    lp_payload = load_lp_json(args.lp_json)
    lp_pass_index = build_lp_index(lp_payload.get("passes", []), status="pass")
    lp_fail_index = build_lp_index(lp_payload.get("failures", []), status="fail")

    matches = find_matches(
        log_smiles_records,
        {"pass": lp_pass_index, "fail": lp_fail_index},
    )

    lp_fail_flat = flatten_lp_index(lp_fail_index, "prompt") + flatten_lp_index(
        lp_fail_index, "generated"
    )
    lp_new_failures = [
        entry
        for entry in lp_fail_flat
        if entry["smiles"] not in log_smiles_set
    ]

    output_payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "log_file": str(args.log_file),
            "lp_json": str(args.lp_json),
            "total_lp_passes": len(lp_payload.get("passes", [])),
            "total_lp_failures": len(lp_payload.get("failures", [])),
        },
        "log_failures": {
            "total_events": len(log_entries),
            "unique_smiles": len(log_smiles_set),
            "by_reason": dict(log_reason_counter),
        },
        "intersections": {
            "lp_caught_failures": matches["caught"],
        },
        "misses": {
            "lp_missed_failures": matches["missed"],
        },
        "lp_only": {
            "lp_new_failures": lp_new_failures,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, sort_keys=False)

    print(f"Wrote comparison summary to {args.output}")


if __name__ == "__main__":
    main()

