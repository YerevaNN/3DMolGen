#!/usr/bin/env python3
"""
Simple smoke test for the SMILES encoder/decoder.

Loads random molecules from the drugs summary dataset, runs encode/decode,
and prints the enriched text alongside its stripped version.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

from rdkit import Chem

from molgen3D.config.paths import get_data_path
from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_v2,
    encode_cartesian_v2,
    strip_smiles,
)
from molgen3D.data_processing.utils import filter_mols
from molgen3D.utils.utils import load_pkl


def _iter_random_drugs(
    drugs_summ_path: Path,
    count: Optional[int],
    seed: int,
) -> Iterable[Tuple[str, Chem.Mol, str]]:
    """Yield up to `count` (SMILES, Mol) pairs sampled at random."""
    with drugs_summ_path.open("r") as fh:
        summary = json.load(fh)

    items = list(summary.items())
    rng = random.Random(seed)
    rng.shuffle(items)

    base_dir = drugs_summ_path.parent
    yielded = 0
    for _, entry in items:
        if count is not None and yielded >= count:
            break

        rel_pickle = entry.get("pickle_path") or entry.get("path")
        if rel_pickle is None:
            continue

        pickle_path = Path(rel_pickle)
        if not pickle_path.is_absolute():
            pickle_path = base_dir / pickle_path
        if not pickle_path.exists():
            continue

        mol_obj = load_pkl(pickle_path)
        mols = filter_mols(mol_obj, failures={}, max_confs=1)
        if not mols:
            continue

        geom_smiles = mol_obj.get("smiles") or mol_obj.get("geom_smiles")
        if geom_smiles is None:
            geom_smiles = Chem.MolToSmiles(
                mols[0],
                canonical=True,
                isomericSmiles=True,
            )

        for mol in mols:
            mol_no_h = Chem.RemoveHs(mol)
            if mol_no_h.GetNumConformers() == 0:
                continue
            yield geom_smiles, mol_no_h, str(pickle_path)
            yielded += 1
            if count is not None and yielded >= count:
                break


CHIRAL_PATTERN = re.compile(r"@")
EXPLICIT_H_PATTERN = re.compile(r"\[[^\]]*H[0-9]*[^\]]*\]")
CHARGE_PATTERN = re.compile(r"\[[^\]]*[+-][0-9]*[^\]]*\]")
ISOMERIC_PATTERN = re.compile(r"[\\/]")


def _has_canonical_chirality(smiles: str) -> bool:
    return bool(CHIRAL_PATTERN.search(smiles))


def _has_explicit_h(smiles: str) -> bool:
    return bool(EXPLICIT_H_PATTERN.search(smiles))


def _has_charge(smiles: str) -> bool:
    return bool(CHARGE_PATTERN.search(smiles))


def _has_isomeric_bond(smiles: str) -> bool:
    return bool(ISOMERIC_PATTERN.search(smiles))


def _maybe_truncate(text: str, limit: Optional[int]) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "…"


def run_smoke_test(args: argparse.Namespace) -> None:
    drugs_summ_path = (
        Path(args.drugs_summ).expanduser().resolve()
        if args.drugs_summ is not None
        else get_data_path("drugs_summary")
    )

    if not drugs_summ_path.exists():
        raise FileNotFoundError(
            f"Drugs summary file not found at {drugs_summ_path}. "
            "Override via --drugs-summ or update paths.yaml."
        )

    print(f"Using drugs summary: {drugs_summ_path}")
    print(f"Sampling {args.samples} molecules (seed={args.seed})\n")

    generator = _iter_random_drugs(drugs_summ_path, args.samples, args.seed)

    def _process_record(idx: int, src_smiles: str, mol: Chem.Mol, pickle_path: str) -> dict:
        enriched, canonical = encode_cartesian_v2(mol, precision=args.precision)
        stripped = strip_smiles(enriched)
        decode_cartesian_v2(enriched)
        flags = []
        if _has_canonical_chirality(canonical):
            flags.append("chiral")
        if _has_explicit_h(canonical):
            flags.append("explicit-H")
        if _has_charge(canonical):
            flags.append("charged")
        if _has_isomeric_bond(canonical):
            flags.append("isomeric")
        return {
            "index": idx,
            "source": src_smiles,
            "canonical": canonical,
            "enriched": enriched,
            "stripped": stripped,
            "pickle": pickle_path,
            "flags": flags,
        }

    flagged = []
    counts = {"chiral": 0, "explicit-H": 0, "charged": 0, "isomeric": 0}
    idx = 0
    try:
        for src_smiles, mol, pickle_path in generator:
            idx += 1
            record = _process_record(idx, src_smiles, mol, pickle_path)
            for flag in record["flags"]:
                counts[flag] += 1
            if record["flags"]:
                flagged.append(record)
    except StopIteration:
        pass

    def _fmt_flags(flags: list[str]) -> str:
        return f" [{' & '.join(flags)}]" if flags else ""

    if not flagged:
        print("No critical molecules (chiral, explicit-H, charged, isomeric) found "
              f"in the first {idx} samples.")
        return

    print("Critical molecules detected:\n")
    for record in flagged:
        print(f"Sample {record['index']:03d}{_fmt_flags(record['flags'])}")
        print(f"  source_smiles : {record['source']}")
        print(f"  canonical     : {record['canonical']}")
        print(f"  enriched      : {_maybe_truncate(record['enriched'], args.truncate_enriched)}")
        print(f"  stripped      : {record['stripped']}")
        print(f"  rdkit_pickle  : {record['pickle']}")
        print("-" * 80)

    print("Summary (counts within sampled subset):")
    for flag, value in counts.items():
        print(f"  {flag:10s}: {value}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode 3D molecules from drugs summary and print results.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of molecules to inspect (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling (default: 0).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Coordinate precision passed to encode_cartesian_v2 (default: 4).",
    )
    parser.add_argument(
        "--drugs-summ",
        type=str,
        default=os.environ.get("GEOM_DRUGS_SUMM"),
        help="Optional path to summary_drugs.json; overrides paths.yaml/data config.",
    )
    parser.add_argument(
        "--truncate-enriched",
        type=int,
        default=100,
        help="Maximum number of characters to display for each enriched string "
        "(default: 100, set <=0 to disable truncation).",
    )
    return parser


if __name__ == "__main__":
    run_smoke_test(build_argparser().parse_args())

