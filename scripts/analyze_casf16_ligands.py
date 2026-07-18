#!/usr/bin/env python3
"""Analyze CASF16 ligand MOL2 conformers with RDKit descriptors."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError as exc:  # pragma: no cover - exercised only outside RDKit envs.
    Chem = None
    Descriptors = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASF16_DIR = Path("/mnt/weka/mbedrosian/data/casf16/CASF16")
DEFAULT_LIGAND_DIR = DEFAULT_CASF16_DIR / "ligands"
DEFAULT_OUTPUT_CSV = DEFAULT_CASF16_DIR / "ligand_descriptors.csv"
DEFAULT_STATS_MD = DEFAULT_CASF16_DIR / "ligand_descriptor_stats.md"

CSV_FIELDS = (
    "ligand_id",
    "source_file",
    "canonical_smiles",
    "rotatable_bonds",
    "heavy_atoms",
    "rotatable_bonds_per_heavy_atom",
    "status",
    "error",
)


@dataclass(frozen=True)
class LigandDescriptor:
    ligand_id: str
    source_file: str
    canonical_smiles: str
    rotatable_bonds: int | None
    heavy_atoms: int | None
    rotatable_bonds_per_heavy_atom: float | None
    status: str
    error: str = ""


def require_rdkit() -> None:
    if RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RDKit is required to analyze MOL2 ligands. Activate the project chemistry "
            "environment first, for example: `conda activate 3dmolgen`."
        ) from RDKIT_IMPORT_ERROR


def descriptor_from_mol(mol, ligand_id: str, source_file: str) -> LigandDescriptor:
    """Compute descriptor fields for an already parsed RDKit molecule."""
    require_rdkit()
    mol_without_h = Chem.RemoveHs(mol)
    canonical_smiles = Chem.MolToSmiles(
        mol_without_h,
        canonical=True,
        isomericSmiles=True,
    )
    rotatable_bonds = int(Descriptors.NumRotatableBonds(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    density = rotatable_bonds / heavy_atoms if heavy_atoms else None
    return LigandDescriptor(
        ligand_id=ligand_id,
        source_file=source_file,
        canonical_smiles=canonical_smiles,
        rotatable_bonds=rotatable_bonds,
        heavy_atoms=heavy_atoms,
        rotatable_bonds_per_heavy_atom=density,
        status="ok",
    )


def descriptor_from_mol2(path: Path, input_dir: Path | None = None) -> LigandDescriptor:
    require_rdkit()
    ligand_id = path.stem
    source_file = str(path.relative_to(input_dir)) if input_dir is not None else str(path)
    try:
        mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
        if mol is None:
            raise ValueError("RDKit returned None")
        return descriptor_from_mol(mol, ligand_id=ligand_id, source_file=source_file)
    except Exception as exc:
        return LigandDescriptor(
            ligand_id=ligand_id,
            source_file=source_file,
            canonical_smiles="",
            rotatable_bonds=None,
            heavy_atoms=None,
            rotatable_bonds_per_heavy_atom=None,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def collect_descriptors(input_dir: Path) -> list[LigandDescriptor]:
    mol2_paths = sorted(input_dir.glob("*.mol2"))
    return [descriptor_from_mol2(path, input_dir=input_dir) for path in mol2_paths]


def percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "mean": None,
            "q3": None,
            "max": None,
        }
    numeric_values = [float(value) for value in values]
    return {
        "count": len(numeric_values),
        "min": min(numeric_values),
        "q1": percentile(numeric_values, 25),
        "median": statistics.median(numeric_values),
        "mean": statistics.fmean(numeric_values),
        "q3": percentile(numeric_values, 75),
        "max": max(numeric_values),
    }


def rotatable_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value <= 10:
        return "7-10"
    return ">10"


def heavy_atom_bucket(value: int) -> str:
    if value <= 20:
        return "<=20"
    if value <= 30:
        return "21-30"
    if value <= 40:
        return "31-40"
    if value <= 50:
        return "41-50"
    return ">50"


def bucket_counts(values: Iterable[int], bucket_func, labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(bucket_func(value) for value in values)
    return {label: counts.get(label, 0) for label in labels}


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = denominator_x * denominator_y
    if denominator == 0:
        return None
    return numerator / denominator


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def top_rows(
    rows: Sequence[LigandDescriptor],
    key,
    top_n: int,
    reverse: bool = True,
) -> list[LigandDescriptor]:
    valid_rows = [row for row in rows if row.status == "ok" and key(row) is not None]
    return sorted(valid_rows, key=key, reverse=reverse)[:top_n]


def build_stats(rows: Sequence[LigandDescriptor], top_n: int = 10) -> dict[str, object]:
    ok_rows = [row for row in rows if row.status == "ok"]
    failed_rows = [row for row in rows if row.status != "ok"]
    smiles_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in ok_rows:
        smiles_to_ids[row.canonical_smiles].append(row.ligand_id)
    duplicates = {
        smiles: ids
        for smiles, ids in sorted(smiles_to_ids.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(ids) > 1
    }

    rotatable_values = [row.rotatable_bonds for row in ok_rows if row.rotatable_bonds is not None]
    heavy_atom_values = [row.heavy_atoms for row in ok_rows if row.heavy_atoms is not None]
    paired_rows = [
        row for row in ok_rows if row.rotatable_bonds is not None and row.heavy_atoms is not None
    ]
    heavy_for_corr = [float(row.heavy_atoms) for row in paired_rows]
    rotatable_for_corr = [float(row.rotatable_bonds) for row in paired_rows]

    return {
        "total_files": len(rows),
        "parsed_molecules": len(ok_rows),
        "parse_failures": len(failed_rows),
        "unique_canonical_smiles": len(smiles_to_ids),
        "duplicate_structure_count": len(duplicates),
        "duplicates": duplicates,
        "rotatable_summary": numeric_summary(rotatable_values),
        "heavy_atom_summary": numeric_summary(heavy_atom_values),
        "rotatable_buckets": bucket_counts(rotatable_values, rotatable_bucket, ["0", "1-3", "4-6", "7-10", ">10"]),
        "heavy_atom_buckets": bucket_counts(heavy_atom_values, heavy_atom_bucket, ["<=20", "21-30", "31-40", "41-50", ">50"]),
        "pearson_heavy_vs_rotatable": pearson_correlation(heavy_for_corr, rotatable_for_corr),
        "spearman_heavy_vs_rotatable": spearman_correlation(heavy_for_corr, rotatable_for_corr),
        "most_flexible": top_rows(rows, lambda row: row.rotatable_bonds, top_n),
        "largest": top_rows(rows, lambda row: row.heavy_atoms, top_n),
        "smallest": top_rows(rows, lambda row: row.heavy_atoms, top_n, reverse=False),
        "highest_rotatable_density": top_rows(
            rows,
            lambda row: row.rotatable_bonds_per_heavy_atom,
            top_n,
        ),
        "failures": failed_rows[:top_n],
    }


def format_number(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(summary: dict[str, object]) -> str:
    lines = [
        "# CASF16 Ligand Descriptor Summary",
        "",
        "## Dataset",
        f"- Total MOL2 files: {summary['total_files']}",
        f"- Parsed molecules: {summary['parsed_molecules']}",
        f"- Parse failures: {summary['parse_failures']}",
        f"- Unique canonical SMILES: {summary['unique_canonical_smiles']}",
        f"- Duplicate structures: {summary['duplicate_structure_count']}",
        "",
    ]
    lines.extend(render_numeric_section("Rotatable Bonds", summary["rotatable_summary"], summary["rotatable_buckets"]))
    lines.extend(render_numeric_section("Heavy Atoms", summary["heavy_atom_summary"], summary["heavy_atom_buckets"]))
    lines.extend(
        [
            "## Size vs Flexibility",
            f"- Pearson correlation: {format_number(summary['pearson_heavy_vs_rotatable'], digits=3)}",
            f"- Spearman correlation: {format_number(summary['spearman_heavy_vs_rotatable'], digits=3)}",
            "",
        ]
    )
    lines.extend(render_ligand_list("Most Flexible Ligands", summary["most_flexible"]))
    lines.extend(render_ligand_list("Largest Ligands", summary["largest"]))
    lines.extend(render_ligand_list("Smallest Ligands", summary["smallest"]))
    lines.extend(render_ligand_list("Highest Rotatable-Bond Density", summary["highest_rotatable_density"]))
    lines.extend(render_duplicate_section(summary["duplicates"]))
    lines.extend(render_failure_section(summary["failures"]))
    return "\n".join(lines).rstrip() + "\n"


def render_numeric_section(title: str, stats: dict[str, object], buckets: dict[str, int]) -> list[str]:
    bucket_text = ", ".join(f"{label}: {count}" for label, count in buckets.items())
    return [
        f"## {title}",
        (
            f"- Count: {stats['count']}, min: {format_number(stats['min'])}, "
            f"Q1: {format_number(stats['q1'])}, median: {format_number(stats['median'])}, "
            f"mean: {format_number(stats['mean'])}, Q3: {format_number(stats['q3'])}, "
            f"max: {format_number(stats['max'])}"
        ),
        f"- Buckets: {bucket_text}",
        "",
    ]


def render_ligand_list(title: str, rows: Sequence[LigandDescriptor]) -> list[str]:
    lines = [f"## {title}"]
    if not rows:
        return lines + ["- None", ""]
    for row in rows:
        lines.append(
            "- "
            f"{row.ligand_id}: rotatable_bonds={format_number(row.rotatable_bonds)}, "
            f"heavy_atoms={format_number(row.heavy_atoms)}, "
            f"density={format_number(row.rotatable_bonds_per_heavy_atom, digits=3)}"
        )
    lines.append("")
    return lines


def render_duplicate_section(duplicates: dict[str, list[str]]) -> list[str]:
    lines = ["## Duplicate Canonical SMILES"]
    if not duplicates:
        return lines + ["- None", ""]
    for smiles, ligand_ids in list(duplicates.items())[:10]:
        lines.append(f"- {', '.join(ligand_ids)} share `{smiles}`")
    lines.append("")
    return lines


def render_failure_section(rows: Sequence[LigandDescriptor]) -> list[str]:
    lines = ["## Parse Failures"]
    if not rows:
        return lines + ["- None", ""]
    for row in rows:
        lines.append(f"- {row.ligand_id}: {row.error}")
    lines.append("")
    return lines


def write_csv(path: Path, rows: Sequence[LigandDescriptor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List canonical SMILES, rotatable bonds, heavy atoms, and stats for CASF16 ligands."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_LIGAND_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--stats-md", type=Path, default=DEFAULT_STATS_MD)
    parser.add_argument("--no-markdown", action="store_true", help="Do not write the Markdown stats file.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of highlighted ligands per section.")
    return parser


def run(args: argparse.Namespace) -> tuple[list[LigandDescriptor], str]:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    rows = collect_descriptors(args.input_dir)
    summary = build_stats(rows, top_n=args.top_n)
    summary_text = render_summary(summary)
    write_csv(args.output_csv, rows)
    if not args.no_markdown:
        write_text(args.stats_md, summary_text)
    return rows, summary_text


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        _, summary_text = run(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(summary_text)
    print(f"descriptor_csv={args.output_csv}")
    if not args.no_markdown:
        print(f"stats_markdown={args.stats_md}")


if __name__ == "__main__":
    main()
