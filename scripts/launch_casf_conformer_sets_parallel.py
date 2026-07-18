#!/usr/bin/env python3
"""Run generate_casf_smiles_conformer_sets.py in parallel over molecule offsets."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_GENERATOR = REPO_ROOT / "src/molgen3D/pharmacophore/generate_casf_smiles_conformer_sets.py"
from molgen3D.pharmacophore.conformer_sets_layout import (
    DEFAULT_CHEMBL_DATASET_ROOT,
    DEFAULT_CHEMBL_MAP_CSV,
    DEFAULT_CORE_LIGAND_DIR,
    DEFAULT_CORE_PHARMA_ROOT,
)

DEFAULT_LIGAND_DIR = DEFAULT_CORE_LIGAND_DIR
DEFAULT_TOPOLOGY_ROOT = DEFAULT_CHEMBL_DATASET_ROOT / "topologies"
DEFAULT_OUTPUT_DIR = DEFAULT_CORE_PHARMA_ROOT
DEFAULT_LOG_DIR = REPO_ROOT / "outputs/conformer_sets_core_chembl3d_exact_intersection_workers"


def generation_dir(output_dir: Path) -> Path:
    return output_dir if output_dir.name == "generation" else output_dir / "generation"


def load_ligand_ids(chembl_map_csv: Path) -> list[str]:
    with chembl_map_csv.open("r", encoding="utf-8", newline="") as handle:
        return [row["ligand_id"].strip() for row in csv.DictReader(handle)]


def discover_offsets(
    chembl_map_csv: Path,
    output_dir: Path,
    skip_existing: bool,
) -> list[int]:
    ligand_ids = load_ligand_ids(chembl_map_csv)
    if not ligand_ids:
        raise FileNotFoundError(f"No rows found in {chembl_map_csv}")

    parts_dir = generation_dir(output_dir) / "manifest_parts"
    offsets: list[int] = []
    for offset, ligand_id in enumerate(ligand_ids):
        if skip_existing and (parts_dir / f"{ligand_id}.tsv").is_file():
            continue
        offsets.append(offset)
    return offsets


def run_one(
    offset: int,
    generator: Path,
    chembl_map_csv: Path,
    ligand_dir: Path,
    topology_root: Path,
    output_dir: Path,
    log_dir: Path,
    num_threads: int,
    minimize_workers: int,
    extra_args: list[str],
) -> tuple[int, int, str]:
    log_path = log_dir / f"offset_{offset:03d}.log"
    cmd = [
        sys.executable,
        str(generator),
        "--chembl_map_csv",
        str(chembl_map_csv),
        "--ligand_dir",
        str(ligand_dir),
        "--chembl3d_topology_root",
        str(topology_root),
        "--output_dir",
        str(output_dir),
        "--molecule_offset",
        str(offset),
        "--limit_molecules",
        "1",
        "--num_threads",
        str(num_threads),
        "--minimize_workers",
        str(minimize_workers),
        *extra_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
    return offset, proc.returncode, status


def merge_manifest(generator: Path, output_dir: Path, chembl_map_csv: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--output_dir",
            str(output_dir),
            "--chembl_map_csv",
            str(chembl_map_csv),
            "--merge_manifest",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel launcher for CASF conformer generation.")
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--chembl_map_csv", type=Path, default=DEFAULT_CHEMBL_MAP_CSV)
    parser.add_argument("--ligand_dir", type=Path, default=DEFAULT_LIGAND_DIR)
    parser.add_argument("--chembl3d_topology_root", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Molecule workers; 0 chooses total_cpu_budget // cpus_per_molecule.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="RDKit embedding threads per molecule process.",
    )
    parser.add_argument("--minimize_workers", type=int, default=None)
    parser.add_argument(
        "--cpus_per_molecule",
        type=int,
        default=4,
        help="CPU budget per molecule process when --workers or --minimize_workers are auto.",
    )
    parser.add_argument(
        "--total_cpu_budget",
        type=int,
        default=0,
        help="Total CPU budget for auto sizing; 0 uses os.cpu_count().",
    )
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--no_merge", action="store_true")
    parser.add_argument(
        "generator_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to generate_casf_smiles_conformer_sets.py",
    )
    return parser


def resolve_parallelism(args: argparse.Namespace, pending_count: int) -> None:
    if args.num_threads <= 0:
        raise ValueError("--num_threads must be positive")
    if args.cpus_per_molecule <= 0:
        raise ValueError("--cpus_per_molecule must be positive")

    cpu_budget = args.total_cpu_budget if args.total_cpu_budget > 0 else (os.cpu_count() or 1)
    if args.workers <= 0:
        args.workers = max(1, cpu_budget // args.cpus_per_molecule)
    args.workers = max(1, min(args.workers, max(1, pending_count)))

    if args.minimize_workers is None or args.minimize_workers <= 0:
        args.minimize_workers = max(1, min(args.cpus_per_molecule, max(1, cpu_budget // args.workers)))


def main() -> None:
    args = build_arg_parser().parse_args()
    extra_args = list(args.generator_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if not args.generator.is_file():
        raise FileNotFoundError(f"Generator not found: {args.generator}")
    if not args.chembl_map_csv.is_file():
        raise FileNotFoundError(f"Intersection CSV not found: {args.chembl_map_csv}")
    if not args.ligand_dir.is_dir():
        raise FileNotFoundError(f"Ligand directory not found: {args.ligand_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = discover_offsets(args.chembl_map_csv, args.output_dir, args.skip_existing)
    total = len(load_ligand_ids(args.chembl_map_csv))
    resolve_parallelism(args, len(offsets) or total)

    print(f"total_molecules={total}", flush=True)
    print(f"pending_offsets={len(offsets)}", flush=True)
    print(f"workers={args.workers}", flush=True)
    print(f"num_threads={args.num_threads}", flush=True)
    print(f"minimize_workers={args.minimize_workers}", flush=True)
    print(f"cpus_per_molecule={args.cpus_per_molecule}", flush=True)
    print(f"output_dir={args.output_dir.resolve()}", flush=True)

    if not offsets:
        print("Nothing to do.", flush=True)
    else:
        failures: list[tuple[int, int]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_one,
                    offset,
                    args.generator,
                    args.chembl_map_csv,
                    args.ligand_dir,
                    args.chembl3d_topology_root,
                    args.output_dir,
                    args.log_dir,
                    args.num_threads,
                    args.minimize_workers,
                    extra_args,
                )
                for offset in offsets
            ]
            done = 0
            for future in as_completed(futures):
                offset, code, status = future.result()
                done += 1
                if code != 0:
                    failures.append((offset, code))
                print(f"[{done}/{len(offsets)}] offset={offset} {status}", flush=True)

        if failures:
            failed = ", ".join(f"{offset}({code})" for offset, code in failures)
            raise SystemExit(f"{len(failures)} worker(s) failed: {failed}")

    if not args.no_merge:
        print("Merging manifest parts...", flush=True)
        merge_manifest(args.generator, args.output_dir, args.chembl_map_csv)
        manifest = generation_dir(args.output_dir) / "manifest.tsv"
        print(f"manifest={manifest.resolve()}", flush=True)


if __name__ == "__main__":
    main()
