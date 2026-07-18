#!/usr/bin/env python3
"""Exact CASF16-to-ChEMBL3D matches via regenerated CASF SMILES only."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from rdkit import Chem
    from rdkit.Geometry import Point3D
except ImportError as exc:  # pragma: no cover
    Chem = None
    Point3D = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None

try:
    from molgen3D.pharmacophore.chembl3d_conformer_loader import (
        prepare_torsion_ref_mol,
    )
    from molgen3D.pharmacophore.generate_casf_smiles_conformer_sets import get_rotatable_torsions
except ImportError:
    prepare_torsion_ref_mol = None
    get_rotatable_torsions = None


DEFAULT_CASF16_DIR = Path("/mnt/weka/mbedrosian/data/casf16/CASF16")
DEFAULT_LIGAND_DIR = DEFAULT_CASF16_DIR / "ligands"
DEFAULT_CHEMBL_INDEX = Path(
    "/mnt/weka/mbedrosian/data/chembl3d_index/chembl3d_topology_smiles_index.csv"
)
DEFAULT_DATA_DIR = Path("/mnt/weka/mbedrosian/data/casf16")
DEFAULT_OUTPUT_CSV = DEFAULT_DATA_DIR / "casf16_core_chembl3d_exact_intersection.csv"
DEFAULT_TOPOLOGY_ROOT = Path("/mnt/weka/mbedrosian/data/chembl3d/topologies")

GROUP_RE = re.compile(r"^\d{3}$")

OUTPUT_FIELDS = (
    "ligand_id",
    "source_file",
    "casf_explicit_isomeric_smiles",
    "casf_explicit_nonisomeric_smiles",
    "casf_heavy_isomeric_smiles",
    "casf_heavy_nonisomeric_smiles",
    "chembl3d_group",
    "chembl3d_mol_id",
    "chembl3d_isomeric_smiles",
    "conformer_count",
)


@dataclass(frozen=True)
class ChemblIndexRow:
    group: str
    mol_id: str
    isomeric_canonical_smiles: str
    conformer_count: int


@dataclass(frozen=True)
class CasfRegenSmiles:
    ligand_id: str
    source_file: str
    status: str
    explicit_isomeric_smiles: str = ""
    explicit_nonisomeric_smiles: str = ""
    heavy_isomeric_smiles: str = ""
    heavy_nonisomeric_smiles: str = ""
    error: str = ""


def require_rdkit() -> None:
    if RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RDKit is required. Run with a chemistry environment, "
            "for example `/home/mbedrosian/.conda/envs/chembl3d/bin/python`."
        ) from RDKIT_IMPORT_ERROR


def casf_regenerated_smiles(mol2_path: Path) -> CasfRegenSmiles:
    """RemoveHs -> AddHs (zero H coords) -> explicit and heavy canonical SMILES."""
    require_rdkit()
    ligand_id = mol2_path.stem
    source_file = mol2_path.name

    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                return CasfRegenSmiles(
                    ligand_id=ligand_id,
                    source_file=source_file,
                    status="mol2_parse_failed",
                    error=f"sanitize_failed: {exc}",
                )
    if mol is None:
        return CasfRegenSmiles(
            ligand_id=ligand_id,
            source_file=source_file,
            status="mol2_parse_failed",
            error="MolFromMol2File returned None",
        )

    mol = Chem.RemoveHs(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers():
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                conf.SetAtomPosition(atom.GetIdx(), Point3D(0.0, 0.0, 0.0))

    explicit_isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    explicit_nonisomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    heavy = Chem.RemoveHs(mol)
    heavy_isomeric = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    heavy_nonisomeric = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=False)

    return CasfRegenSmiles(
        ligand_id=ligand_id,
        source_file=source_file,
        status="ok",
        explicit_isomeric_smiles=explicit_isomeric,
        explicit_nonisomeric_smiles=explicit_nonisomeric,
        heavy_isomeric_smiles=heavy_isomeric,
        heavy_nonisomeric_smiles=heavy_nonisomeric,
    )


def discover_ligand_mol2_files(ligand_dir: Path) -> list[Path]:
    return sorted(ligand_dir.glob("*.mol2"))


def load_casf_regenerated(ligand_dir: Path) -> tuple[dict[str, CasfRegenSmiles], int, int]:
    by_ligand_id: dict[str, CasfRegenSmiles] = {}
    processed = 0
    failed = 0
    for mol2_path in discover_ligand_mol2_files(ligand_dir):
        processed += 1
        regen = casf_regenerated_smiles(mol2_path)
        by_ligand_id[regen.ligand_id] = regen
        if regen.status != "ok":
            failed += 1
    return by_ligand_id, processed, failed


def parse_index_row(fields: list[str]) -> ChemblIndexRow | None:
    if len(fields) == 7:
        group, mol_id, _original, isomeric, _heavy, _rot, raw_count = fields
        if not GROUP_RE.match(group):
            return None
        raw_count = raw_count.strip()
        return ChemblIndexRow(
            group=group,
            mol_id=mol_id,
            isomeric_canonical_smiles=isomeric,
            conformer_count=int(raw_count) if raw_count.isdigit() else 0,
        )

    if len(fields) >= 9:
        group, mol_id, isomeric, raw_count = fields[0], fields[2], fields[5], fields[8]
        if not GROUP_RE.match(group):
            return None
        raw_count = raw_count.strip()
        return ChemblIndexRow(
            group=group,
            mol_id=mol_id,
            isomeric_canonical_smiles=isomeric,
            conformer_count=int(raw_count) if raw_count.isdigit() else 0,
        )

    return None


def lookup_chembl_hits(
    path: Path,
    target_smiles: set[str],
) -> dict[str, list[ChemblIndexRow]]:
    hits: dict[str, list[ChemblIndexRow]] = {}
    dedupe: dict[tuple[str, str, str, str], ChemblIndexRow] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return hits
        for fields in reader:
            row = parse_index_row(fields)
            if row is None or row.isomeric_canonical_smiles not in target_smiles:
                continue
            key = (row.isomeric_canonical_smiles, row.group, row.mol_id)
            existing = dedupe.get(key)
            if existing is None or row.conformer_count > existing.conformer_count:
                dedupe[key] = row

    for row in dedupe.values():
        hits.setdefault(row.isomeric_canonical_smiles, []).append(row)
    for rows in hits.values():
        rows.sort(key=lambda item: (item.group, item.mol_id))
    return hits


def build_topology_index(
    chembl_hits: dict[str, list[ChemblIndexRow]],
    topology_root: Path,
) -> dict[tuple[str, str], Chem.Mol]:
    needed_by_group: dict[str, set[str]] = {}
    for hits in chembl_hits.values():
        for hit in hits:
            needed_by_group.setdefault(hit.group, set()).add(hit.mol_id)
    groups = sorted(needed_by_group)
    print(f"indexing_topology_groups={len(groups)}", flush=True)
    topology_index: dict[tuple[str, str], Chem.Mol] = {}
    for idx, group in enumerate(groups, start=1):
        needed = needed_by_group[group]
        found: set[str] = set()
        sdf_path = topology_root / f"{int(group):03d}.sdf"
        if sdf_path.is_file():
            for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False):
                if mol is None:
                    continue
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                prop_mol_id = mol.GetProp("mol_id") if mol.HasProp("mol_id") else ""
                mol_id = prop_mol_id or name
                if not mol_id or mol_id not in needed or mol_id in found:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                topology_index[(group, mol_id)] = Chem.Mol(mol)
                found.add(mol_id)
                if found == needed:
                    break
        if idx % 10 == 0 or idx == len(groups):
            print(f"topology_groups_indexed={idx}/{len(groups)}", flush=True)
    return topology_index


def topology_mol_from_index(
    hit: ChemblIndexRow,
    topology_index: dict[tuple[str, str], Chem.Mol],
) -> Chem.Mol | None:
    return topology_index.get((hit.group, hit.mol_id))


def pick_chembl_hit(
    hits: tuple[ChemblIndexRow, ...],
    topology_index: dict[tuple[str, str], Chem.Mol] | None,
    *,
    require_topology_sdf: bool = True,
) -> ChemblIndexRow | None:
    if not hits:
        return None
    if topology_index is not None:
        for hit in hits:
            if topology_mol_from_index(hit, topology_index) is not None:
                return hit
        if require_topology_sdf:
            return None
    if require_topology_sdf:
        return None
    return hits[0]


def chembl_hit_has_rotatable_bonds(
    hit: ChemblIndexRow,
    topology_index: dict[tuple[str, str], Chem.Mol],
) -> bool:
    if prepare_torsion_ref_mol is None or get_rotatable_torsions is None:
        return True
    mol = topology_mol_from_index(hit, topology_index)
    ref = prepare_torsion_ref_mol(mol)
    return ref is not None and bool(get_rotatable_torsions(ref))


def resolve_eligible_chembl_hit(
    hits: tuple[ChemblIndexRow, ...],
    topology_index: dict[tuple[str, str], Chem.Mol] | None,
) -> ChemblIndexRow | None | str:
    """Return a Chembl hit, None (missing SDF), or 'no_rotatable_bonds'."""
    if not hits:
        return None
    hit = pick_chembl_hit(hits, topology_index, require_topology_sdf=True)
    if hit is None:
        return None
    if topology_index is not None and not chembl_hit_has_rotatable_bonds(hit, topology_index):
        return "no_rotatable_bonds"
    return hit


def build_exact_match_rows(
    casf_by_ligand: dict[str, CasfRegenSmiles],
    chembl_hits: dict[str, list[ChemblIndexRow]],
    topology_index: dict[tuple[str, str], Chem.Mol] | None = None,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    excluded: dict[str, list[str]] = {
        "no_smiles_match": [],
        "missing_topology_sdf": [],
        "no_rotatable_bonds": [],
    }
    eligible_by_smiles: dict[str, ChemblIndexRow | None | str] = {}
    hit_smiles = list(chembl_hits.items())
    for idx, (smiles, hits) in enumerate(hit_smiles, start=1):
        eligible_by_smiles[smiles] = resolve_eligible_chembl_hit(tuple(hits), topology_index)
        if idx % 200 == 0 or idx == len(hit_smiles):
            print(f"eligible_smiles_resolved={idx}/{len(hit_smiles)}", flush=True)

    for regen in casf_by_ligand.values():
        if regen.status != "ok":
            continue
        hits = chembl_hits.get(regen.heavy_isomeric_smiles)
        if not hits:
            excluded["no_smiles_match"].append(regen.ligand_id)
            continue
        eligible = eligible_by_smiles[regen.heavy_isomeric_smiles]
        if eligible == "no_rotatable_bonds":
            excluded["no_rotatable_bonds"].append(regen.ligand_id)
            continue
        if eligible is None:
            excluded["missing_topology_sdf"].append(regen.ligand_id)
            continue
        hit = eligible
        rows.append(
            {
                "ligand_id": regen.ligand_id,
                "source_file": regen.source_file,
                "casf_explicit_isomeric_smiles": regen.explicit_isomeric_smiles,
                "casf_explicit_nonisomeric_smiles": regen.explicit_nonisomeric_smiles,
                "casf_heavy_isomeric_smiles": regen.heavy_isomeric_smiles,
                "casf_heavy_nonisomeric_smiles": regen.heavy_nonisomeric_smiles,
                "chembl3d_group": hit.group,
                "chembl3d_mol_id": hit.mol_id,
                "chembl3d_isomeric_smiles": hit.isomeric_canonical_smiles,
                "conformer_count": hit.conformer_count,
            }
        )
    rows.sort(key=lambda row: (str(row["ligand_id"]), str(row["chembl3d_mol_id"])))
    return rows, excluded


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate CASF16 SMILES (RemoveHs -> AddHs with zero H coords) and "
            "write exact matches against the ChEMBL3D topology index. Output rows "
            "require a ChEMBL3D topology SDF entry and at least one rotatable bond."
        )
    )
    parser.add_argument("--ligand-dir", type=Path, default=DEFAULT_LIGAND_DIR)
    parser.add_argument("--chembl-index", type=Path, default=DEFAULT_CHEMBL_INDEX)
    parser.add_argument("--topology-root", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.ligand_dir.is_dir():
        raise FileNotFoundError(f"CASF ligand directory not found: {args.ligand_dir}")
    if not args.chembl_index.is_file():
        raise FileNotFoundError(f"ChEMBL3D index not found: {args.chembl_index}")

    casf_by_ligand, processed, failed = load_casf_regenerated(args.ligand_dir)
    target_smiles = {
        regen.heavy_isomeric_smiles
        for regen in casf_by_ligand.values()
        if regen.status == "ok" and regen.heavy_isomeric_smiles
    }
    print(f"casf_ligands_processed={processed}", flush=True)
    print(f"casf_regen_failures={failed}", flush=True)
    print(f"target_heavy_isomeric_smiles={len(target_smiles)}", flush=True)

    print(f"Scanning ChEMBL3D index: {args.chembl_index}", flush=True)
    chembl_hits = lookup_chembl_hits(args.chembl_index, target_smiles)
    print(f"chembl_hit_smiles={len(chembl_hits)}", flush=True)

    topology_index = build_topology_index(chembl_hits, args.topology_root)
    print(f"topology_index_entries={len(topology_index)}", flush=True)

    rows, excluded = build_exact_match_rows(casf_by_ligand, chembl_hits, topology_index)
    matched_ligands = {row["ligand_id"] for row in rows}
    write_csv(args.output_csv, rows)

    print(f"exact_matched_ligands={len(matched_ligands)}", flush=True)
    print(f"exact_match_rows={len(rows)}", flush=True)
    for reason, ligand_ids in excluded.items():
        print(f"excluded_{reason}={len(ligand_ids)}", flush=True)
        if ligand_ids:
            print(f"excluded_{reason}_ligands={','.join(sorted(ligand_ids))}", flush=True)
    print(f"output_csv={args.output_csv.resolve()}", flush=True)
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
