"""Load ChEMBL3D conformer coordinates from zarr + topology SDF templates."""

from __future__ import annotations

from pathlib import Path

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    Chem = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None

try:
    import numpy as np
    import zarr
    from rdkit.Geometry import Point3D
except ImportError as exc:  # pragma: no cover
    np = None
    zarr = None
    Point3D = None
    ZARR_IMPORT_ERROR = exc
else:
    ZARR_IMPORT_ERROR = None


def require_dependencies() -> None:
    if ZARR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ChEMBL3D zarr loading requires numpy and zarr. Run with the ChEMBL3D "
            "environment, for example `/home/mbedrosian/.conda/envs/chembl3d/bin/python`."
        ) from ZARR_IMPORT_ERROR


def decode_mol_id(value: object) -> str:
    if isinstance(value, bytes):
        raw = value
    elif hasattr(value, "tobytes"):
        raw = value.tobytes()
    else:
        raw = bytes(value)
    return raw.decode("utf-8", errors="replace").rstrip(" \x00")


def find_mol_id_indices(mol_id_array, mol_id: str) -> list[int]:
    """Return every exact row index for a ChEMBL3D molecule id."""
    require_dependencies()
    encoded = mol_id.encode("utf-8")
    values = np.asarray(mol_id_array[:])
    matches = np.where(values == encoded)[0]
    return [int(index) for index in matches.tolist()]


def load_topology_mol(group: str, mol_id: str, topology_root: Path) -> Chem.Mol | None:
    if Chem is None:
        raise RuntimeError("RDKit is required to load ChEMBL3D topology SDF files.") from RDKIT_IMPORT_ERROR
    sdf_path = topology_root / f"{int(group):03d}.sdf"
    if not sdf_path.exists():
        return None
    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        prop_mol_id = mol.GetProp("mol_id") if mol.HasProp("mol_id") else ""
        if prop_mol_id == mol_id or name == mol_id:
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                raise ValueError(f"Failed to sanitize topology molecule {mol_id} in {sdf_path}") from exc
            return Chem.Mol(mol)
    return None


def prepare_torsion_ref_mol(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None or mol.GetNumConformers() == 0:
        return None
    prepared = Chem.Mol(mol)
    if not any(atom.GetAtomicNum() == 1 for atom in prepared.GetAtoms()):
        prepared = Chem.AddHs(prepared, addCoords=True)
    return prepared


def load_torsion_ref_from_chembl3d(
    group: str,
    mol_id: str,
    topology_root: Path,
) -> Chem.Mol | None:
    """Load torsion perturbation seed from ChEMBL3D topology SDF coordinates."""
    return prepare_torsion_ref_mol(load_topology_mol(group, mol_id, topology_root))


def load_torsion_ref_from_mol2(mol2_path: Path) -> Chem.Mol | None:
    """Load torsion seed from a CASF mol2 when ChEMBL3D topology SDF has no entry."""
    if Chem is None:
        raise RuntimeError("RDKit is required to load mol2 torsion references.") from RDKIT_IMPORT_ERROR
    if not mol2_path.is_file():
        return None

    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return prepare_torsion_ref_mol(mol)


def load_torsion_ref(
    group: str,
    mol_id: str,
    topology_root: Path,
    mol2_path: Path | None = None,
) -> tuple[Chem.Mol | None, str]:
    """Prefer ChEMBL3D topology SDF; fall back to CASF mol2 coordinates."""
    ref = load_torsion_ref_from_chembl3d(group, mol_id, topology_root)
    if ref is not None:
        return ref, "chembl3d_topology_sdf"
    if mol2_path is not None:
        ref = load_torsion_ref_from_mol2(mol2_path)
        if ref is not None:
            return ref, "casf_mol2_fallback"
    return None, "unavailable"


def _set_coords_from_row(template: Chem.Mol, coord_row) -> Chem.Mol:
    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    n_atoms = mol.GetNumAtoms()
    if len(coord_row) != n_atoms:
        raise ValueError(
            f"Coordinate row length {len(coord_row)} != topology atom count {n_atoms}"
        )
    conf = Chem.Conformer(n_atoms)
    for atom_idx in range(n_atoms):
        point = Point3D(
            float(coord_row[atom_idx][0]),
            float(coord_row[atom_idx][1]),
            float(coord_row[atom_idx][2]),
        )
        conf.SetAtomPosition(atom_idx, point)
    mol.AddConformer(conf, assignId=True)
    return mol


def _topology_atomic_numbers(template: Chem.Mol) -> list[int]:
    return [atom.GetAtomicNum() for atom in template.GetAtoms()]


def _validate_atomic_numbers(numbers_row, expected: list[int], row_idx: int, mol_id: str) -> None:
    observed = [int(value) for value in np.asarray(numbers_row).tolist()]
    if observed != expected:
        raise ValueError(
            f"Atomic numbers row {row_idx} for {mol_id} does not match topology: "
            f"{observed} != {expected}"
        )


def load_chembl3d_conformers(
    group: str,
    mol_id: str,
    topology_root: Path,
    zarr_root: Path,
    limit: int | None = None,
) -> list[Chem.Mol]:
    require_dependencies()
    template = load_topology_mol(group, mol_id, topology_root)
    if template is None:
        sdf_path = topology_root / f"{int(group):03d}.sdf"
        raise FileNotFoundError(
            f"ChEMBL3D topology molecule {mol_id} not found in {sdf_path}"
        )

    group_path = zarr_root / f"{int(group):03d}"
    mol_id_path = group_path / "mol_id"
    coord_path = group_path / "coord"
    numbers_path = group_path / "numbers"
    if not mol_id_path.exists() or not coord_path.exists() or not numbers_path.exists():
        missing = [
            str(path)
            for path in (mol_id_path, coord_path, numbers_path)
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing ChEMBL3D zarr array(s) for group {int(group):03d}: {missing}")

    mol_id_array = zarr.open_array(str(mol_id_path), mode="r")
    coord_array = zarr.open_array(str(coord_path), mode="r")
    numbers_array = zarr.open_array(str(numbers_path), mode="r")
    row_indices = find_mol_id_indices(mol_id_array, mol_id)
    if not row_indices:
        return []

    if limit is not None:
        row_indices = row_indices[:limit]

    mols: list[Chem.Mol] = []
    expected_numbers = _topology_atomic_numbers(template)
    for row_idx in row_indices:
        _validate_atomic_numbers(numbers_array[row_idx], expected_numbers, row_idx, mol_id)
        mol = _set_coords_from_row(
            template,
            coord_array[row_idx],
        )
        mol.SetProp("_Name", mol_id)
        mol.SetProp("chembl3d_group", f"{int(group):03d}")
        mol.SetProp("chembl3d_mol_id", mol_id)
        mols.append(mol)
    return mols
