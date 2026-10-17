#!/usr/bin/env python3
import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional
import numpy as np
import torch
from rdkit import Chem

from train_on_nabladft.tokenize_utils import get_embedded_smiles


@dataclass(frozen=True)
class EmbeddedRow:
    embedded_smiles: str
    source: str
    conformer_id: Optional[int]
    energy_ev: Optional[float]


def fail_if_missing(path: str, name: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {name}: {p}")


def load_nabla_embedded(nabla_dir: str, limit: Optional[int]) -> Generator[EmbeddedRow, None, None]:
    fail_if_missing(nabla_dir, "nabla_dir")
    dir_path = Path(nabla_dir)
    train_file = dir_path / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"Expected NABLA train file not found: {train_file}")
    yielded = 0
    with train_file.open("r") as f:
        for line in f:
            if limit is not None and yielded >= limit:
                return
            emb = line.strip()
            if not emb:
                continue
            yield EmbeddedRow(embedded_smiles=emb, source="nabla", conformer_id=None, energy_ev=None)
            yielded += 1


def load_geom_embedded(geom_root: str, limit: Optional[int]) -> Generator[EmbeddedRow, None, None]:
    root = Path(geom_root)
    drugs = root / "drugs"
    if not drugs.exists():
        raise FileNotFoundError(f"Expected GEOM 'drugs' subset not found under {geom_root}")
    files = sorted([p for p in drugs.glob("*.pickle") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No GEOM pickle files found under {geom_root}")
    yielded = 0
    for pkl in files:
        with pkl.open("rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(f"Error loading GEOM pickle file {pkl}: {e}")
                continue
        conformers = data.get("conformers", [])
        if not conformers:
            continue
        for conf_idx, cdata in enumerate(conformers):
            if limit is not None and yielded >= limit:
                return
            mol = cdata.get("rd_mol")
            if mol is None or not isinstance(mol, Chem.Mol):
                continue
            try:
                emb = get_embedded_smiles(mol)
            except Exception as e:
                print(f"Error embedding GEOM molecule {pkl}: {e}")
                continue
            raw_energy = None
            if isinstance(cdata, dict):
                raw_energy = cdata.get("totalenergy")
            e_ev = (raw_energy * 27.2114) if isinstance(raw_energy, (int, float)) else None
            yield EmbeddedRow(embedded_smiles=emb, source="geom", conformer_id=conf_idx, energy_ev=e_ev)
            yielded += 1

def load_chembl3d_from_h(chembl_root: str, limit: Optional[int]) -> Generator[EmbeddedRow, None, None]:
    root = Path(chembl_root)
    smiles_path = root / "train_smiles.pickle"
    coords_path_npy = root / "train_h.npy"
    coords_path_pt = root / "train_h.pt"
    atom_types_path = root / "train_atom_types_h.npy"
    n_atoms_path = root / "train_n_h.pickle"

    if not smiles_path.exists() or not atom_types_path.exists():
        return

    if not coords_path_npy.exists() and not coords_path_pt.exists():
        return

    with smiles_path.open("rb") as f:
        smiles_list = pickle.load(f)

    atom_types_obj = np.load(atom_types_path, allow_pickle=True, mmap_mode='r')

    if coords_path_npy.exists():
        coords_arr = np.load(str(coords_path_npy), allow_pickle=True, mmap_mode='r')
    else:
        coords_obj = torch.load(str(coords_path_pt), map_location="cpu", weights_only=False)
        if isinstance(coords_obj, torch.Tensor):
            coords_arr = coords_obj.detach().cpu().numpy()
        elif isinstance(coords_obj, (list, tuple)):
            coords_arr = np.array([c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c) for c in coords_obj], dtype=object)
        else:
            coords_arr = None

    cumsum_n: Optional[np.ndarray] = None
    if n_list is not None and coords_arr is not None and coords_arr.ndim == 2:
        cumsum_n = np.cumsum([0] + n_list[:-1])

    def get_coords_i(idx: int) -> Optional[np.ndarray]:
        if coords_arr is None or idx >= len(smiles_list):
            return None
        if coords_arr.dtype == object:
            return coords_arr[idx]
        if coords_arr.ndim == 3 and idx < coords_arr.shape[0]:
            if n_list is not None and idx < len(n_list):
                n_i = int(n_list[idx])
                return coords_arr[idx, :n_i, :]
            return coords_arr[idx]
        if coords_arr.ndim == 2 and cumsum_n is not None and idx < len(n_list):
            start = int(cumsum_n[idx])
            end = start + int(n_list[idx])
            return coords_arr[start:end]
        return None

    def get_atom_types_i(idx: int) -> Optional[np.ndarray]:
        if atom_types_obj.dtype == object:
            return atom_types_obj[idx]
        if atom_types_obj.ndim == 2:
            if n_list is not None and idx < len(n_list):
                n_i = int(n_list[idx])
                return atom_types_obj[idx, :n_i]
            return atom_types_obj[idx]
        return None

    yielded = 0
    total = len(smiles_list)
    for i in range(total):
        if limit is not None and yielded >= limit:
            return
        smiles = smiles_list[i]
        coords_i = get_coords_i(i)
        atom_types_i = get_atom_types_i(i)
        if coords_i is None or atom_types_i is None:
            continue
        mask_heavy = atom_types_i != 1
        if mask_heavy.shape[0] != coords_i.shape[0]:
            continue
        heavy_coords = coords_i[mask_heavy]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_heavy = mol.GetNumAtoms()
        if heavy_coords.shape[0] != n_heavy:
            continue
        conf = Chem.Conformer(n_heavy)
        for atom_idx in range(n_heavy):
            x, y, z = map(float, heavy_coords[atom_idx])
            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        try:
            emb = get_embedded_smiles(mol)
        except Exception:
            continue
        yield EmbeddedRow(embedded_smiles=emb, source="chembl3d", conformer_id=0, energy_ev=None)
        yielded += 1


def load_chembl3d_embedded(chembl_root: str, limit: Optional[int]) -> Generator[EmbeddedRow, None, None]:
    fail_if_missing(chembl_root, "chembl3d_root")
    yield from load_chembl3d_from_h(chembl_root=chembl_root, limit=limit)


def write_csv(rows: Iterable[EmbeddedRow], output_csv: str) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embedded_smiles", "source", "conformer_id", "energy_ev"])
        for r in rows:
            writer.writerow([r.embedded_smiles, r.source, r.conformer_id if r.conformer_id is not None else "", r.energy_ev if r.energy_ev is not None else ""])


def collect_rows(nabla_dir: str, geom_root: str, chembl_root: str, per_source_limit: Optional[int]) -> Generator[EmbeddedRow, None, None]:
    yield from load_nabla_embedded(nabla_dir=nabla_dir, limit=per_source_limit)
    yield from load_geom_embedded(geom_root=geom_root, limit=per_source_limit)
    yield from load_chembl3d_embedded(chembl_root=chembl_root, limit=per_source_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate embedded SMILES from NABLA, GEOM, and CHEMBL3D into a CSV; energies only when present (GEOM, some CHEMBL3D pickles)")
    parser.add_argument("--nabla_dir", type=str, default="/nfs/h100/raid/chem/nablaDFT/emb_smiles")
    parser.add_argument("--geom_root", type=str, default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder")
    parser.add_argument("--chembl3d_root", type=str, default="/nfs/h100/raid/chem/chembl3d")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--per_source_limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_csv(
        rows=collect_rows(
            nabla_dir=args.nabla_dir,
            geom_root=args.geom_root,
            chembl_root=args.chembl3d_root,
            per_source_limit=args.per_source_limit,
        ),
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
