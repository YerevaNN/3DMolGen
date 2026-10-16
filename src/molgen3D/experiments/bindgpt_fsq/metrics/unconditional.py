"""Evaluation 1: the BindGPT/MolDiff-style unconditional table.

Molecule-level properties, local-geometry distributions and substructure
frequencies come from the pinned MolDiff evaluator vendored under
``bindgpt_geom_fsq/vendor/moldiff`` (see its ``PROVENANCE.json``: upstream
commit, per-file hashes, and the two import/compat rewrites that were needed).

The distance convention and every binning choice are ported from MolDiff's
released ``analyze_generated.ipynb`` into :func:`compare_with_ref`, unchanged:
the range is the pooled values trimmed at the sixth smallest and sixth largest
observation, bins are fixed-width (0.02 A for lengths, 5 degrees for angles and
dihedrals) or centred integers for discrete quantities, histograms are taken
with ``density=True``, floored at ``1e-10`` and renormalized, and the reported
number is SciPy's ``jensenshannon`` -- the Jensen-Shannon **distance**, not the
squared divergence.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from scipy.spatial.distance import jensenshannon

from molgen3D.experiments.bindgpt_fsq import config as experiment_config

RDLogger.DisableLog("rdApp.*")

#: Bin widths, from MolDiff's notebook.
WIDTH_LENGTH = 0.02
WIDTH_ANGLE = 5.0
WIDTH_DIHEDRAL = 5.0

RING_SIZE_KEYS = tuple(f"cnt_ring{size}" for size in range(3, 10))
BOND_TYPE_KEYS = ("cnt_bond1", "cnt_bond2", "cnt_bond3", "cnt_bond4")


def _vendor_on_path() -> None:
    vendor = experiment_config.EXPERIMENT_ROOT / "vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def moldiff():
    """Import the vendored MolDiff evaluator."""
    _vendor_on_path()
    from moldiff import evaluation  # noqa: PLC0415 - deliberately late

    return evaluation


def get_jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """MolDiff's distance: SciPy Jensen-Shannon **distance**."""
    return float(jensenshannon(np.asarray(p, dtype=float), np.asarray(q, dtype=float)))


def compare_with_ref(
    value_list: Sequence[Sequence[float]],
    width: Optional[float] = None,
    num_bins: int = 50,
    discrete: bool = False,
    idx_ref: int = 0,
) -> Tuple[List[float], np.ndarray, List[np.ndarray]]:
    """Ported verbatim in behaviour from MolDiff's ``analyze_generated.ipynb``."""
    pooled = np.concatenate([np.asarray(values, dtype=float) for values in value_list])
    pooled = pooled[~np.isnan(pooled)]
    if pooled.size < 11:
        return [float("nan")] * len(value_list), np.asarray([]), []

    ordered = np.sort(pooled)
    max_value = ordered[-5]
    min_value = ordered[5]
    if max_value <= min_value:
        max_value = min_value + (width or 1.0)

    if not discrete:
        bins = (
            np.arange(min_value, max_value + width, width)
            if width is not None
            else np.linspace(min_value, max_value, num_bins)
        )
    else:
        bins = np.arange(min_value, max_value + 1.5) - 0.5
    if bins.size < 2:
        return [float("nan")] * len(value_list), bins, []

    histograms: List[np.ndarray] = []
    for values in value_list:
        hist, _ = np.histogram(np.asarray(values, dtype=float), bins=bins, density=True)
        hist = hist + 1e-10
        histograms.append(hist / hist.sum())

    return [get_jsd(histograms[idx_ref], hist) for hist in histograms], bins, histograms


def _frequency_jsd(
    reference: Dict[str, Sequence[float]], generated: Dict[str, Sequence[float]]
) -> float:
    """JS over how often each predefined SMARTS occurs, MolDiff's variant."""
    keys = list(reference)
    left = np.array([len(reference[key]) + 1e-10 for key in keys])
    right = np.array([len(generated.get(key, ())) + 1e-10 for key in keys])
    return get_jsd(left / left.sum(), right / right.sum())


@dataclass
class Distributions:
    """Everything one population contributes to the table."""

    name: str
    n_molecules: int
    per_molecule: Dict[str, np.ndarray]
    lengths: Dict[str, np.ndarray]
    angles: Dict[str, np.ndarray]
    dihedral: Dict[str, np.ndarray]

    def column(self, key: str) -> np.ndarray:
        return self.per_molecule[key]


def collect_distributions(
    mols: Sequence[Chem.Mol], name: str, parallel: bool = False
) -> Distributions:
    """Run the MolDiff metric collectors over one population."""
    evaluation = moldiff()
    mols = [mol for mol in mols if mol is not None]

    per_molecule: Dict[str, List[float]] = {}
    for metric in ("drug_chem", "count_prop", "frags_counts", "ring_topo"):
        rows = evaluation.get_metric(mols, metric, parallel=parallel)
        for row in rows:
            for key, value in row.items():
                per_molecule.setdefault(key, []).append(float(value))

    local3d = evaluation.Local3D()
    local3d.get_predefined()
    lengths = local3d.calc_frequent(mols, "length", parallel=parallel)
    angles = local3d.calc_frequent(mols, "angle", parallel=parallel)
    dihedral = local3d.calc_frequent(mols, "dihedral", parallel=parallel)

    return Distributions(
        name=name,
        n_molecules=len(mols),
        per_molecule={
            key: np.asarray(values, dtype=float) for key, values in per_molecule.items()
        },
        lengths={key: np.asarray(value, dtype=float) for key, value in lengths.items()},
        angles={key: np.asarray(value, dtype=float) for key, value in angles.items()},
        dihedral={key: np.asarray(value, dtype=float) for key, value in dihedral.items()},
    )


def _mean_over_smarts(
    reference: Dict[str, np.ndarray],
    generated: Dict[str, np.ndarray],
    width: float,
) -> Tuple[float, Dict[str, float]]:
    per_smarts: Dict[str, float] = {}
    for key, values in reference.items():
        other = generated.get(key, np.asarray([], dtype=float))
        jsd_list, _, _ = compare_with_ref([values, other], width=width, discrete=False)
        per_smarts[key] = jsd_list[1]
    finite = [value for value in per_smarts.values() if np.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan")), per_smarts


def compare(
    reference: Distributions, generated: Distributions
) -> Dict[str, object]:
    """Produce every field of the unconditional result table."""
    table: Dict[str, object] = {
        "reference": reference.name,
        "generated": generated.name,
        "n_reference_molecules": reference.n_molecules,
        "n_generated_molecules": generated.n_molecules,
    }

    # Drug-likeness scalars, higher is better.
    for key, label in (
        ("qed", "qed_mean"),
        ("sa", "sa_normalized_mean"),
        ("lipinski", "lipinski_mean"),
        ("logp", "logp_mean"),
    ):
        if key in generated.per_molecule:
            table[label] = float(np.nanmean(generated.per_molecule[key]))
            table[f"reference_{label}"] = float(np.nanmean(reference.per_molecule[key]))

    # Local geometry, mean JS distance over the predefined SMARTS.
    table["js_bond_length"], table["js_bond_length_per_smarts"] = _mean_over_smarts(
        reference.lengths, generated.lengths, WIDTH_LENGTH
    )
    table["js_bond_angle"], table["js_bond_angle_per_smarts"] = _mean_over_smarts(
        reference.angles, generated.angles, WIDTH_ANGLE
    )
    table["js_dihedral"], table["js_dihedral_per_smarts"] = _mean_over_smarts(
        reference.dihedral, generated.dihedral, WIDTH_DIHEDRAL
    )

    # Substructure frequency distances.
    table["js_bond_types_frequency"] = _frequency_jsd(reference.lengths, generated.lengths)
    table["js_bond_pairs_frequency"] = _frequency_jsd(reference.angles, generated.angles)
    table["js_bond_triplets_frequency"] = _frequency_jsd(
        reference.dihedral, generated.dihedral
    )

    # Bond-type composition, summed over molecules.
    if all(key in reference.per_molecule for key in BOND_TYPE_KEYS):
        left = np.array(
            [reference.per_molecule[key].sum() for key in BOND_TYPE_KEYS]
        ) + 1e-10
        right = np.array(
            [generated.per_molecule.get(key, np.zeros(1)).sum() for key in BOND_TYPE_KEYS]
        ) + 1e-10
        table["js_bond_type_counts"] = get_jsd(left / left.sum(), right / right.sum())

    # Bonds per atom, and ring counts.
    if {"n_bonds", "n_atoms"} <= set(reference.per_molecule):
        values = [
            dist.per_molecule["n_bonds"] / np.maximum(1.0, dist.per_molecule["n_atoms"])
            for dist in (reference, generated)
        ]
        jsd_list, _, _ = compare_with_ref(values, width=0.01, discrete=False)
        table["js_bonds_per_atom"] = jsd_list[1]

    if "n_rings" in reference.per_molecule:
        jsd_list, _, _ = compare_with_ref(
            [reference.per_molecule["n_rings"], generated.per_molecule["n_rings"]],
            width=0.5,
            discrete=True,
        )
        table["js_n_rings"] = jsd_list[1]

    ring_sizes: Dict[str, float] = {}
    for key in RING_SIZE_KEYS:
        if key not in reference.per_molecule:
            continue
        jsd_list, _, _ = compare_with_ref(
            [reference.per_molecule[key], generated.per_molecule.get(key, np.zeros(1))],
            width=0.01,
            discrete=True,
        )
        ring_sizes[key] = jsd_list[1]
    if ring_sizes:
        finite = [value for value in ring_sizes.values() if np.isfinite(value)]
        table["js_ring_sizes"] = float(np.mean(finite)) if finite else float("nan")
        table["js_ring_sizes_per_size"] = ring_sizes

    # Intersecting rings: MolDiff's hub atoms (atoms in three or more rings).
    if "n_hub_atoms" in generated.per_molecule:
        table["intersecting_ring_atoms_mean"] = float(
            np.nanmean(generated.per_molecule["n_hub_atoms"])
        )
        table["reference_intersecting_ring_atoms_mean"] = float(
            np.nanmean(reference.per_molecule["n_hub_atoms"])
        )
        ring_membership = [
            f"n_atoms_in_{index}_rings" for index in range(0, 11)
        ]
        if all(key in reference.per_molecule for key in ring_membership):
            left = np.array(
                [reference.per_molecule[key].sum() for key in ring_membership]
            ) + 1e-10
            right = np.array(
                [
                    generated.per_molecule.get(key, np.zeros(1)).sum()
                    for key in ring_membership
                ]
            ) + 1e-10
            table["js_ring_membership"] = get_jsd(left / left.sum(), right / right.sum())

    # Distribution diagnostics.
    for key, label in (("n_atoms", "atom_count"), ("weight", "molecular_weight")):
        if key in generated.per_molecule:
            table[f"{label}_mean"] = float(np.nanmean(generated.per_molecule[key]))
            table[f"reference_{label}_mean"] = float(
                np.nanmean(reference.per_molecule[key])
            )
    return table


def uniqueness_and_novelty(
    smiles: Sequence[str], training_smiles: Optional[Iterable[str]] = None
) -> Dict[str, object]:
    """Uniqueness among the valid population and novelty against training."""
    unique = set(smiles)
    payload: Dict[str, object] = {
        "n_valid": len(smiles),
        "n_unique": len(unique),
        "uniqueness": len(unique) / max(1, len(smiles)),
    }
    if training_smiles is not None:
        known = set(training_smiles)
        novel = [value for value in unique if value not in known]
        payload["novelty"] = len(novel) / max(1, len(unique))
        payload["n_novel"] = len(novel)
        payload["n_training_reference"] = len(known)
    return payload


# --------------------------------------------------------------------------- #
# Populations
# --------------------------------------------------------------------------- #


def load_generated(
    attempts_path: str | Path, limit: int = 1_000
) -> Tuple[List[Chem.Mol], List[str], Dict[str, object]]:
    """Rebuild the primary population from saved raw attempts.

    The primary population is the first ``limit`` valid connected molecules in
    deterministic sample order; raw-attempt statistics are computed over every
    attempt up to that point.
    """
    from rdkit.Geometry import Point3D

    mols: List[Chem.Mol] = []
    smiles: List[str] = []
    attempts = 0
    failures: Counter = Counter()

    with open(attempts_path) as handle:
        for line in handle:
            payload = json.loads(line)
            attempts += 1
            reason = payload.get("failure_reason")
            connected = payload.get("connected")
            if reason or not connected:
                failures[reason or "disconnected"] += 1
                continue
            coordinates = payload.get("coordinates")
            canonical = payload.get("parsed_canonical_smiles")
            if not coordinates or not canonical:
                failures["other"] += 1
                continue
            mol = Chem.MolFromSmiles(payload["right_smiles"])
            if mol is None or mol.GetNumAtoms() != len(coordinates):
                failures["atom_count_mismatch"] += 1
                continue
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for index, (x, y, z) in enumerate(coordinates):
                conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
            mol.RemoveAllConformers()
            mol.AddConformer(conformer, assignId=True)
            mols.append(mol)
            smiles.append(canonical)
            if len(mols) >= limit:
                break

    diagnostics = {
        "attempts_to_population": attempts,
        "population": len(mols),
        "failure_reasons": dict(failures),
        "valid_connected_yield": len(mols) / max(1, attempts),
    }
    return mols, smiles, diagnostics


def load_reference(source: str = "test", limit: Optional[int] = None) -> List[Chem.Mol]:
    """Load the frozen reference population, one conformer per molecule.

    ``test`` is the frozen 1,000-molecule benchmark set and is the primary
    reference; ``valid`` is the held-out GEOM validation split and is the
    secondary compatibility table. The two are never combined.
    """
    if source == "test":
        import cloudpickle

        with open(experiment_config.BENCHMARK_TEST_PICKLE, "rb") as handle:
            payload = cloudpickle.load(handle)
        mols: List[Chem.Mol] = []
        for _, entry in sorted(payload.items()):
            confs = entry.get("confs") or []
            if confs:
                mols.append(Chem.RemoveHs(confs[0]))
            if limit and len(mols) >= limit:
                break
        return mols

    if source == "valid":
        from molgen3D.experiments.bindgpt_fsq.data.extract_source import (
            iter_geom_jsonl,
            strip_coordinates,
        )
        from molgen3D.experiments.bindgpt_fsq.sequences import (
            atom_coordinates,
            parse_enriched,
        )
        from rdkit.Geometry import Point3D
        from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles

        mols = []
        seen: set[str] = set()
        for row in iter_geom_jsonl("valid", verify_historical_fsq=False):
            graph = strip_coordinates(row.enriched_text)
            if graph in seen:
                continue  # one conformer per molecule, matching the test path
            seen.add(graph)
            mol = Chem.MolFromSmiles(strip_smiles(graph))
            if mol is None:
                continue
            try:
                coordinates = atom_coordinates(parse_enriched(row.enriched_text))
            except Exception:  # noqa: BLE001
                continue
            if mol.GetNumAtoms() != coordinates.shape[0]:
                continue
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for index, (x, y, z) in enumerate(coordinates):
                conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
            mol.RemoveAllConformers()
            mol.AddConformer(conformer, assignId=True)
            mols.append(mol)
            if limit and len(mols) >= limit:
                break
        return mols

    raise ValueError(f"unknown reference source {source!r}; use 'test' or 'valid'")
