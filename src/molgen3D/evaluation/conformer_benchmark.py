"""Geometric conformer evaluation metrics for benchmark analysis."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolAlign, rdMolTransforms

PAIRWISE_SAMPLE_CAP = 120
RANDOM_SEED = 42
BOND_LENGTH_FAIL_FRAC = 0.15
ANGLE_FAIL_DEG = 15.0

PRIMARY_CLUSTER_THRESHOLDS = (0.5, 0.75, 1.0, 1.25)
SUPPLEMENTARY_CLUSTER_THRESHOLDS = (2.0, 3.0)
ALL_CLUSTER_THRESHOLDS = PRIMARY_CLUSTER_THRESHOLDS + SUPPLEMENTARY_CLUSTER_THRESHOLDS
CASF_GEOMETRIC_CLUSTER_THRESHOLDS = (0.5, 1.0, 2.0, 3.0)
BIOACTIVE_HIT_THRESHOLDS = (0.5, 1.0, 1.5, 2.0)
COVERAGE_THRESHOLDS = (1.0, 2.0)


def threshold_tag(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def mol_no_h(mol: Chem.Mol) -> Chem.Mol:
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        try:
            return Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            return mol


def best_aligned_rmsd(gen: Chem.Mol, ref: Chem.Mol) -> float:
    """Heavy-atom symmetry-aware aligned RMSD between two single-conformer mols."""
    a = mol_no_h(gen)
    b = mol_no_h(ref)
    if a.GetNumAtoms() != b.GetNumAtoms():
        return math.nan
    try:
        return float(rdMolAlign.GetBestRMS(a, b, prbId=0, refId=0))
    except Exception:
        try:
            atom_map = [(idx, idx) for idx in range(a.GetNumAtoms())]
            return float(rdMolAlign.AlignMol(a, b, prbCid=0, refCid=0, atomMap=atom_map))
        except Exception:
            return math.nan


def rmsd_profile(
    gen_mols: Sequence[Chem.Mol],
    ref_mol: Chem.Mol,
    thresholds: Sequence[float] = BIOACTIVE_HIT_THRESHOLDS,
) -> dict[str, float]:
    values = [best_aligned_rmsd(mol, ref_mol) for mol in gen_mols]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {
            "best_rmsd": math.nan,
            "median_rmsd": math.nan,
            "p90_rmsd": math.nan,
            **{f"frac_lt_{str(t).replace('.', 'p')}": math.nan for t in thresholds},
        }
    arr = np.asarray(finite, dtype=float)
    out = {
        "best_rmsd": float(np.min(arr)),
        "median_rmsd": float(np.median(arr)),
        "p90_rmsd": float(np.percentile(arr, 90)),
    }
    for threshold in thresholds:
        tag = str(threshold).replace(".", "p")
        out[f"frac_lt_{tag}"] = float(np.mean(arr <= threshold))
    return out


def bioactive_hit_at_k(
    gen_mols: Sequence[Chem.Mol],
    ref_mol: Chem.Mol,
    thresholds: Sequence[float] = BIOACTIVE_HIT_THRESHOLDS,
) -> dict[str, float]:
    profile = rmsd_profile(gen_mols, ref_mol, thresholds=thresholds)
    best = profile["best_rmsd"]
    hits = {}
    for threshold in thresholds:
        tag = str(threshold).replace(".", "p")
        hits[f"hit_{tag}"] = float(best <= threshold) if math.isfinite(best) else math.nan
    hits["best_rmsd"] = best
    hits["median_rmsd"] = profile["median_rmsd"]
    return hits


def sample_mols(mols: Sequence[Chem.Mol], cap: int = PAIRWISE_SAMPLE_CAP) -> list[Chem.Mol]:
    if len(mols) <= cap:
        return list(mols)
    rng = random.Random(RANDOM_SEED)
    indices = sorted(rng.sample(range(len(mols)), cap))
    return [mols[i] for i in indices]


def heavy_rmsd_same_topology(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    return best_aligned_rmsd(mol_a, mol_b)


def greedy_cluster_count(mols: Sequence[Chem.Mol], threshold: float) -> int:
    """Deterministic greedy clustering in generation order (no subsampling)."""
    centers: list[Chem.Mol] = []
    for mol in mols:
        if not centers:
            centers.append(mol)
            continue
        distances = [heavy_rmsd_same_topology(mol, center) for center in centers]
        finite = [value for value in distances if math.isfinite(value)]
        if not finite or min(finite) >= threshold:
            centers.append(mol)
    return len(centers)


def clusters_per_100(mols: Sequence[Chem.Mol], threshold: float) -> float:
    """Return greedy RMSD clusters normalized to 100 conformers for one ligand."""
    if not mols:
        return math.nan
    return 100.0 * greedy_cluster_count(mols, threshold) / len(mols)


def greedy_cluster_assignments(mols: Sequence[Chem.Mol], threshold: float) -> list[int]:
    """Return cluster id per conformer (0-indexed) using greedy rule."""
    assignments: list[int] = []
    centers: list[Chem.Mol] = []
    for mol in mols:
        if not centers:
            centers.append(mol)
            assignments.append(0)
            continue
        distances = [heavy_rmsd_same_topology(mol, center) for center in centers]
        finite = [(idx, value) for idx, value in enumerate(distances) if math.isfinite(value)]
        if not finite:
            centers.append(mol)
            assignments.append(len(centers) - 1)
            continue
        nearest_idx, nearest_dist = min(finite, key=lambda item: item[1])
        if nearest_dist < threshold:
            assignments.append(nearest_idx)
        else:
            centers.append(mol)
            assignments.append(len(centers) - 1)
    return assignments


def cluster_entropy(mols: Sequence[Chem.Mol], threshold: float = 1.0) -> float:
    if len(mols) <= 1:
        return 0.0
    assignments = greedy_cluster_assignments(mols, threshold)
    counts = np.bincount(np.asarray(assignments, dtype=int))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = math.log(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def cluster_distribution_metrics(assignments: Sequence[int]) -> dict[str, float]:
    """Describe how evenly one ligand's conformers occupy greedy RMSD clusters.

    ``cluster_entropy`` is Shannon entropy over cluster occupancies normalized by
    log(cluster_count), so 1.0 means conformers are evenly spread across clusters
    and values near 0.0 mean one cluster dominates. ``largest_cluster_fraction``
    is the fraction of conformers assigned to the largest cluster.
    """
    if len(assignments) == 0:
        return {
            "cluster_count": 0.0,
            "largest_cluster_fraction": math.nan,
            "cluster_entropy": math.nan,
            "effective_cluster_count": math.nan,
            "simpson_concentration": math.nan,
            "singleton_fraction": math.nan,
        }
    counts = np.bincount(np.asarray(assignments, dtype=int))
    counts = counts[counts > 0]
    total = float(counts.sum())
    probs = counts / total
    entropy = -float(np.sum(probs * np.log(probs))) if len(probs) > 1 else 0.0
    normalized_entropy = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0
    simpson = float(np.sum(probs * probs))
    return {
        "cluster_count": float(len(counts)),
        "largest_cluster_fraction": float(np.max(probs)),
        "cluster_entropy": normalized_entropy,
        "effective_cluster_count": float(math.exp(entropy)),
        "simpson_concentration": simpson,
        "singleton_fraction": float(np.mean(counts == 1)),
    }


def greedy_cluster_metrics(
    mols: Sequence[Chem.Mol],
    thresholds: Sequence[float] = CASF_GEOMETRIC_CLUSTER_THRESHOLDS,
) -> dict[str, float]:
    """Compute per-ligand greedy RMSD cluster counts and normalized density metrics."""
    """Greedy RMSD-aligned clustering plus occupancy metrics for each threshold."""
    out: dict[str, float] = {"conformer_count": float(len(mols))}
    if not mols:
        for threshold in thresholds:
            tag = threshold_tag(threshold)
            out[f"greedy_clusters_{tag}"] = math.nan
            out[f"clusters_per_100_{tag}"] = math.nan
            out[f"cluster_entropy_{tag}"] = math.nan
            out[f"largest_cluster_fraction_{tag}"] = math.nan
            out[f"effective_clusters_{tag}"] = math.nan
            out[f"simpson_concentration_{tag}"] = math.nan
            out[f"singleton_fraction_{tag}"] = math.nan
        return out
    for threshold in thresholds:
        tag = threshold_tag(threshold)
        assignments = greedy_cluster_assignments(mols, threshold)
        dist = cluster_distribution_metrics(assignments)
        cluster_count = dist["cluster_count"]
        out[f"greedy_clusters_{tag}"] = cluster_count
        out[f"clusters_per_100_{tag}"] = 100.0 * cluster_count / len(mols)
        out[f"cluster_entropy_{tag}"] = dist["cluster_entropy"]
        out[f"largest_cluster_fraction_{tag}"] = dist["largest_cluster_fraction"]
        out[f"effective_clusters_{tag}"] = dist["effective_cluster_count"]
        out[f"simpson_concentration_{tag}"] = dist["simpson_concentration"]
        out[f"singleton_fraction_{tag}"] = dist["singleton_fraction"]
    return out


def duplicate_fraction(mols: Sequence[Chem.Mol], threshold: float, cap: int = PAIRWISE_SAMPLE_CAP) -> float:
    sampled = sample_mols(mols, cap=cap)
    if len(sampled) < 2:
        return math.nan
    duplicates = 0
    total = 0
    for mol_a, mol_b in itertools.combinations(sampled, 2):
        value = heavy_rmsd_same_topology(mol_a, mol_b)
        if not math.isfinite(value):
            continue
        total += 1
        if value < threshold:
            duplicates += 1
    return duplicates / total if total else math.nan


def pairwise_rmsd_stats(mols: Sequence[Chem.Mol], cap: int = PAIRWISE_SAMPLE_CAP) -> dict[str, float]:
    sampled = sample_mols(mols, cap=cap)
    values = []
    for mol_a, mol_b in itertools.combinations(sampled, 2):
        value = heavy_rmsd_same_topology(mol_a, mol_b)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {"pairwise_mean": math.nan, "pairwise_p90": math.nan}
    arr = np.asarray(values, dtype=float)
    return {
        "pairwise_mean": float(np.mean(arr)),
        "pairwise_p90": float(np.percentile(arr, 90)),
    }


def get_rotatable_torsions(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern)
    torsions: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for j, k in matches:
        bond_key = tuple(sorted((j, k)))
        if bond_key in seen:
            continue
        seen.add(bond_key)
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)
        neighbors_j = [
            atom.GetIdx()
            for atom in atom_j.GetNeighbors()
            if atom.GetIdx() != k and atom.GetAtomicNum() > 1
        ]
        neighbors_k = [
            atom.GetIdx()
            for atom in atom_k.GetNeighbors()
            if atom.GetIdx() != j and atom.GetAtomicNum() > 1
        ]
        if not neighbors_j:
            neighbors_j = [atom.GetIdx() for atom in atom_j.GetNeighbors() if atom.GetIdx() != k]
        if not neighbors_k:
            neighbors_k = [atom.GetIdx() for atom in atom_k.GetNeighbors() if atom.GetIdx() != j]
        if neighbors_j and neighbors_k:
            torsions.append((neighbors_j[0], j, k, neighbors_k[0]))
    return torsions


def mean_torsion_std_deg(mols: Sequence[Chem.Mol]) -> float:
    if not mols:
        return math.nan
    torsions = get_rotatable_torsions(mols[0])
    if not torsions:
        return math.nan
    profiles = []
    for mol in mols:
        conf = mol.GetConformer(0)
        profiles.append([rdMolTransforms.GetDihedralDeg(conf, *torsion) for torsion in torsions])
    return float(np.mean(np.std(np.asarray(profiles, dtype=float), axis=0)))


def chembl_coverage_fraction(
    gen_mols: Sequence[Chem.Mol],
    chembl_mols: Sequence[Chem.Mol],
    threshold: float,
) -> float:
    if not chembl_mols:
        return math.nan
    covered = 0
    for ref in chembl_mols:
        best = min((best_aligned_rmsd(gen, ref) for gen in gen_mols), default=math.nan)
        if math.isfinite(best) and best <= threshold:
            covered += 1
    return covered / len(chembl_mols)


def chembl_reference_rmsd_stats(
    gen_mols: Sequence[Chem.Mol],
    chembl_mols: Sequence[Chem.Mol],
) -> dict[str, float]:
    if not gen_mols or not chembl_mols:
        return {
            "best_rmsd": math.nan,
            "median_best_rmsd": math.nan,
            "mean_median_rmsd": math.nan,
        }
    per_gen_best = []
    for gen in gen_mols:
        rmsds = [best_aligned_rmsd(gen, ref) for ref in chembl_mols]
        finite = [v for v in rmsds if math.isfinite(v)]
        per_gen_best.append(min(finite) if finite else math.nan)
    per_gen_median = []
    for gen in gen_mols:
        rmsds = [best_aligned_rmsd(gen, ref) for ref in chembl_mols]
        finite = [v for v in rmsds if math.isfinite(v)]
        per_gen_median.append(float(np.median(finite)) if finite else math.nan)
    finite_best = [v for v in per_gen_best if math.isfinite(v)]
    finite_median = [v for v in per_gen_median if math.isfinite(v)]
    return {
        "best_rmsd": min(finite_best) if finite_best else math.nan,
        "median_best_rmsd": float(np.median(finite_best)) if finite_best else math.nan,
        "mean_median_rmsd": float(np.mean(finite_median)) if finite_median else math.nan,
    }


def check_validity(mol: Chem.Mol) -> tuple[bool, str]:
    if mol is None:
        return False, "none_mol"
    if mol.GetNumConformers() == 0:
        return False, "no_conformer"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return False, f"sanitize:{type(exc).__name__}"
    conf = mol.GetConformer(0)
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        if not all(map(math.isfinite, (pos.x, pos.y, pos.z))):
            return False, "non_finite_coords"
    return True, "ok"


def check_smiles_match(gen_mol: Chem.Mol, ref_smiles: str, isomeric: bool = True) -> bool:
    if not ref_smiles:
        return False
    try:
        gen_smiles = Chem.MolToSmiles(mol_no_h(gen_mol), isomericSmiles=isomeric)
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is None:
            return gen_smiles == ref_smiles
        ref_canonical = Chem.MolToSmiles(ref_mol, isomericSmiles=isomeric)
        return gen_smiles == ref_canonical
    except Exception:
        return False


def posebusters_pass(mol: Chem.Mol) -> tuple[bool, str]:
    try:
        from posebusters import PoseBusters
    except ImportError:
        return False, "posebusters_unavailable"
    try:
        bust = PoseBusters()
        frame = bust.bust([mol], None, None, full_report=False)
        passed = bool(frame.all(axis=1).iloc[0])
        return passed, "ok" if passed else "posebusters_fail"
    except Exception as exc:
        return False, f"posebusters_error:{type(exc).__name__}"


def forcefield_energy(mol: Chem.Mol, conf_id: int = 0, ff_variant: str = "MMFF94s") -> float:
    try:
        minimized = mol.GetProp("minimization_applied").strip().lower() == "true" if mol.HasProp("minimization_applied") else False
        if minimized and mol.HasProp("post_min_energy"):
            return float(mol.GetProp("post_min_energy"))
    except Exception:
        pass
    try:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return math.nan
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=ff_variant)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return math.nan
        return float(ff.CalcEnergy())
    except Exception:
        return math.nan


def energy_stats(mols: Sequence[Chem.Mol]) -> dict[str, float]:
    values = [forcefield_energy(mol) for mol in mols]
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {
            "energy_count": 0.0,
            "energy_min": math.nan,
            "energy_max": math.nan,
            "energy_median": math.nan,
            "energy_std": math.nan,
        }
    return {
        "energy_count": float(finite.size),
        "energy_min": float(np.min(finite)),
        "energy_max": float(np.max(finite)),
        "energy_median": float(np.median(finite)),
        "energy_std": float(np.std(finite)),
    }


def _minimize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    copy = Chem.Mol(mol)
    try:
        if not AllChem.MMFFHasAllMoleculeParams(copy):
            return None
        props = AllChem.MMFFGetMoleculeProperties(copy, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(copy, props, confId=0)
        if ff is None:
            return None
        ff.Minimize(maxIts=200)
        return copy
    except Exception:
        return None


def bond_angle_stats(mol: Chem.Mol) -> dict[str, float]:
    minimized = _minimize_copy(mol)
    if minimized is None:
        return {
            "bond_mae": math.nan,
            "angle_mae": math.nan,
            "bond_fail": math.nan,
            "angle_fail": math.nan,
        }
    conf_obs = mol.GetConformer(0)
    conf_ref = minimized.GetConformer(0)
    bond_errors = []
    bond_fails = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        obs = conf_obs.GetAtomPosition(i).Distance(conf_obs.GetAtomPosition(j))
        ref = conf_ref.GetAtomPosition(i).Distance(conf_ref.GetAtomPosition(j))
        if ref <= 1e-6:
            continue
        rel_err = abs(obs - ref) / ref
        bond_errors.append(rel_err)
        if rel_err > BOND_LENGTH_FAIL_FRAC:
            bond_fails += 1
    angle_errors = []
    angle_fails = 0
    for atom in mol.GetAtoms():
        if atom.GetDegree() < 2:
            continue
        neighbors = list(atom.GetNeighbors())
        for idx_a in range(len(neighbors)):
            for idx_b in range(idx_a + 1, len(neighbors)):
                a = neighbors[idx_a].GetIdx()
                b = atom.GetIdx()
                c = neighbors[idx_b].GetIdx()
                obs = rdMolTransforms.GetAngleDeg(conf_obs, a, b, c)
                ref = rdMolTransforms.GetAngleDeg(conf_ref, a, b, c)
                err = abs(obs - ref)
                angle_errors.append(err)
                if err > ANGLE_FAIL_DEG:
                    angle_fails += 1
    n_bonds = len(bond_errors)
    n_angles = len(angle_errors)
    return {
        "bond_mae": float(np.mean(bond_errors)) if bond_errors else math.nan,
        "angle_mae": float(np.mean(angle_errors)) if angle_errors else math.nan,
        "bond_fail": bond_fails / n_bonds if n_bonds else math.nan,
        "angle_fail": angle_fails / n_angles if n_angles else math.nan,
    }


def diversity_metrics(mols: Sequence[Chem.Mol]) -> dict[str, float]:
    if not mols:
        out: dict[str, float] = {
            "conformer_count": 0.0,
            "mean_torsion_std_deg": math.nan,
            "cluster_entropy_1p0": math.nan,
            "duplicate_fraction_1p0": math.nan,
            "pairwise_mean": math.nan,
            "pairwise_p90": math.nan,
        }
        for threshold in ALL_CLUSTER_THRESHOLDS:
            tag = str(threshold).replace(".", "p")
            out[f"greedy_clusters_{tag}"] = math.nan
            out[f"clusters_per_100_{tag}"] = math.nan
        return out
    pairwise = pairwise_rmsd_stats(mols)
    out = {
        "conformer_count": float(len(mols)),
        "mean_torsion_std_deg": mean_torsion_std_deg(mols),
        "cluster_entropy_1p0": cluster_entropy(mols, threshold=1.0),
        "duplicate_fraction_1p0": duplicate_fraction(mols, threshold=1.0),
        **pairwise,
    }
    for threshold in ALL_CLUSTER_THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        count = greedy_cluster_count(mols, threshold)
        out[f"greedy_clusters_{tag}"] = float(count)
        out[f"clusters_per_100_{tag}"] = clusters_per_100(mols, threshold)
    return out


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return math.nan
    xs = np.asarray([p[0] for p in pairs], dtype=float)
    ys = np.asarray([p[1] for p in pairs], dtype=float)
    x_rank = pd_rank(xs)
    y_rank = pd_rank(ys)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def pd_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def rank_series(values: Sequence[float], ascending: bool = True) -> list[float]:
    indexed = [(idx, value) for idx, value in enumerate(values)]
    finite = [(idx, value) for idx, value in indexed if math.isfinite(value)]
    finite.sort(key=lambda item: item[1], reverse=not ascending)
    ranks = [math.nan] * len(values)
    for rank, (idx, _value) in enumerate(finite, start=1):
        ranks[idx] = float(rank)
    return ranks


def rank_methods(
    metric_df: list[dict],
    method_key: str,
    rank_columns: dict[str, bool],
) -> tuple[list[dict], dict[str, float]]:
    """Add rank columns and pairwise Spearman correlations between rank columns."""
    rows = [dict(row) for row in metric_df]
    for column, ascending in rank_columns.items():
        values = [row.get(column, math.nan) for row in rows]
        ranks = rank_series(values, ascending=ascending)
        for row, rank in zip(rows, ranks):
            row[f"{column}_rank"] = rank
    correlations: dict[str, float] = {}
    columns = list(rank_columns)
    for idx_a in range(len(columns)):
        for idx_b in range(idx_a + 1, len(columns)):
            col_a, col_b = columns[idx_a], columns[idx_b]
            values_a = [row.get(f"{col_a}_rank", math.nan) for row in rows]
            values_b = [row.get(f"{col_b}_rank", math.nan) for row in rows]
            correlations[f"{col_a}_vs_{col_b}"] = spearman_correlation(values_a, values_b)
    return rows, correlations


def safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(statistics.mean(finite)) if finite else math.nan


def safe_median(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(statistics.median(finite)) if finite else math.nan


def safe_sum(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(sum(finite)) if finite else math.nan


def num_rotatable_bonds(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(Descriptors.NumRotatableBonds(mol))
