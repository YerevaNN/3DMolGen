"""
AIMNet2 utility functions for molecular geometry calculations.

Adapted from LoQI repository (https://github.com/isayevlab/LoQI)
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from typing import Tuple, List, Dict, Any
from tqdm.auto import tqdm
from time import perf_counter


def generate_canonical_key(*components):
    """
    Generate a canonical key for any molecular component (atoms, bonds).
    This works for angles, bond lengths, and torsions.
    """
    key1 = tuple(components)
    key2 = tuple(reversed(components))
    return min(key1, key2)


@torch.jit.script
class FIRE:
    """Fast Inertial Relaxation Engine optimizer."""

    def __init__(self, M: int, N: int, device: str):
        ## default parameters
        self.dt_max = 0.1
        self.Nmin = 5
        self.maxstep = 0.1
        self.finc = 1.2
        self.fdec = 0.8
        self.astart = 0.1
        self.fa = 0.99
        self.dt_start = 0.1

        self.v = torch.zeros(M, N, 3, device=device)
        self.Nsteps = torch.zeros(M, dtype=torch.long, device=device)
        self.dt = torch.full((M,), self.dt_start, device=device)
        self.a = torch.full((M,), self.astart, device=device)

    def __call__(self, forces):
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            self.v = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
            self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            self.v[w_vf] = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)

            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = torch.tensor(self.astart, device=self.a.device)
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = torch.tensor(0.0, device=self.v.device)
            self.a[w_vf] = torch.tensor(self.astart, device=self.a.device)
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = torch.tensor(0, device=self.v.device)

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return dr

    def clean(self, mask) -> bool:
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True

    def extend(self, n: int) -> bool:
        self.v = torch.cat([self.v, torch.zeros(n, self.v.shape[1], self.v.shape[2], dtype=self.v.dtype, device=self.v.device)], dim=0)
        self.Nsteps = torch.cat([self.Nsteps, torch.zeros(n, dtype=self.Nsteps.dtype, device=self.Nsteps.device)], dim=0)
        self.dt = torch.cat([self.dt, torch.full((n, ), self.dt_start, device=self.dt.device)], dim=0)
        self.a = torch.cat([self.a, torch.full((n, ), self.astart, device=self.a.device)], dim=0)
        return True


def group_opt(model, coord, numbers, charge, batchsize=None, fmax=2e-3, max_nstep=5000, device="cuda"):
    """Geometry optimization using FIRE optimizer."""
    num_converged = 0
    converged_coord = []
    converged_idx = []
    converged_energy = []
    unconverged_coord = []
    unconverged_idx = []
    unconverged_energy = []
    idx = torch.arange(coord.shape[0]).cuda()
    runtime = torch.zeros_like(idx, dtype=torch.float32)
    nstep = torch.zeros_like(idx)

    if batchsize is None:
       batchsize = len(numbers)
    act_idx = idx[:batchsize]
    act_i = batchsize
    act_coord = coord[act_idx]
    act_numbers = numbers[act_idx]
    act_charge = charge[act_idx]
    act_runtime = torch.zeros(act_idx.shape[0]).cuda()
    act_nstep = torch.zeros(act_idx.shape[0]).cuda()

    istep = 0
    opt = FIRE(act_coord.shape[0], act_coord.shape[1], device)
    pbar = tqdm(total=len(coord), leave=True)
    pbar1 = tqdm(total=max_nstep, leave=False)
    _t = perf_counter()

    with torch.no_grad():
      while istep < max_nstep:
        _need_ext = False

        with torch.enable_grad():
          act_coord.requires_grad_(True)
          d = dict(coord=act_coord, numbers=act_numbers, charge=act_charge)
          dout = model(d)
          e = dout['energy']
          if 'forces' in dout:
              f = dout['forces']
          else:
              f = - torch.autograd.grad([e.sum()], [act_coord], retain_graph=False)[0]

        w = act_nstep >= max_nstep
        if w.any():
           nw = ~w
           unconverged_coord.append(act_coord[w].detach())
           unconverged_idx.append(act_idx[w])
           unconverged_energy.append(e[w].detach())
           nstep[act_idx[w]] = act_nstep[w].long()
           runtime[act_idx[w]] = act_runtime[w]
           act_idx = act_idx[nw]
           act_coord = act_coord[nw]
           act_numbers = act_numbers[nw]
           act_charge = act_charge[nw]
           act_runtime = act_runtime[nw]
           act_nstep = act_nstep[nw]
           f = f[nw]
           e = e[nw]
           opt.clean(nw)
           _need_ext = True

        if istep and not istep % 10:
          _t1 = perf_counter()
          act_runtime += (_t1 - _t) / act_runtime.shape[0]
          _t = _t1
          act_nstep += 10
          _fmax = f.norm(dim=-1).max(dim=-1)[0]
          w = _fmax < fmax
          if w.any():
            pbar.update(w.sum().item())
            num_converged += w.sum().item()
            converged_coord.append(act_coord[w].detach())
            converged_idx.append(act_idx[w])
            converged_energy.append(e[w].detach())
            nstep[act_idx[w]] = act_nstep[w].long()
            runtime[act_idx[w]] = act_runtime[w]

            nw = ~w
            act_idx = act_idx[nw]
            act_coord = act_coord[nw]
            act_numbers = act_numbers[nw]
            act_charge = act_charge[nw]
            act_runtime = act_runtime[nw]
            act_nstep = act_nstep[nw]

            f = f[nw]
            opt.clean(nw)
            _need_ext = act_i <= coord.shape[0]

        _prev_istep = istep

        istep += 1

        assert act_coord.shape[0] == opt.v.shape[0]
        if act_coord.numel() > 0:
            act_coord += opt(f)

        if _need_ext:
            _n_add = batchsize - act_idx.shape[0]
            act_idx = torch.cat([act_idx, idx[act_i:act_i+_n_add]], dim=0)
            _n1 = act_coord.shape[0]
            act_coord = torch.cat([act_coord, coord[act_i:act_i+_n_add]], dim=0)
            act_numbers = torch.cat([act_numbers, numbers[act_i:act_i+_n_add]], dim=0)
            act_charge = torch.cat([act_charge, charge[act_i:act_i+_n_add]], dim=0)
            act_runtime = torch.cat([act_runtime, torch.zeros(act_charge.shape[0]-act_runtime.shape[0], device='cuda')], dim=0)
            act_nstep = torch.cat([act_nstep, torch.zeros(act_charge.shape[0]-act_nstep.shape[0], device='cuda')])

            act_i += _n_add
            opt.extend(act_coord.shape[0] - _n1)
            istep = 0

        pbar1.update(istep - _prev_istep)

        if act_coord.numel() == 0:
            break

    pbar.close()
    pbar1.close()
    converged = torch.zeros_like(charge)
    res_coord = torch.zeros_like(coord)
    res_forces = torch.zeros_like(coord)
    res_energy = torch.zeros_like(charge, dtype=torch.double)

    if len(converged_idx) > 0:
        opt_coord = torch.cat(converged_coord, dim=0)
        opt_idx = torch.cat(converged_idx, dim=0)
        opt_energy = torch.cat(converged_energy)

        converged[opt_idx] = 1
        res_coord[opt_idx] = opt_coord
        res_energy[opt_idx] = opt_energy
        res_forces[opt_idx] = fmax/1.73205

    if not w.all():
        res_coord[act_idx] = act_coord
        res_energy[act_idx] = e
        res_forces[act_idx] = f
        nstep[act_idx] = max_nstep

    if len(unconverged_idx) > 0:
        opt_coord = torch.cat(unconverged_coord, dim=0)
        opt_idx = torch.cat(unconverged_idx, dim=0)
        opt_energy = torch.cat(unconverged_energy)

        res_coord[opt_idx] = opt_coord
        res_energy[opt_idx] = opt_energy
        res_forces[opt_idx] = -fmax/1.73205

    return converged, res_coord, res_energy, res_forces, nstep


def check_topology(adjacency_matrix, numbers, coordinates):
    """Check molecular topology consistency."""
    try:
        # Create RDKit molecule from adjacency matrix and atomic numbers
        mol = Chem.RWMol()
        atom_map = {}

        # Add atoms
        for i, atomic_num in enumerate(numbers):
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_map[i] = atom_idx

        # Add bonds based on adjacency matrix
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if adjacency_matrix[i, j] > 0:
                    # Determine bond order from adjacency matrix value
                    bond_order = int(adjacency_matrix[i, j])
                    if bond_order == 1:
                        bond_type = Chem.BondType.SINGLE
                    elif bond_order == 2:
                        bond_type = Chem.BondType.DOUBLE
                    elif bond_order == 3:
                        bond_type = Chem.BondType.TRIPLE
                    else:
                        bond_type = Chem.BondType.SINGLE

                    mol.AddBond(atom_map[i], atom_map[j], bond_type)

        # Convert to regular molecule and sanitize
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)

        # Add conformer with coordinates
        if coordinates is not None:
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for i, coord in enumerate(coordinates):
                conformer.SetAtomPosition(i, coord)
            mol.AddConformer(conformer)

        return [True], mol

    except Exception as e:
        return [False], str(e)


def compute_bond_lengths_diff(mol_pair):
    """Compute bond length differences between molecule pairs."""
    mol1, mol2 = mol_pair

    # Compute bond lengths for each molecule
    bond_lengths1 = compute_bond_lengths(mol1)
    bond_lengths2 = compute_bond_lengths(mol2)

    # Dictionary to store bond length differences
    bond_diff_dict = {}

    # Find common keys and compute differences
    common_keys = set(bond_lengths1.keys()) & set(bond_lengths2.keys())

    for key in common_keys:
        lengths1 = np.array(bond_lengths1[key][0])
        lengths2 = np.array(bond_lengths2[key][0])

        # Compute differences (assuming same number of bonds of each type)
        if len(lengths1) == len(lengths2):
            diffs = np.abs(lengths1 - lengths2)
            bond_diff_dict[key] = (diffs.tolist(), len(diffs))
        else:
            # If different numbers, compute all pairwise differences
            diffs = []
            for l1 in lengths1:
                for l2 in lengths2:
                    diffs.append(abs(l1 - l2))
            bond_diff_dict[key] = (diffs, len(diffs))

    return bond_diff_dict


def compute_bond_lengths(rdkit_mol):
    """Compute bond lengths for a single molecule (LoQI style)."""
    bond_lengths = {}

    conf = rdkit_mol.GetConformer()

    for bond in rdkit_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1_type, atom2_type = rdkit_mol.GetAtomWithIdx(
            idx1).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(idx2).GetAtomicNum()
        bond_type_numeric = int(bond.GetBondType())
        length = rdMolTransforms.GetBondLength(conf, idx1, idx2)
        key = generate_canonical_key(atom1_type, bond_type_numeric, atom2_type)
        if key not in bond_lengths:
            bond_lengths[key] = [[], 0]
        bond_lengths[key][0].append(length)
        bond_lengths[key][1] += 1
    return bond_lengths


def compute_bond_angles_diff(mol_pair):
    """Compute bond angle differences between molecule pairs."""
    mol1, mol2 = mol_pair

    # Compute bond angles for each molecule
    bond_angles1 = compute_bond_angles(mol1)
    bond_angles2 = compute_bond_angles(mol2)

    # Dictionary to store angle differences
    angle_diff_dict = {}

    # Find common keys and compute differences
    common_keys = set(bond_angles1.keys()) & set(bond_angles2.keys())

    for key in common_keys:
        angles1 = np.array(bond_angles1[key][0])
        angles2 = np.array(bond_angles2[key][0])

        # Compute differences (assuming same number of angles of each type)
        if len(angles1) == len(angles2):
            diffs = np.abs(angles1 - angles2)
            angle_diff_dict[key] = (diffs.tolist(), len(diffs))
        else:
            # If different numbers, compute all pairwise differences
            diffs = []
            for a1 in angles1:
                for a2 in angles2:
                    diffs.append(abs(a1 - a2))
            angle_diff_dict[key] = (diffs, len(diffs))

    return angle_diff_dict


def compute_bond_angles(rdkit_mol):
    """Compute bond angles for a single molecule (LoQI style)."""
    bond_angles = {}
    conf = rdkit_mol.GetConformer()

    for atom in rdkit_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                idx1, idx2, idx3 = neighbors[i].GetIdx(), atom.GetIdx(), neighbors[j].GetIdx()
                atom1_type, atom2_type, atom3_type = rdkit_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                bond_type_1 = int(rdkit_mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
                bond_type_2 = int(rdkit_mol.GetBondBetweenAtoms(idx2, idx3).GetBondType())

                angle_init = rdMolTransforms.GetAngleDeg(conf, idx1, idx2, idx3)

                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type)
                if key not in bond_angles:
                    bond_angles[key] = [[], 0]
                bond_angles[key][0].append(angle_init)
                bond_angles[key][1] += 1

    return bond_angles


def compute_torsion_angles_diff(mol_pair):
    """Compute torsion angle differences between molecule pairs."""
    mol1, mol2 = mol_pair

    # Compute torsion angles for each molecule
    torsion_angles1 = compute_torsion_angles(mol1)
    torsion_angles2 = compute_torsion_angles(mol2)

    # Dictionary to store torsion differences
    torsion_diff_dict = {}

    # Find common keys and compute differences
    common_keys = set(torsion_angles1.keys()) & set(torsion_angles2.keys())

    for key in common_keys:
        torsions1 = np.array(torsion_angles1[key][0])
        torsions2 = np.array(torsion_angles2[key][0])

        # Compute differences (handling periodic nature of torsion angles)
        if len(torsions1) == len(torsions2):
            diffs = []
            for t1, t2 in zip(torsions1, torsions2):
                diff = abs(t1 - t2)
                diff = min(diff, 360 - diff)  # Handle circular nature
                diffs.append(diff)
            torsion_diff_dict[key] = (diffs, len(diffs))
        else:
            # If different numbers, compute all pairwise differences
            diffs = []
            for t1 in torsions1:
                for t2 in torsions2:
                    diff = abs(t1 - t2)
                    diff = min(diff, 360 - diff)
                    diffs.append(diff)
            torsion_diff_dict[key] = (diffs, len(diffs))

    return torsion_diff_dict


def compute_torsion_angles(rdkit_mol):
    """Compute torsion angles for a single molecule (LoQI style)."""
    torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsionSmarts)

    torsion_angles = {}

    init_conf = rdkit_mol.GetConformer()

    torsion_matches = rdkit_mol.GetSubstructMatches(torsion_query)

    for match in torsion_matches:
        idx2, idx3 = match[0], match[1]
        bond = rdkit_mol.GetBondBetweenAtoms(idx2, idx3)

        for b1 in rdkit_mol.GetAtomWithIdx(idx2).GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in rdkit_mol.GetAtomWithIdx(idx3).GetBonds():
                if b2.GetIdx() == bond.GetIdx() or b2.GetIdx() == b1.GetIdx():
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                if idx4 == idx1:
                    continue

                atom1_type, atom2_type, atom3_type, atom4_type = rdkit_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(
                    idx3).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(idx4).GetAtomicNum()
                bond_type_1 = int(b1.GetBondType())
                bond_type_2 = int(bond.GetBondType())
                bond_type_3 = int(b2.GetBondType())

                angle = rdMolTransforms.GetDihedralDeg(init_conf, idx1, idx2, idx3, idx4)
                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type, bond_type_3, atom4_type)

                if key not in torsion_angles:
                    torsion_angles[key] = [[], 0]
                torsion_angles[key][0].append(angle)
                torsion_angles[key][1] += 1

    return torsion_angles
