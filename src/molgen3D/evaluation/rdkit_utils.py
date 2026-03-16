"""RDKit utility functions - imported locally to avoid pickling issues."""

from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs


def clean_confs(smi, confs):
    """Clean conformers by checking SMILES consistency."""
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


def correct_smiles(true_confs):
    """Find the most common SMILES from conformers."""
    from statistics import mode, StatisticsError

    conf_smis = []
    for c in true_confs:
        conf_smi = Chem.MolToSmiles(RemoveHs(c))
        conf_smis.append(conf_smi)

    try:
        common_smi = mode(conf_smis)
    except StatisticsError:
        return None  # these should be cleaned by hand

    if sum(common_smi == smi for smi in conf_smis) == len(conf_smis):
        return mode(conf_smis)
    else:
        print('consensus', common_smi)  # these should probably also be investigated manually
        return common_smi


def get_unique_smiles(confs):
    """Get unique SMILES and their counts."""
    from collections import Counter
    smiles = [Chem.MolToSmiles(RemoveHs(c), isomericSmiles=False) for c in confs]
    smiles_count = Counter(smiles)
    return smiles_count


def process_molecules_remove_hs(model_preds):
    """Process molecules by removing hydrogens from conformers."""
    return {smi: [RemoveHs(m) for m in confs] for smi, confs in model_preds.items()}


def _best_rmsd(probe, ref, use_alignmol: bool):
    """Calculate RMSD between two molecules."""
    from rdkit.Chem import rdMolAlign as MA
    from rdkit import Chem

    try:
        if use_alignmol:
            return float(MA.AlignMol(probe, ref))
        return float(MA.GetBestRMS(probe, ref))
    except Exception:
        # Fallback: strip stereochemistry and retry. This rescues cases where
        # molecules share the same graph but stereo labels prevent matching.
        try:
            probe_no_stereo = Chem.Mol(probe)
            ref_no_stereo = Chem.Mol(ref)
            Chem.RemoveStereochemistry(probe_no_stereo)
            Chem.RemoveStereochemistry(ref_no_stereo)

            if use_alignmol:
                return float(MA.AlignMol(probe_no_stereo, ref_no_stereo))

            # Try explicit atom maps with chirality disabled.
            matches = probe_no_stereo.GetSubstructMatches(
                ref_no_stereo,
                uniquify=False,
                useChirality=False,
                maxMatches=2048,
            )
            if matches:
                best = None
                for match in matches:
                    atom_map = [(int(match[i]), i) for i in range(len(match))]
                    rms = float(MA.GetBestRMS(probe_no_stereo, ref_no_stereo, map=atom_map))
                    if best is None or rms < best:
                        best = rms
                if best is not None:
                    return best

            return float(MA.GetBestRMS(probe_no_stereo, ref_no_stereo))
        except Exception:
            import numpy as np
            return np.nan

