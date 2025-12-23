import ast
import math
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Geometry import Point3D
import numpy as np


def truncate(x, precision=4):
    """Format a float with exactly ``precision`` decimal places (truncation, not rounding)."""
    if precision < 0:
        raise ValueError("precision must be non-negative")

    value = float(x)
    if precision == 0:
        return str(int(math.trunc(value)))

    factor = 10 ** precision
    truncated = math.trunc(value * factor) / factor
    if abs(truncated) < 10 ** (-precision):
        truncated = 0.0  # avoid "-0"

    # Always produce exactly precision decimal places
    text = f"{truncated:.{precision}f}"
    return text


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_float_token(token: str) -> float:
    matches = list(_NUMERIC_TOKEN_RE.finditer(token))
    if not matches:
        raise ValueError(f"Bad float token: {token}")
    return float(matches[-1].group(0))


# SMILES tokenizer ---------------------------------------------------------
# Groups:
# 1: bracket atom        (\[[^\]]+\])
# 2: %dd... ring closure (%\d{2,})
# 3: bare atom           ([A-Z][a-z]?)
# 4: aromatic atom       ([cnopsb])
# 5: bond symbols        (=|#|:|\/|\\|-)
# 6: '('                 (\()
# 7: ')'                 (\))
# 8: ring digit          (\d)
# 9: dot                 (\.)
_PERIODIC_TABLE = Chem.GetPeriodicTable()
_ELEMENT_SYMBOLS = {
    _PERIODIC_TABLE.GetElementSymbol(atomic_num)
    for atomic_num in range(1, 119)
}
_TWO_LETTER_SYMBOLS = {sym for sym in _ELEMENT_SYMBOLS if len(sym) == 2}
_THREE_LETTER_SYMBOLS = {sym for sym in _ELEMENT_SYMBOLS if len(sym) == 3}
_AROMATIC_SYMBOLS = set("cnopsb")
_BRACKET_COORD_RE = re.compile(r"(\[[^\]]+\])<[^>]*>")
_COORD_BLOCK_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")
_ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "b", "c", "n", "o", "p", "s"}

def strip_smiles(s: str) -> str:
    """
    Normalize enriched SMILES strings into a 'canonical-ish' comparison form.

    Supported inputs:
      - Legacy enriched strings:  C<...>N<...> (atoms without brackets + coords)
      - Current enriched strings: [C]<...>[N]<...> (atoms wrapped in brackets)
      - Plain SMILES:            C[NH2+]Cc1...

    Steps:
      1. Remove every <...> coordinate block (first the bracketed form, then the legacy bare form).
      2. Collapse decorative carbon H-counts: [CH3],[CH2],[CH],[cH] -> C/c.
      3. Drop brackets around simple atoms: [C]->C, [c]->c, [N]->N, ...
      4. Keep chemically meaningful brackets: [NH2+], [nH], [H], [Pt+2], etc.
    """

    if not s:
        return ""

    s = _WHITESPACE_RE.sub('', s)
    s = _BRACKET_COORD_RE.sub(r"\1", s)
    base_smiles = _COORD_BLOCK_RE.sub('', s)

    # 2) normalize bracket atoms
    def repl(m: re.Match) -> str:
        inner = m.group(1)  # e.g. 'CH3', 'cH', 'N', 'NH2+', 'nH', 'H'

        # Carbon with decorative H-counts: [CH3], [CH2], [CH], [CH0], [cH], [cH1], ...
        if re.fullmatch(r'([Cc])H\d*', inner):
            return inner[0]  # 'C' or 'c'

        # Drop brackets around simple organic-subset atoms (no isotopes/charges/H)
        if (
            inner in _ORGANIC_SUBSET
            and inner != "H"
        ):
            return inner  # drop brackets

        # Everything else: keep bracketed, e.g. [NH2+], [nH], [O-], [H], [Pt+2], [13C]
        return f'[{inner}]'

    return re.sub(r'\[([^\]]+)\]', repl, base_smiles)

def _expected_plain_token(atom) -> str:
    if atom.GetIsAromatic():
        symbol = atom.GetSymbol()
        if symbol == "C":
            return "c"
        if symbol == "N":
            return "n"
        if symbol == "O":
            return "o"
        if symbol == "S":
            return "s"
        if symbol == "P":
            return "p"
        if symbol == "B":
            return "b"
        return symbol.lower()
    return atom.GetSymbol()


def tokenize_smiles(smiles_str, expected_atom_tokens=None):
    """Tokenize a canonical SMILES string into atom/non-atom tokens."""
    tokens = []
    i = 0
    n = len(smiles_str)
    expected_idx = 0
    multi_letter_atoms = {sym for sym in _ELEMENT_SYMBOLS if len(sym) > 1}

    while i < n:
        ch = smiles_str[i]

        if ch == "[":
            end = smiles_str.find("]", i + 1)
            if end == -1:
                raise ValueError(f"Unmatched '[' in SMILES: {smiles_str}")
            tokens.append({"type": "atom", "text": smiles_str[i : end + 1]})
            i = end + 1
            if expected_atom_tokens is not None:
                expected_idx += 1
            continue

        if ch == "%":
            j = i + 1
            while j < n and smiles_str[j].isdigit():
                j += 1
            if j - i <= 2:  # need at least two digits after '%'
                raise ValueError(f"Invalid ring closure token near position {i} in {smiles_str}")
            tokens.append({"type": "nonatom", "text": smiles_str[i:j]})
            i = j
            continue

        if ch in "=#:/\\-":
            tokens.append({"type": "nonatom", "text": ch})
            i += 1
            continue

        if ch in "()":
            tokens.append({"type": "nonatom", "text": ch})
            i += 1
            continue

        if ch == ".":
            tokens.append({"type": "nonatom", "text": ch})
            i += 1
            continue

        if ch.isdigit():
            tokens.append({"type": "nonatom", "text": ch})
            i += 1
            continue

        if ch.isalpha():
            if ch.isupper():
                expected_token = (
                    expected_atom_tokens[expected_idx]
                    if expected_atom_tokens is not None and expected_idx < len(expected_atom_tokens)
                    else None
                )
                symbol = ch
                # Try three-letter, then two-letter element symbols
                for length, symbol_set in ((3, _THREE_LETTER_SYMBOLS), (2, _TWO_LETTER_SYMBOLS)):
                    candidate = smiles_str[i : i + length]
                    tail = candidate[1:]
                    if (
                        len(candidate) == length
                        and tail.isalpha()
                        and tail.islower()
                        and candidate in symbol_set
                        and candidate in multi_letter_atoms
                    ):
                        if expected_token is not None and candidate != expected_token:
                            continue
                        symbol = candidate
                        i += length
                        tokens.append({"type": "atom", "text": symbol})
                        if expected_atom_tokens is not None:
                            expected_idx += 1
                        break
                else:
                    tokens.append({"type": "atom", "text": symbol})
                    i += 1
                    if expected_atom_tokens is not None:
                        expected_idx += 1
                    continue

                continue

            if ch in _AROMATIC_SYMBOLS:
                tokens.append({"type": "atom", "text": ch})
                i += 1
                if expected_atom_tokens is not None:
                    expected_idx += 1
                continue

        raise ValueError(f"Unrecognized SMILES character '{ch}' at position {i} in {smiles_str}")

    return tokens


def _format_atom_descriptor(atom, *, allow_chirality: bool = True):
    """Return a bracketed atom descriptor that preserves valence information."""
    symbol = atom.GetSymbol()
    aromatic = atom.GetIsAromatic()
    if aromatic and len(symbol) == 1:
        symbol_text = symbol.lower()
    else:
        symbol_text = symbol

    descriptor = symbol_text

    chiral = atom.GetChiralTag()
    total_h = atom.GetTotalNumHs()

    if allow_chirality:
        if chiral == ChiralType.CHI_TETRAHEDRAL_CW:
            descriptor += "@"
        elif chiral == ChiralType.CHI_TETRAHEDRAL_CCW:
            descriptor += "@@"

    if (
        allow_chirality
        and not atom.GetIsAromatic()
        and "H" not in descriptor
        and total_h > 0
    ):
        descriptor += "H" if total_h == 1 else f"H{total_h}"

    charge = atom.GetFormalCharge()
    if charge != 0:
        sign = "+" if charge > 0 else "-"
        magnitude = abs(charge)
        descriptor += sign if magnitude == 1 else f"{sign}{magnitude}"

    return f"[{descriptor}]"

_CARBON_DESCRIPTOR_RE = re.compile(r"^\[(?P<iso>\d+)?(?P<elem>[Cc])(?P<tail>.*)\]$")
_CARBON_DECORATIVE_TAIL_RE = re.compile(r"^H\d*$")


def _normalize_atom_descriptor(descriptor: str) -> str:
    """
    Collapse decorative hydrogen counts on neutral carbon descriptors.
    """
    match = _CARBON_DESCRIPTOR_RE.match(descriptor)
    if not match or match.group("iso"):
        return descriptor

    tail = match.group("tail")
    if not tail:
        return descriptor

    if any(ch in tail for ch in "@+-.:/\\"):
        return descriptor

    if _CARBON_DECORATIVE_TAIL_RE.fullmatch(tail):
        return f"[{match.group('elem')}]"

    return descriptor


def encode_cartesian_v2(mol, precision=4):
    """Serialize a 3D RDKit Mol into the enriched text representation."""
    mol_no_h = Chem.RemoveHs(mol)
    if mol_no_h.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer / 3D coordinates.")

    smiles = Chem.MolToSmiles(
        mol_no_h,
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
        allBondsExplicit=False,
    )

    if not mol_no_h.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("Mol is missing _smilesAtomOutputOrder after MolToSmiles.")

    atom_order_raw = mol_no_h.GetProp("_smilesAtomOutputOrder")
    atom_order = list(map(int, ast.literal_eval(atom_order_raw)))

    expected_atom_tokens = [
        _expected_plain_token(mol_no_h.GetAtomWithIdx(idx)) for idx in atom_order
    ]

    tokens = tokenize_smiles(smiles, expected_atom_tokens=expected_atom_tokens)
    out_parts = []
    atom_idx_in_smiles = 0
    conformer = mol_no_h.GetConformer()

    for token in tokens:
        if token["type"] == "atom":
            if atom_idx_in_smiles >= len(atom_order):
                raise ValueError("SMILES atom tokens exceed atom order mapping.")

            rd_idx = atom_order[atom_idx_in_smiles]
            atom_text = token["text"]
            if atom_text.startswith("["):
                atom_descriptor = atom_text
            else:
                atom_descriptor = f"[{atom_text}]"

            pos = conformer.GetAtomPosition(rd_idx)
            coords = (
                truncate(pos.x, precision),
                truncate(pos.y, precision),
                truncate(pos.z, precision),
            )

            out_parts.append(f"{atom_descriptor}<{','.join(coords)}>")
            atom_idx_in_smiles += 1
        else:
            out_parts.append(token["text"])

    if atom_idx_in_smiles != len(atom_order):
        raise ValueError(
            f"Atom count mismatch: mapped {atom_idx_in_smiles} atoms but expected {len(atom_order)}."
        )

    enriched_string = "".join(out_parts)
    return enriched_string, smiles


# Enriched-string tokenizer ------------------------------------------------
_ENRICHED_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\])<([^>]+)>|(%\d{2,})|(=|#|:|\/|\\|-)|(\()|(\))|(\d)|(\.)"
)

def tokenize_enriched(enriched):
    """Tokenize the enriched representation back into atoms (with coords) and other tokens."""
    tokens = []
    pos = 0
    for match in _ENRICHED_TOKEN_PATTERN.finditer(enriched):
        if match.start() != pos:
            raise ValueError(
                f"Unrecognized enriched fragment: {enriched[pos:match.start()]} in {enriched}"
            )

        if match.group(1):
            coord_str = match.group(2)
            parts = [p.strip() for p in coord_str.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Bad coord triplet: {coord_str}")
            coords = tuple(_parse_float_token(p) for p in parts)
            tokens.append(
                {
                    "type": "atom_with_coords",
                    "atom_desc": match.group(1),
                    "coords": coords,
                }
            )
        elif match.group(3):
            tokens.append({"type": "nonatom", "text": match.group(3)})
        elif match.group(4):
            tokens.append({"type": "nonatom", "text": match.group(4)})
        elif match.group(5):
            tokens.append({"type": "nonatom", "text": match.group(5)})
        elif match.group(6):
            tokens.append({"type": "nonatom", "text": match.group(6)})
        elif match.group(7):
            tokens.append({"type": "nonatom", "text": match.group(7)})
        elif match.group(8):
            tokens.append({"type": "nonatom", "text": match.group(8)})

        pos = match.end()

    if pos != len(enriched):
        raise ValueError(f"Unparsed trailing enriched fragment: {enriched[pos:]} in {enriched}")

    return tokens


def decode_cartesian_v2(enriched_string):
    """Reconstruct an RDKit Mol (with conformer) from the enriched string produced by the encoder."""
    tokens = tokenize_enriched(enriched_string)

    smiles_parts = []
    coords = []
    for token in tokens:
        if token["type"] == "atom_with_coords":
            desc = token["atom_desc"]
            desc_inner = desc[1:-1]
            if desc_inner in _ORGANIC_SUBSET:
                smiles_parts.append(desc_inner)
            else:
                smiles_parts.append(desc)
            coords.append(token["coords"])
        else:
            smiles_parts.append(token["text"])

    smiles = "".join(smiles_parts)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to parse rebuilt SMILES: {smiles}")
    if mol.GetNumAtoms() != len(coords):
        raise ValueError(
            f"Atom count mismatch: mol has {mol.GetNumAtoms()} atoms, coords list has {len(coords)} entries."
        )

    Chem.SanitizeMol(mol)

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(idx, Point3D(x, y, z))
    mol.AddConformer(conformer, assignId=True)
    return mol


def embed_3d_conformer_from_smiles(smiles, seed=0):
    """Generate a 3D conformer for a SMILES, drop implicit hydrogens, and return the resulting Mol."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol_h = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol_h, randomSeed=seed)
    if status != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles} (status {status})")

    try:
        mmff_status = AllChem.MMFFOptimizeMolecule(mol_h)
        if mmff_status != 0:
            raise ValueError("MMFF optimization did not converge")
    except Exception:
        uff_status = AllChem.UFFOptimizeMolecule(mol_h)
        if uff_status != 0:
            raise RuntimeError(f"UFF optimization failed for {smiles}")

    mol_no_h = Chem.RemoveHs(mol_h)
    if mol_no_h.GetNumConformers() == 0:
        raise RuntimeError("No conformer present after RemoveHs.")

    Chem.MolToSmiles(
        mol_no_h,
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
        allBondsExplicit=False,
    )
    if mol_no_h.HasProp("_smilesAtomOutputOrder"):
        order = list(map(int, ast.literal_eval(mol_no_h.GetProp("_smilesAtomOutputOrder"))))
        mol_no_h = Chem.RenumberAtoms(mol_no_h, order)

    return mol_no_h


def coords_rmsd(mol_a, mol_b):
    """Compute RMSD between conformer-0 coordinates assuming identical atom order."""
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        raise ValueError("Cannot compare coordinates for molecules with different atom counts.")

    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()
    n = mol_a.GetNumAtoms()
    if n == 0:
        return 0.0

    sse = 0.0
    for idx in range(n):
        pa = conf_a.GetAtomPosition(idx)
        pb = conf_b.GetAtomPosition(idx)
        dx = pa.x - pb.x
        dy = pa.y - pb.y
        dz = pa.z - pb.z
        sse += dx * dx + dy * dy + dz * dz

    rmsd_rdkit = AllChem.GetBestRMS(mol_a, mol_b)
    return min(math.sqrt(sse / n), rmsd_rdkit)

def get_bins_for_coords(ranges, bin_size=0.05):
    """Get bins for coordinates based on the ranges and bin size."""
    bins = []
    for start, end in ranges:
        bins.append(np.arange(start, end, bin_size))
    return bins

def coords_to_bins(coords, bins):
    """Convert coordinates to bins."""
    return np.digitize(coords, bins)


def bins_to_coords(bin_indices, bins, use_bin_center=False):
    """
    Convert bin indices to coordinates by choosing a value within the bin interval for each bin.
    
    Parameters:
        bin_indices (array-like): Indices of the bins.
        bins (array-like): The bin edges as used in np.digitize (e.g., output from np.arange).
        use_bin_center (bool): If True, use the center of the bin; if False, uniformly sample within the bin.
    
    Returns:
        np.ndarray: Coordinates either as bin centers or randomly sampled within each bin.
    """
    coords = []
    for bin_idx in bin_indices:
        # Find the left and right edges of the bin
        if bin_idx <= 0:
            left = bins[0]
            right = bins[1] if len(bins) > 1 else bins[0]
        elif bin_idx >= len(bins):
            left = bins[-1]
            right = bins[-1] + (bins[-1] - bins[-2]) if len(bins) > 1 else bins[-1] + 1.0
        else:
            left = bins[bin_idx - 1]
            right = bins[bin_idx]
        # Choose value: random within [left, right) or use center
        if use_bin_center:
            coord = (left + right) / 2.0
        else:
            coord = np.random.uniform(left, right)
        coords.append(coord)
    return np.array(coords)



def encode_cartesian_binned(mol, bin_size=0.05, ranges=None):
    """
    Serialize a 3D RDKit Mol into an enriched text representation where
    the Cartesian coordinates are replaced by bin indices.

    Returns:
        enriched_string (str): SMILES-like string with [atom]<ix,iy,iz> tokens.
        smiles (str): Canonical SMILES of the heavy-atom molecule.
        bins (list[np.ndarray]): [bins_x, bins_y, bins_z] used for binning.
        ranges (list[tuple[float, float]]): Axis ranges used to construct bins.
    """
    mol_no_h = Chem.RemoveHs(mol)
    if mol_no_h.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer / 3D coordinates.")

    smiles = Chem.MolToSmiles(
        mol_no_h,
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
        allBondsExplicit=False,
    )

    if not mol_no_h.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("Mol is missing _smilesAtomOutputOrder after MolToSmiles.")

    atom_order_raw = mol_no_h.GetProp("_smilesAtomOutputOrder")
    atom_order = list(map(int, ast.literal_eval(atom_order_raw)))

    expected_atom_tokens = [
        _expected_plain_token(mol_no_h.GetAtomWithIdx(idx)) for idx in atom_order
    ]

    tokens = tokenize_smiles(smiles, expected_atom_tokens=expected_atom_tokens)

    # Build bins per axis: use fixed global range [-9, 9] unless explicitly provided.
    if ranges is None:
        ranges = [(-21.0, 21.0), (-21.0, 21.0), (-21.0, 21.0)]
    if len(ranges) != 3:
        raise ValueError("ranges must be a sequence of three (start, end) tuples.")
    bins = get_bins_for_coords(ranges, bin_size=bin_size)
    if len(bins) != 3:
        raise ValueError("get_bins_for_coords must return three bin arrays (x, y, z).")
    # Determine zero-padding width per axis; always at least 3 digits
    digits = [max(3, len(str(len(b)))) for b in bins]

    out_parts = []
    atom_idx_in_smiles = 0
    conformer = mol_no_h.GetConformer()

    for token in tokens:
        if token["type"] == "atom":
            if atom_idx_in_smiles >= len(atom_order):
                raise ValueError("SMILES atom tokens exceed atom order mapping.")

            rd_idx = atom_order[atom_idx_in_smiles]
            atom_text = token["text"]
            if atom_text.startswith("["):
                atom_descriptor = atom_text
            else:
                atom_descriptor = f"[{atom_text}]"

            pos = conformer.GetAtomPosition(rd_idx)

            # Map each coordinate to a bin index (np.digitize-style).
            ix = int(coords_to_bins(np.array([pos.x]), bins[0])[0])
            iy = int(coords_to_bins(np.array([pos.y]), bins[1])[0])
            iz = int(coords_to_bins(np.array([pos.z]), bins[2])[0])

            # Zero-pad indices per axis to a fixed width (>=3)
            ix_txt = f"{ix:0{digits[0]}d}"
            iy_txt = f"{iy:0{digits[1]}d}"
            iz_txt = f"{iz:0{digits[2]}d}"

            out_parts.append(f"{atom_descriptor}<{ix_txt},{iy_txt},{iz_txt}>")
            atom_idx_in_smiles += 1
        else:
            out_parts.append(token["text"])

    if atom_idx_in_smiles != len(atom_order):
        raise ValueError(
            f"Atom count mismatch: mapped {atom_idx_in_smiles} atoms but expected {len(atom_order)}."
        )

    enriched_string = "".join(out_parts)
    return enriched_string, smiles, bins, ranges


def decode_cartesian_binned(enriched_string, bins, use_bin_center=True):
    """
    Reconstruct an RDKit Mol (with conformer) from a binned enriched string.

    The string must have been produced by ``encode_cartesian_binned`` using
    the same set of ``bins`` (one array per axis). Bin indices are turned
    back into coordinates by uniformly sampling within each bin interval.
    """
    if len(bins) != 3:
        raise ValueError("bins must be a sequence of three bin arrays (x, y, z).")

    tokens = tokenize_enriched(enriched_string)

    smiles_parts = []
    coords = []
    for token in tokens:
        if token["type"] == "atom_with_coords":
            desc = token["atom_desc"]
            desc_inner = desc[1:-1]
            if desc_inner in _ORGANIC_SUBSET:
                smiles_parts.append(desc_inner)
            else:
                smiles_parts.append(desc)

            # token["coords"] are floats parsed from the text; interpret them
            # as (possibly float) bin indices and round to nearest int.
            ix_f, iy_f, iz_f = token["coords"]
            ix = int(round(ix_f))
            iy = int(round(iy_f))
            iz = int(round(iz_f))

            x = float(bins_to_coords([ix], bins[0], use_bin_center=use_bin_center)[0])
            y = float(bins_to_coords([iy], bins[1], use_bin_center=use_bin_center)[0])
            z = float(bins_to_coords([iz], bins[2], use_bin_center=use_bin_center)[0])
            coords.append((x, y, z))
        else:
            smiles_parts.append(token["text"])

    smiles = "".join(smiles_parts)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to parse rebuilt SMILES from binned string: {smiles}")
    if mol.GetNumAtoms() != len(coords):
        raise ValueError(
            f"Atom count mismatch: mol has {mol.GetNumAtoms()} atoms, coords list has {len(coords)} entries."
        )

    Chem.SanitizeMol(mol)

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(idx, Point3D(x, y, z))
    mol.AddConformer(conformer, assignId=True)
    return mol
