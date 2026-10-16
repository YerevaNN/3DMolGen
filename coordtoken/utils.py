import ast
import math
import os
import re
import sys
import time
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Geometry import Point3D
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from tqdm.auto import tqdm

VOCAB = [
    '#', '%10', '%11', '%12', '%13', '%14', '%15', '%16', '%17', '%18', '%19', '%20', '(', ')', 
    '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '[Ag]', '[Al]', '[AsH2]', 
    '[AsH3]', '[As]', '[B-]', '[B@@-]', '[BH-]', '[BH2-]', '[BH]', '[B]', '[Ba]', '[Bi]', '[Br+]', 
    '[Br]', '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH+]', '[CH-]', '[CH2+]', 
    '[CH2-]', '[CH2]', '[CH3-]', '[CH3]', '[CH]', '[C]', '[CaH]', '[Ca]', '[Cl-]', '[Cl]', '[Cs]', 
    '[Cu@SP]', '[F-]', '[F]', '[Ga]', '[H+]', '[H]', '[I+2]', '[I+]', '[I-]', '[IH]', '[I]', '[K]', 
    '[Li]', '[MgH]', '[Mg]', '[Mn@SP]', '[Mn]', '[N+]', '[N-2]', '[N-]', '[N@+]', '[N@@+]', 
    '[N@@H+]', '[N@H+]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[NH4+]', '[NH]', '[N]', '[Na]', 
    '[O+]', '[O-]', '[OH+]', '[OH2+]', '[OH]', '[O]', '[P+]', '[P-2]', '[P-]', '[P@+]', '[P@@+]', 
    '[P@@H+]', '[P@@H2+]', '[P@@H2]', '[P@@H3+]', '[P@@H]', '[P@@]', '[P@H+]', '[P@H2+]', '[P@H2]', 
    '[P@H3+]', '[P@H]', '[P@OH]', '[P@TB]', '[P@]', '[PH+]', '[PH2+]', '[PH2]', '[PH]', '[P]', 
    '[Pb]', '[Rb]', '[S+2]', '[S+]', '[S-]', '[S@+]', '[S@@+]', '[S@@H]', '[S@@]', '[S@H]', 
    '[S@OH]', '[S@SP]', '[S@TB]', '[S@]', '[SH+3]', '[SH+]', '[SH2+]', '[SH2]', '[SH3+2]', '[SH3]', 
    '[SH4]', '[SH]', '[S]', '[Se-]', '[SeH]', '[Se]', '[Si+2]', '[Si+3]', '[Si+]', '[Si-]', 
    '[Si@@H]', '[Si@@]', '[Si@H]', '[Si@]', '[SiH-]', '[SiH2]', '[SiH3]', '[SiH4]', '[SiH]', 
    '[Si]', '[Sr]', '[Zn]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[c]', '[n+]', '[n-]', '[nH+]', 
    '[nH]', '[n]', '[o+]', '[oH+]', '[o]', '[p+]', '[p-]', '[pH]', '[p]', '[s+2]', '[s+]', '[sH+]', 
    '[s]', '[se]', '[si+]', '[si-]', '[si]', '\\', 
]

TOKEN_TO_IDX = {tok: i for i, tok in enumerate(VOCAB)}
V = len(VOCAB) #192

def truncate(x, precision=4):
    """Format a float with at most ``precision`` decimal places (truncation, not rounding)."""
    if precision < 0:
        raise ValueError("precision must be non-negative")

    value = float(x)
    if precision == 0:
        return str(int(math.trunc(value)))

    factor = 10 ** precision
    truncated = math.trunc(value * factor) / factor
    if abs(truncated) < 10 ** (-precision):
        truncated = 0.0  # avoid "-0"

    text = f"{truncated:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def _parse_float_token(token: str) -> float:
    matches = list(_NUMERIC_TOKEN_RE.finditer(token))
    if not matches:
        raise ValueError(f"Bad float token: {token}")
    return float(matches[-1].group(0))


# SMILES tokenizer ---------------------------------------------------------
# Groups:
# 1: bracket atom        (\[[^\]]+\])
# 2: %dd ring closure     (%\d{2})
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
        # if symbol == "C":
        #     return "c"
        # if symbol == "N":
        #     return "n"
        # if symbol == "O":
        #     return "o"
        # if symbol == "S":
        #     return "s"
        # if symbol == "P":
        #     return "p"
        # if symbol == "B":
        #     return "b"
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
            if i + 2 < n and smiles_str[i+1:i+3].isdigit():
                tokens.append({"type": "nonatom", "text": smiles_str[i:i+3]})
                i += 3
                continue
            else:
                raise ValueError(f"Invalid ring closure token near position {i} in {smiles_str}")

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
    return enriched_string


def smiles_coords_to_enriched(smiles: str, coords, precision: int = 4) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    n_atoms = mol.GetNumAtoms()
    if coords.shape[0] != n_atoms:
        raise ValueError(f"Coords mismatch: {coords.shape[0]} rows vs {n_atoms} atoms")
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        x, y, z = map(float, coords[i])
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
    return encode_cartesian_v2(mol, precision=precision)


# Enriched-string tokenizer ------------------------------------------------
_ENRICHED_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\])<([^>]+)>|(%\d{2})|(=|#|:|\/|\\|-)|(\()|(\))|(\d)|(\.)"
)

def build_fsq_string(enriched_text, codes): # token<code>token<code>...
    tokens = tokenize_enriched(enriched_text)
    
    new_parts = []
    if len(tokens) != len(codes):
        raise ValueError(f"Mismatch: {len(tokens)} != {len(codes)}")
        
    for t_idx, token in enumerate(tokens):
        code_val = codes[t_idx]
        if token["type"] == "atom_with_coords":
            desc = token["atom_desc"]
            new_parts.append(f"{desc}<{code_val}>")
        else:
            new_parts.append(f"{token['text']}<{code_val}>")
            
    return "".join(new_parts)

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
    # Complete text-specified stereo before attaching geometry.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

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
    return math.sqrt(sse / n)


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


def tokenize_and_encode(enriched_smiles: str) -> tuple[np.ndarray, np.ndarray]:
    pos = 0
    n = 0
    for match in _ENRICHED_TOKEN_PATTERN.finditer(enriched_smiles):
        if match.start() != pos:
            raise ValueError(
                f"Unrecognized enriched fragment: {enriched_smiles[pos:match.start()]} in {enriched_smiles}"
            )
        pos = match.end()
        n += 1

    if pos != len(enriched_smiles):
        raise ValueError(
            f"Unparsed trailing enriched fragment: {enriched_smiles[pos:]} in {enriched_smiles}"
        )

    feats = np.zeros((n, V + 3), dtype=np.float32)
    tok_ids = np.empty(n, dtype=np.int64)
    token_to_idx = TOKEN_TO_IDX
    v = V
    pos = 0
    i = 0

    for match in _ENRICHED_TOKEN_PATTERN.finditer(enriched_smiles):
        if match.start() != pos:
            raise ValueError(
                f"Unrecognized enriched fragment: {enriched_smiles[pos:match.start()]} in {enriched_smiles}"
            )

        atom_desc = match.group(1)
        if atom_desc is not None:
            coord_str = match.group(2)
            c0, c1, c2 = [p.strip() for p in coord_str.split(",")]
            idx = token_to_idx[atom_desc]
            tok_ids[i] = idx
            feats[i, idx] = 1.0
            feats[i, v] = float(c0)
            feats[i, v + 1] = float(c1)
            feats[i, v + 2] = float(c2)
        else:
            text = (
                match.group(3)
                or match.group(4)
                or match.group(5)
                or match.group(6)
                or match.group(7)
                or match.group(8)
            )
            idx = token_to_idx[text]
            tok_ids[i] = idx
            feats[i, idx] = 1.0

        pos = match.end()
        i += 1

    return feats, tok_ids


_FSQ_PAIR_RE = re.compile(r"(.*?)<(\d+)>")

def parse_fsq_text(fsq_text: str) -> tuple[list[str], np.ndarray]:
    pairs = _FSQ_PAIR_RE.findall(fsq_text)
    if not pairs:
        raise ValueError("FSQ text parsing failed: no token<code> pairs found.")
    tokens = [t for (t, _) in pairs]
    codes = np.array([int(c) for (_, c) in pairs], dtype=np.int64)
    return tokens, codes

def tokens_to_vocab_onehot(tokens: list[str]) -> np.ndarray:
    """
    Rebuild the vocab one-hot slice [T, V] from FSQ token strings.

    This matches tokenize_and_encode(), because build_fsq_string() stores:
    - atom positions as token['atom_desc']
    - non-atom positions as token['text']

    So each token string in fsq_text is already the exact TOKEN_TO_IDX key.
    """
    n = len(tokens)
    x = np.zeros((n, V), dtype=np.float32)
    for i, tok in enumerate(tokens):
        idx = TOKEN_TO_IDX[tok]
        x[i, idx] = 1.0
    return x

def format_enriched_from_tokens_and_coords(tokens: list[str], coords: np.ndarray, precision: int = 4) -> str:
    """
    Build enriched text in the same style as encode_cartesian_v2:
    - atom tokens: [AtomDesc]<x,y,z>
    - non-atom tokens: raw token text

    Assumes atom descriptors are bracketed (e.g. [C], [NH+], ...),
    which matches the strings used by tokenize_and_encode/build_fsq_string.
    """
    out_parts = []
    for tok, xyz in zip(tokens, coords):
        if tok.startswith("["):
            x = truncate(float(xyz[0]), precision)
            y = truncate(float(xyz[1]), precision)
            z = truncate(float(xyz[2]), precision)
            out_parts.append(f"{tok}<{x},{y},{z}>")
        else:
            out_parts.append(tok)
    return "".join(out_parts)


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, manual_total):
        super().__init__(refresh_rate=50)
        self.manual_total = manual_total
        self._last_update_time = 0
        self._last_batch_idx = -1

    def init_train_tqdm(self):
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
            mininterval=30,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.train_progress_bar is not None:
            self.train_progress_bar.total = self.manual_total
            self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
        self._last_update_time = time.perf_counter()
        self._last_batch_idx = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        is_last_batch = (batch_idx + 1 == trainer.num_training_batches)
        should_update = (now - self._last_update_time >= 30) or is_last_batch
        
        if should_update and self.train_progress_bar is not None:
            batches_since_update = batch_idx - self._last_batch_idx
            if batches_since_update > 0:
                self.train_progress_bar.update(batches_since_update)
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)
            self._last_update_time = now
            self._last_batch_idx = batch_idx


class MFUCallback(L.Callback):
    def __init__(self, max_tokens, peak_tflops_per_gpu=989.0, print_every_s=30.0, train_flop_factor=3.0):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.peak = float(peak_tflops_per_gpu)
        self.print_every_s = float(print_every_s)
        self.train_flop_factor = float(train_flop_factor)

        self.last_t = None
        self.tok_local = 0
        self.loss_sum_local = None
        self.loss_count = 0

        self.flops_per_tok = None  # precomputed
        self.peak_source = "explicit"

    # _infer_peak_tflops_per_gpu is retained for reference but not used by default anymore
    # because peak_tflops_per_gpu is now fixed to 989.0 by default in __init__.
    #@staticmethod
    #def _infer_peak_tflops_per_gpu(device) -> tuple[float, str]:
    #    env_peak = os.environ.get("MFU_PEAK_TFLOPS")
    #    if env_peak:
    #        return float(env_peak), "env"
    #
    #    name = torch.cuda.get_device_name(device).lower()
    #    if "h100" in name:
    #        return 989.0, torch.cuda.get_device_name(device)
    #    if "a100" in name:
    #        return 312.0, torch.cuda.get_device_name(device)
    #    if "v100" in name:
    #        return 125.0, torch.cuda.get_device_name(device)
    #
    #    raise RuntimeError(
    #        f"Unknown GPU '{torch.cuda.get_device_name(device)}' for MFU peak auto-detection. "
    #        "Set MFU_PEAK_TFLOPS in the environment or pass peak_tflops_per_gpu explicitly."
    #    )

    @staticmethod
    def _transformer_forward_flops_per_token(d_model: int, n_layers: int, T: int, stacks: int = 2) -> float:
        """
        Forward FLOPs/token for TransformerEncoderLayer-like blocks.
        - stacks=2 for your enc+dec.
        - per layer per token:
            * qkv + out projections: ~8 d^2
            * MLP d->4d->d: ~16 d^2
            * attention score/value matmuls: ~4 d T
          => ~ (24 d^2 + 4 d T)
        FLOPs count multiply+add as two floating-point ops.
        """
        d = float(d_model)
        L = float(n_layers)
        T = float(T)
        per_layer = (24.0 * d * d) + (4.0 * d * T)
        return float(stacks) * L * per_layer

    @staticmethod
    def _dense_addons_forward_flops_per_token(vocab_size: int, d_model: int, levels_len: int) -> float:
        """
        Count the non-transformer dense projections in the forward path:
        - tok_emb: (V+3)->d
        - pre_q: d->64->levels_len
        - post_q: (V+levels_len)->256->d
        - out: d->3
        """
        V = float(vocab_size)
        d = float(d_model)
        Lc = float(levels_len)

        # linear FLOPs proxy: ~2*m*n per token (mul+add)
        tok_emb = 2.0 * (V + 3.0) * d
        pre_q = 2.0 * d * 64.0 + 2.0 * 64.0 * Lc
        post_q = 2.0 * (V + Lc) * 256.0 + 2.0 * 256.0 * d
        out = 2.0 * d * 3.0
        return tok_emb + pre_q + post_q + out

    def on_fit_start(self, trainer, pl_module):
        # self.peak is always set from __init__ (default 989.0), so no auto-detect path needed.
        self.peak_source = "explicit"
        h = pl_module.hparams
        T = self.max_tokens  # your batch tensor is fixed-length [B,MAX_TOKENS,...]
        tf = self._transformer_forward_flops_per_token(h.d_model, h.n_layers, T, stacks=2)
        vocab_size = int(pl_module.tok_emb.in_features - 3)
        addons = self._dense_addons_forward_flops_per_token(vocab_size, h.d_model, levels_len=len(h.levels))
        self.flops_per_tok = self.train_flop_factor * (tf + addons)

        if trainer.is_global_zero:
            tqdm.write(
                f"[MFU~] peakTFLOPs/GPU={self.peak:,.1f} ({self.peak_source}) | "
                f"train_FLOPs/token≈{self.flops_per_tok:,.0f}"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # local loss window (rank0 print only)
        loss = outputs.detach() if torch.is_tensor(outputs) else outputs["loss"].detach()
        self.loss_sum_local = loss.clone() if self.loss_sum_local is None else (self.loss_sum_local + loss)
        self.loss_count += 1

        # local token count
        x = batch[0]
        B, T = x.shape[:2]
        self.tok_local += int(B * T)

        now = time.perf_counter()
        if self.last_t is None:
            self.last_t = now
            return

        dt = now - self.last_t
        if dt < self.print_every_s:
            return

        # local-only throughput (no collectives)
        tok_per_s_local = self.tok_local / max(dt, 1e-9)

        # MFU proxy using local rate (rank0 only print)
        tflops_local = (tok_per_s_local * float(self.flops_per_tok)) / 1e12
        mfu_local = (tflops_local / self.peak)

        if trainer.is_global_zero:
            avg_loss_rank0 = float((self.loss_sum_local / max(1, self.loss_count)).item())
            tqdm.write(
                f"[MFU~] tok/s(local rank0)={tok_per_s_local:,.0f} | "
                f"TFLOPs(rank0)≈{tflops_local:,.1f} | MFU≈{mfu_local:.2%} | "
                f"loss(rank0_window)={avg_loss_rank0:.4f} | "
                f"VRAM(max)={torch.cuda.max_memory_reserved()/1024**3:.1f}GB"
            )

        # reset window
        self.last_t = now
        self.tok_local = 0
        self.loss_sum_local = None
        self.loss_count = 0
