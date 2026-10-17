import re, ast
from rdkit import Chem
import numpy as np

# ATOM_TOKENS = ['[S@TB20]', '[S@TB19]', '[S@TB18]', '[S@TB17]', '[S@TB16]', '[S@TB15]', '[S@TB14]', '[S@TB13]', '[S@TB12]', '[S@TB11]', '[S@TB10]', '[S@OH29]', '[S@OH28]', '[S@OH27]', '[S@OH26]', '[S@OH24]', '[S@OH23]', '[S@OH22]', '[S@OH21]', '[S@OH20]', '[S@OH18]', '[S@OH17]', '[S@OH16]', '[S@OH15]', '[S@OH12]', '[S@OH11]', '[P@TB20]', '[P@TB19]', '[P@TB18]', '[P@TB17]', '[P@TB16]', '[P@TB15]', '[P@TB14]', '[P@TB13]', '[P@TB12]', '[P@TB11]', '[P@TB10]', '[P@OH29]', '[Si@@H]', '[S@TB9]', '[S@TB6]', '[S@TB5]', '[S@TB4]', '[S@TB3]', '[S@TB2]', '[S@TB1]', '[S@SP3]', '[S@SP2]', '[S@SP1]', '[S@OH6]', '[S@OH5]', '[S@OH3]', '[P@TB9]', '[P@TB8]', '[P@TB7]', '[P@TB6]', '[P@TB5]', '[P@TB4]', '[P@TB3]', '[P@TB2]', '[P@TB1]', '[P@SP2]', '[P@@H2]', '[Ge@@H]', '[SiH4]', '[SiH3]', '[SiH2]', '[Si@H]', '[Si@@]', '[P@H3]', '[P@H2]', '[P@@H]', '[GeH3]', '[GeH2]', '[Ge@@]', '[GaH3]', '[C@@H]', '[siH]', '[SiH]', '[Si@]', '[SeH]', '[S@@]', '[PH2]', '[P@H]', '[P@@]', '[NH2]', '[GeH]', '[Ge@]', '[CH3]', '[CH2]', '[C@H]', '[C@@]', '[AsH]', '[si]', '[se]', '[pH]', '[nH]', '[n+]', '[Zn]', '[Ti]', '[Si]', '[Se]', '[SH]', '[S@]', '[PH]', '[P@]', '[OH]', '[O-]', '[NH]', '[N+]', '[Mg]', '[He]', '[Ge]', '[Cl]', '[Ca]', '[CH]', '[C@]', '[Br]', '[Be]', '[BH]', '[As]', '[Ar]', '[c]', '[S]', '[P]', '[O]', '[N]', '[H]', '[F]', '[C]', '[B]', 'Zn', 'Ti', 'Si', 'Se', 'Mg', 'He', 'Ge', 'Ga', 'Cl', 'Ca', 'Br', 'Be', 'As', 'Ar', 's', 'r', 'p', 'o', 'n', 'l', 'i', 'g', 'e', 'c', 'b', 'a', 'Z', 'T', 'S', 'P', 'O', 'N', 'M', 'H', 'G', 'F', 'C', 'B', 'A']
ATOM_TOKENS = ['[nH]', 'Br', 'Cl', 'C', 'F', 'N', 'O', 'S', 'c', 'n', 'o', 's']
ATOM_RE = "|".join(re.escape(tok) for tok in ATOM_TOKENS)
ATOM_SET = set(ATOM_TOKENS)
pattern_atom = re.compile(rf"({ATOM_RE}|.)")

OTHER_TOKENS = ['#', '(', ')', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '[', ']']

VOCAB = ATOM_TOKENS + OTHER_TOKENS
sorted_vocab = sorted(VOCAB, key=len, reverse=True)
vocab_pat = "|".join(re.escape(tok) for tok in sorted_vocab)
pattern = re.compile(rf"({vocab_pat})(?:<([^>]+)>)?")
token_to_idx = {tok: i for i, tok in enumerate(VOCAB)}

# with open('logs/tokenization_errors.txt', 'a') as f:
#     print('starting...', file=f)


def truncate(x, precision=4):
    s = repr(x)
    if '.' in s:
        intp, frac = s.split('.', 1)
        frac = (frac + '0' * (precision - 1))[:precision]   # frac = (frac + )[:precision]
        return f"{intp}.{frac}"
    else:
        return s  # integer

def get_embedded_smiles(mol, precision=4):
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    atom_order = list(map(int, ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))))
    pos = mol.GetConformer().GetPositions()

    tokens = pattern_atom.findall(smiles)
    out, idx = [], 0
    for tok in tokens:
        if tok in ATOM_TOKENS:
            ai = atom_order[idx]
            x, y, z = map(float, pos[ai])
            out.append(f"{tok}<{truncate(x,precision)},{truncate(y,precision)},{truncate(z,precision)}>")
            idx += 1
        else:
            out.append(tok)
    return "".join(out)

def tokenize_and_encode(embedded_smiles):
    """ Returns feature_vectors: numpy array of shape (n_tokens, vocab_size+3) """
    token_indices, coords = [], []
    for m in pattern.finditer(embedded_smiles):
        tok, coord = m.group(1), m.group(2)
        token_indices.append(token_to_idx[tok])
        if coord:
            x, y, z = map(float, coord.split(','))
        else:
            x, y, z = 0., 0., 0.
        coords.append((x, y, z))

    # Check for unknown tokens - remove all known patterns, anything left is unknown
    # unmatched = pattern.sub('', embedded_smiles)
    # if unmatched:
    #     with open('logs/tokenization_errors.txt', 'a') as f:
    #         print(f'Unknown sequence: {unmatched} in {embedded_smiles}', file=f)

    n, V = len(token_indices), len(VOCAB)
    feats = np.zeros((n, V + 3), dtype=np.float32)
    for i, (idx, (x, y, z)) in enumerate(zip(token_indices, coords)):
        feats[i, idx] = 1.0
        feats[i, V:]  = (x, y, z)
    return feats
