# Enriched SMILES Representation

This document describes the text-only molecular representation implemented in [`smiles_encoder_decoder.py`](../src/molgen3D/data_processing/smiles_encoder_decoder.py). The format augments canonical isomeric SMILES with per-atom 3D coordinates so language models can copy the topology verbatim and only predict coordinates.

## 1. Objectives

- Preserve full chemical fidelity: stereochemistry, aromaticity, formal charges, isotopes, and explicit hydrogens.
- Keep the SMILES topology unchanged so autoregressive models can copy those tokens directly.
- Store a `<x,y,z>` block (truncated coordinates) immediately after every atom descriptor.
- Prevent unnecessary token fragmentation by keeping atomic metadata inside brackets and avoiding artificial separators.
- Ensure lossless decoding back to the original RDKit molecule plus conformer.

## 2. High-Level Procedure

1. **Hydrogen suppression** – call `Chem.RemoveHs` on the embedded molecule to obtain a hydrogen-suppressed graph. RDKit removes “normal” hydrogen atoms and keeps chemically required explicit ones (e.g., `[nH]`, `[NH2+]`, `[NH3+]`, isotopic `[2H]`, `[3H]`, etc.).
2. **Canonical ordering** – generate the canonical isomeric SMILES *with* `_smilesAtomOutputOrder` so that atom indices remain deterministic.
3. **Atom rewriting** – replace each atom token with a bracketed descriptor that records:
   - Element symbol (lowercase for aromatic atoms such as `c`, `n`, `o`, `p`, `s`, `b`).
   - Tetrahedral chirality (`@`, `@@`).
   - Only explicit hydrogens that remain in the RDKit graph (e.g., `[C@H]`, `[nH]`, `[NH2+]`).
   - Formal charges (`+`, `-`, `+2`, etc.).
   - Isotopic masses when present.
4. **Coordinate embedding** – append a `<x,y,z>` block immediately after every descriptor. Coordinates are truncated (not rounded) to the configured precision (default: 4 decimals).
5. **Topology preservation** – leave all bond symbols, ring indices, branching parentheses, and punctuation untouched. The enriched text is still a valid SMILES string once the `<...>` blocks are stripped.

Example pattern:

```
[C]<1.25,0.03,-0.94>[C]<2.55,0.11,-0.44>(=[O]<3.42,0.85,-1.22>)[N+H2]<1.82,-1.40,0.12>
```

## 3. Atom Descriptor Rules

- **Element & aromaticity** – aromatic atoms use lowercase (`c`, `n`, `o`, `p`, `s`, `b`); all other atoms use the usual atomic symbol (`C`, `N`, `O`, `Cl`, `Br`, etc.).
- **Chirality** – tetrahedral stereochemistry becomes `@` (clockwise) or `@@` (counterclockwise). Example: `C[C@H](O)N` → `[C][C@H]...`.
- **Formal charge** – appended verbatim at the end of the descriptor (`[NH2+]`, `[O-]`, `[P+2]`).
- **Explicit hydrogens only** – replicate only hydrogens that appear as atoms in the RDKit graph:
  - Keep `[nH]`, `[NH2+]`, `[C@H]`, `[2H]`, etc.
  - Do **not** introduce `[CH3]`, `[CH2]`, `[CH]`, or neutral `[cH]` — these hydrogens are implicit and must stay implicit.
- **Isotopes** – isotopic labels remain inside the descriptor (e.g., `[13C@H]`).
- **Normalization** – decorative neutral carbon descriptors like `[CH3]`, `[CH2]`, `[CH]`, `[cH]` collapse back to `[C]` or `[c]` to avoid redundant tokens. This mirrors the “no explicit implicit-H” rule.

## 4. Coordinate Blocks

- Format: `<x,y,z>` directly after the descriptor with no separator.
- Values are truncated to `precision` decimal places (default 4). Truncation avoids rounding-induced drift and eliminates “-0.0000”.
- Example: `[NH2+]<1.2345,-0.9921,0.0043>`.

## 5. Encoder & Decoder Guarantees

- **Encoding** (`encode_cartesian_v2`):
  - Runs `Chem.RemoveHs`, generates the canonical hydrogen-suppressed SMILES (with `_smilesAtomOutputOrder`), tokenizes it, and replaces each atom token by `[descriptor]<x,y,z>` while copying non-atom tokens verbatim.
  - Throws if atom counts diverge or the molecule lacks a conformer.
- **Decoding** (`decode_cartesian_v2`):
  - Tokenizes `[descriptor]<coords>` blocks and SMILES delimiters, rebuilds a bracket-rich SMILES string (e.g., `[C]`, `[N]`, `[nH]`), parses it with RDKit, and assigns the stored coordinates to a new conformer.
  - Raised errors indicate parsing failures or atom-count mismatches.

## 6. Worked Example

Input SMILES: `C[NH2+]Cc1sccc1C`

Enriched output (truncated for brevity):

```
[C]<4.0182,1.0374,0.0007>[NH2+]<3.0038,0.0615,0.4375>[C]<2.0988,-0.4840,-0.6413>[c]<0.6963,-0.5971,-0.1451>1[s]<0.0012,0.5221,0.3002>[c]<...>[c]<...>[c]<...>1[C]<...>
```

Properties:
- No implicit-H descriptors appear.
- Ring indices, branches, and bond symbols are unchanged.
- Decoding this string recovers the original SMILES and the exact conformer coordinates.

## 7. Why This Format

1. **Autoregressive friendliness** – a model can copy the SMILES topology tokens exactly, focusing its predictions on the `<x,y,z>` numbers for each atom.
2. **Hydrogen consistency** – canonical SMILES, enriched SMILES, and the decoded molecule all share the same explicit H atoms.
3. **Lossless round-trip** – topology, stereochemistry, charges, isotopes, and coordinates are perfectly recoverable.
4. **Tokenizer efficiency** – descriptors live inside brackets, which LLM tokenizers already treat as cohesive fragments; coordinates use typical numeric tokens (e.g., `-0.12`, `1.345`).

## 8. Reference Implementation

See `encode_cartesian_v2`, `decode_cartesian_v2`, `strip_smiles`, and related helpers in [`src/molgen3D/data_processing/smiles_encoder_decoder.py`](../src/molgen3D/data_processing/smiles_encoder_decoder.py) for the authoritative logic. These utilities are invoked from preprocessing and evaluation pipelines to ensure every dataset sample conforms to this specification.
