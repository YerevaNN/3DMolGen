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
3. **Atom mirroring** – treat the hydrogen-suppressed canonical SMILES as the *sole* source of truth for atom descriptors.
   - If RDKit already emitted a bracket token (e.g., `[C@@H]`, `[nH]`, `[NH2+]`, `[13CH3]`), reuse it verbatim.
   - If the token is a bare atom (`C`, `c`, `N`, `Cl`, …), simply wrap it as `[C]`, `[c]`, `[N]`, `[Cl]`, etc.
   - **Never** invent `@/@@`, `H/Hn`, charges, or isotopes that RDKit did not include.
4. **Coordinate embedding** – append a `<x,y,z>` block immediately after every descriptor. Coordinates are truncated (not rounded) to the configured precision (default: 4 decimals).
5. **Topology preservation** – leave all bond symbols, ring indices, branching parentheses, and punctuation untouched. The enriched text is still a valid SMILES string once the `<...>` blocks are stripped.

Example pattern:

```
[C]<1.25,0.03,-0.94>[C]<2.55,0.11,-0.44>(=[O]<3.42,0.85,-1.22>)[N+H2]<1.82,-1.40,0.12>
```

## 3. Atom Descriptor Rules

- **Source of truth** – descriptors are copied directly from `Chem.MolToSmiles(RemoveHs(mol), isomericSmiles=True)`. The encoder never inspects `atom.GetChiralTag()`, `GetTotalNumHs()`, etc. to “fix” tokens.
- **Bare atoms** – only the minimal wrapping occurs: `C` → `[C]`, `c` → `[c]`, `Cl` → `[Cl]`. These tokens contain *no* `@`, `H`, charge, or isotope suffixes.
- **Decorated atoms** – RDKit-controlled bracket tokens (e.g., `[C@@H]`, `[nH]`, `[NH2+]`, `[13CH3]`, `[Pt+2]`) are preserved byte-for-byte.
- **Implicit-H chirality** – if RDKit removed a stereocenter because its configuration depended solely on an implicit hydrogen, the canonical SMILES will not contain `@/@@`, and therefore neither will the enriched string.
- **Explicit hydrogens** – only explicit hydrogens that survive `RemoveHs` (such as `[nH]`, `[NH2+]`, isotopic `[2H]`) appear, exactly as RDKit printed them.

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
