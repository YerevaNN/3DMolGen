## **Molecular Coordinate Encoding**

To represent molecular geometries as text suitable for autoregressive modeling, we design a reversible **Cartesian coordinate–embedded SMILES format**.

The goal is to preserve both chemical topology and atom-level spatial information in a single linearized sequence, allowing the model to copy all chemistry tokens from the input and predict only the numerical coordinates.

---

### **3.2.1 Overview**

Given a molecule M with a valid 3D conformer, we denote its canonical isomeric SMILES by

S = s_1 s_2 \dots s_T,

where each s_i is a token representing either an atom or a non-atom SMILES symbol (bond, ring index, branch, etc.).

We construct an enriched string \tilde{S} by embedding the Cartesian coordinates of every atom token directly into the sequence while keeping the original SMILES syntax unchanged.

---

### **3.2.2 Atom and non-atom tokens**

Each SMILES string is tokenized into two types:

- **Atom tokens**: C, N, O, [nH], [Pt+], etc.
    
    These correspond to atoms present in the molecular graph.
    
- **Non-atom tokens**: bond and structural symbols (=, #, /, \, (, ), ring digits such as 1 or %12, and the dot .).
    
    These define bonding, branching, and connectivity but carry no coordinates.
    

Tokenization ensures that every valid SMILES character sequence can be parsed back without ambiguity.

---

### **3.2.3 Hydrogen handling**

Before encoding, we remove only **implicit hydrogens** using

M’ = \text{RemoveHs}(M),

which drops hydrogens inferable from valence rules.

Explicit or chemically necessary hydrogens—such as the aromatic nitrogen hydrogen in [nH] or the four hydrogens of [NH4+]—are retained.

Hydrogen count annotations on carbon atoms (e.g. [CH3], [cH]) are normalized to [C] or [c], as they do not affect topology or geometry and would unnecessarily enlarge the token vocabulary.

---

### **3.2.4 Canonical ordering and coordinate extraction**

We obtain a **canonical isomeric SMILES** for M’:

S = \text{MolToSmiles}(M’, \text{canonical=True}, \text{isomericSmiles=True}).

RDKit provides a property _smilesAtomOutputOrder that maps each SMILES atom index to its corresponding atom index in the molecule.

Using this mapping, the encoder retrieves the 3D coordinates (x_i, y_i, z_i) from the conformer for every atom appearing in S.

---

### **3.2.5 Embedding coordinates**

Each atom token s_i is replaced by a bracketed **atom descriptor** followed by its coordinates enclosed in angle brackets:

s_i \;\rightarrow\; [A_i]\langle x_i, y_i, z_i\rangle ,

where

- [A_i] is the normalized atomic symbol (e.g. [C], [nH], [Pt+]),
- coordinates are truncated to four decimal places for compactness:
    
    x_i = \text{round}(x_i^{\text{true}}, 10^{-4}).
    
    Non-atom tokens are inserted verbatim between atoms, preserving the exact bonding syntax.
    

The final encoded string is the concatenation of all tokens:

\tilde{S} = t_1 t_2 \dots t_K , \qquad
t_k \in \{ [A_i]\langle x_i, y_i, z_i\rangle,\, \text{non-atom symbol} \}.

---

### **3.2.6 Decoding**

Decoding is fully deterministic:

1. Tokenize \tilde{S} by matching patterns of the form
    
    (\[[^\]]+\])<([^>]+)> for atom-with-coordinate pairs and all remaining SMILES symbols.
    
2. Reconstruct the plain SMILES string by discarding coordinate triplets and concatenating only the atom descriptors and structural tokens.
3. Parse this SMILES with RDKit to obtain a new molecule M’’.
4. Create a conformer for M’’ and assign each atom its stored coordinates (x_i, y_i, z_i).

Because atom order and bonding tokens are preserved exactly, the process is lossless up to the chosen coordinate precision.

---

### **3.2.7 Example**

For the molecule ethanol (CCO) with 3D coordinates truncated to four decimals:

| **Atom** | **Coordinates (Å)** | **Encoded token** |
| --- | --- | --- |
| C₁ | (0.0000, 0.0000, 0.0000) | [C]<0.0000,0.0000,0.0000> |
| C₂ | (1.5274, 0.0000, 0.0000) | [C]<1.5274,0.0000,0.0000> |
| O₃ | (2.0511, 1.4132, 0.0000) | [O]<2.0511,1.4132,0.0000> |

The resulting encoded string is:

\tilde{S} = [C]\langle0.0000,0.0000,0.0000\rangle
[C]\langle1.5274,0.0000,0.0000\rangle
[O]\langle2.0511,1.4132,0.0000\rangle .

When decoded, this string regenerates the same SMILES CCO and restores the atomic coordinates with RMSD < 10⁻³ Å.

---

### **3.2.8 Rationale**

This encoding was chosen to:

- **Preserve chemical topology exactly** by keeping the original SMILES syntax intact.
- **Provide direct textual access to 3D information**, enabling autoregressive models to condition on atomic identities and predict continuous coordinates in-sequence.
- **Ensure reversibility**—the encoded text can be parsed back into a valid molecule with a conformer.
- **Reduce vocabulary complexity** by normalizing redundant hydrogen annotations while keeping chemically essential hydrogens explicit.

This format forms the input–output representation used throughout our molecular coordinate modeling experiments.

## Handling Hydrogens in smiles

Here’s a concise summary of which **hydrogens actually affect molecular geometry** — i.e. must be **kept as explicit atoms** with coordinates — and which can safely be dropped or inferred.

---

### **Keep (explicit hydrogens that affect geometry)**

| **Case** | **Why it matters** | **Example SMILES** | **Explanation** |
| --- | --- | --- | --- |
| **Aromatic N–H in heterocycles** | Defines aromaticity & ring planarity; dropping it changes electron count and geometry. | c1ncc[nH]1 (imidazole) | The [nH] must stay; it contributes to aromatic delocalization. |
| **Protonated amines / ammonium** | The N–H hydrogens determine tetrahedral geometry; required for charge balance. | [NH4+] → [N+]([H])([H])([H])[H] | These [H] atoms are real and change the shape (sp³ vs. sp²). |
| **Hydrogens on charged or polar atoms** | Affect hydrogen bonding & orientation; removing them distorts geometry. | [NH3+], [OH2], [SH] | Their positions influence dipole and local 3D structure. |
| **Bridging or special H (e.g. metal hydrides)** | Chemically bonded to metals or bridging atoms — unique geometry. | [PtH2], [BH4-] | Must be kept; they are not implicit. |
| **Deuterium / tritium labels (D, T)** | Explicit isotopes have different mass and slightly different bond lengths. | [2H]C([2H])([H])[2H] | Keep for correct mass/geometry. |

---

### **Drop (implicit or redundant hydrogens)**

| **Case** | **Why it can be dropped** | **Example SMILES** | **Explanation** |
| --- | --- | --- | --- |
| **Standard C–H bonds** | Geometry is fixed by valence (sp²/sp³); RDKit can infer positions. | CC=O, c1ccccc1 | [CH3] or [cH] can safely be treated as [C] or [c]. |
| **Hydrogens implied by valence on O, N, S (neutral)** | Single bonds to O/N/S automatically imply 1–2 hydrogens. | OCC, CCN | Implicit H positions are predictable and not unique. |
| **Terminal methyls or methylenes** | Fully determined by carbon hybridization; no structural ambiguity. | [CH3]CH2OH | [CH3] is just [C] with implicit Hs. |

---

### **In short**

Keep hydrogens that:

- define **aromaticity**, **charge**, or **hybridization**,
- **cannot be re-inferred** from valence,
- or **participate in hydrogen bonding** geometry (e.g., [nH], [NH3+], [OH2]).

Drop hydrogens that:

- are **implied by valence** and **don’t carry charge or special roles**, e.g., [cH], [CH2], [CH3].

---

**Rule of thumb for your encoder:**

Keep whatever survives Chem.RemoveHs(mol) — those are the hydrogens that genuinely affect 3D geometry.