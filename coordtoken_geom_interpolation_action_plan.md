# Action Plan: Conformer Interpolation Evaluation

## Goal

First verify that interpolation itself produces useful conformers when the endpoints are known ground-truth conformers. Only if this sanity check succeeds, run the previously planned interpolation experiment using model-generated conformers.

## Stage 1: Ground-Truth Interpolation Sanity Check

For each molecule in the GEOM test set, let `k` be the number of available reference conformers.

### 1. Select anchor conformers

- Randomly select `k/2` ground-truth conformers.
- Use a fixed random seed.
- If `k` is odd, use `floor(k/2)`.
- Require at least two selected conformers; otherwise skip the molecule and log it.

### 2. Sample interpolation pairs

- Randomly sample pairs from the selected ground-truth conformers.
- Do not pair a conformer with itself.
- Sampling with replacement is allowed so that enough outputs can be generated.
- Generate exactly `2k` interpolated conformers per molecule.
- Save the sampled endpoint indices and the random seed.

Use interpolation coefficient:

\[
\alpha = 0.5
\]

for all pairs.

### 3. Latent-space interpolation

For every sampled pair:

1. Encode both ground-truth conformers with the existing tokenizer.
2. Obtain their continuous latent representations.
3. Compute the midpoint latent:
   \[
   z_{\text{mid}}=0.5z_1+0.5z_2
   \]
4. Decode the midpoint latent into a conformer.

This produces `2k` latent-interpolated conformers.

### 4. Token-space interpolation

For every sampled pair, create both token-space variants.

#### Deterministic token condition

1. Tokenize both ground-truth conformers.
2. Match tokens by sequence position.
3. Convert each coordinate token into its 12-bit binary representation.
4. Identify the bit positions where the two endpoint tokens differ.
5. At every second differing bit position, take the bit from the second endpoint.
6. At the remaining differing positions, keep the bit from the first endpoint.
7. Keep identical bits unchanged and decode the resulting token sequence.

#### Random token condition

Repeat steps 1–4, then:

5. Randomly select approximately half of the differing positions and take those
   bits from the second endpoint.
6. At the remaining differing positions, keep the bit from the first endpoint.
   For an odd number of differing bits, randomly choose between `floor(d/2)` and
   `ceil(d/2)` bits from the second endpoint.
7. Keep identical bits unchanged.
8. Convert the resulting 12-bit code back to a token.
9. Decode the complete interpolated token sequence.

Retain the original deterministic every-second-differing-bit method as the
`token` condition. The randomized balanced-bit method is an additional
`random_token` condition. Each produces `2k` conformers from the same sampled
endpoint pairs.

Use a fixed, saved random seed. Repeated occurrences of the same endpoint pair
must draw independently so they can produce different token-space
interpolations while remaining reproducible.

Do not:

- average token IDs;
- interpolate token embeddings;
- replace whole tokens;
- insert or delete tokens.

### 5. Evaluate the sanity check

For each molecule, evaluate four sets with the existing GEOM evaluation pipeline:

1. the selected `k/2` ground-truth anchors;
2. `2k` latent-interpolated conformers;
3. `2k` deterministic token-interpolated conformers;
4. `2k` randomized token-interpolated conformers.

Report exactly the metrics already used by the current GEOM benchmark code.

The main question is:

> Do interpolated conformers improve coverage of the full reference conformer set beyond the selected `k/2` ground-truth anchors?

Use the same full ground-truth conformer set as the evaluation target for all
four conditions.

Also report:

- interpolation validity rate;
- tokenizer/decode failure rate;
- duplicate conformer rate;
- per-molecule metric changes relative to the `k/2`-anchor baseline.

### 6. Success criterion

Proceed to Stage 2 only if at least one interpolation method:

- improves the main GEOM coverage metric over the `k/2` ground-truth-anchor baseline;
- does not cause a major degradation in precision;
- has an acceptable validity and decoding-success rate.

Do not choose the better method using isolated visual examples; use the aggregate benchmark results.

## Stage 2: Interpolation Between Generated Conformers

Run this stage only after Stage 1 confirms that interpolation is meaningful.

For each molecule:

1. Generate `k` base conformers using the existing conformer-generation model.
2. Compute pairwise symmetry-aware RMSD among the generated conformers.
3. Order them into an approximate minimum-length closed cycle using nearest-neighbor ordering and optional 2-opt refinement.
4. Interpolate once along every neighboring edge, including the last-to-first edge.
5. Produce:
   - baseline: `k` generated conformers;
   - latent interpolation: `k` generated + `k` interpolated = `2k`;
   - token interpolation: `k` generated + `k` interpolated = `2k`.
6. Evaluate all three sets with the existing GEOM benchmark pipeline.

The same generated anchors and the same cycle must be reused for latent-space and token-space interpolation.

## Efficiency and Reproducibility

- Inspect the current codebase first and reuse existing generation, tokenizer, decoding, and evaluation functions.
- Generate or tokenize each conformer only once.
- Cache latent and token representations.
- Save results incrementally and make the pipeline resumable.
- Store random seeds and sampled pairs.
- Avoid unnecessary SDF files; save only what is needed for debugging or final visualization.
- Allow metric evaluation to be rerun without repeating interpolation inference.

## Outputs

Save:

- selected ground-truth anchor indices;
- sampled interpolation pairs;
- latent-interpolated conformers;
- token-interpolated conformers;
- per-molecule metrics;
- aggregate metrics;
- validity, duplication, and failure statistics;
- Stage 1 pass/fail decision;
- Stage 2 results if Stage 1 succeeds;
- configuration, checkpoint, Git commit, and random seeds.
