# Conformer interpolation evaluation

Stage 1 asks whether interpolating between FSQ representations of known GEOM
conformers generates *new* conformers that improve coverage of the reference
ensemble. **Stage 1 failed the gate in every configuration, so Stage 2 is not
authorized** — see [Results](#results).

Implementation: `src/molgen3D/evaluation/conformer_interpolation.py`.

## Run

```bash
GEN_JOB=$(sbatch --parsable scripts/run_conformer_interpolation_stage1.sbatch)
sbatch --dependency=afterok:${GEN_JOB} \
  scripts/run_conformer_interpolation_stage1_eval.sbatch

# Smoke run
MAX_MOLECULES=10 PHASE=generate \
OUTPUT_DIR=outputs/conformer_interpolation/stage1_smoke \
sbatch scripts/run_conformer_interpolation_stage1.sbatch
```

Defaults: dataset `/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle`,
the 4096-code (`[2] * 12`) FSQ checkpoint, seed `20260727`, `--pairing-strategy
random`. All overridable via the batch script's environment variables.

The GPU job only generates; the dependent CPU job computes RMSD metrics without
holding an H100. Each molecule is written independently under `molecules/`, so
either phase resumes safely on resubmission.

## Which molecules and pairs get interpolated

Molecules are read in ground-truth-pickle order; `--max-molecules N` truncates
to the first N (smoke runs only). For each molecule with `k` reference
conformers:

1. **Seed.** A per-molecule seed is derived as `sha256(seed \0 molecule_key)`,
   so selection is independent of processing order and of resume boundaries.
2. **Anchors.** `floor(k/2)` references are sampled without replacement. The
   molecule is skipped as `skipped_too_few_anchors` when `floor(k/2) < 2`
   (i.e. `k < 4`). The remaining references are never used as endpoints, but
   metrics still score against the *complete* ensemble.
3. **Encode gate.** All anchors are FSQ-encoded; if their token sequences are
   not identical the molecule is skipped as `token_sequence_mismatch`, since
   bit-level interpolation requires a shared sequence.
4. **Pairs.** Exactly `2k` endpoint pairs (`output_multiplier = 2`), by strategy:
   - `random` — each pair drawn as two distinct anchors; pairs may repeat
     across draws.
   - `balanced_local` — anchors ranked by L2 distance in latent space, then a
     round-robin over neighbor rank (nearest first, cycling through a shuffled
     anchor order) so no dense region consumes the budget before every anchor
     participates. Unordered pairs stay unique until exhausted, then the ranked
     list is cycled; endpoint order is randomized.

## Interpolation operators

All at `alpha = 0.5`, 12-bit codes:

- **latent** — midpoint of the continuous 12-dimensional `pre_q` outputs,
  decoded without re-quantization.
- **token** — MSB-first bit comparison; take endpoint 2 at the 2nd, 4th, …
  differing bit and keep endpoint 1 elsewhere.
- **random_token** — control: per token, take a random `floor(d/2)` or
  `ceil(d/2)` of its `d` differing bits from endpoint 2 (seeded per molecule).

Metrics reuse the existing GEOM COV-R/COV-P/MAT-R/MAT-P implementation and
thresholds, always against the complete reference ensemble. The duplicate rate
is the fraction of decoded conformers matching an earlier output below
`0.001 Å` best RMSD; a distance-matrix signature restricts the expensive
symmetry-aware check to plausible duplicates.

## Outputs

- `config.json` — input, checkpoint, Git commit, source hashes, seed, thresholds.
- `molecules/*.pickle` — anchors, endpoint indices, cached representations,
  decoded conformers, failure counts, per-molecule metrics.
- `per_molecule_metrics.csv` — metric values and change vs. the anchor baseline.
- `aggregate_metrics.pickle` — full threshold curves.
- `summary.json` — aggregates, rates, and the Stage 1 decision.

## Gate

At least one interpolation method must, at the `0.75 Å` threshold, improve
COV-R, lose at most `0.05` COV-P, and reach `0.90` validity and `0.90` decode
success. Stage 2 must not start while `stage_two_authorized` is false.

## Results

Full ground-truth run: 1,000 molecules in, **904 complete**, 58 skipped for too
few anchors, 38 for token-sequence mismatch (tokenizer failure rate 4.0%).
202,708 interpolated conformers per condition. Anchor baseline at `0.75 Å`:
**COV-R 0.850, COV-P 1.000**.

| Pairing | Method | COV-R@0.75 | ΔCOV-R | COV-P@0.75 | ΔCOV-P | Dup. rate | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| random | latent | 0.572 | −0.277 | 0.302 | −0.698 | 6.1% | fail |
| random | random_token | 0.459 | −0.391 | 0.177 | −0.823 | 0.0% | fail |
| random | token | 0.430 | −0.420 | 0.171 | −0.829 | 0.0% | fail |
| balanced_local | latent | 0.733 | −0.117 | 0.556 | −0.444 | 1.1% | fail |
| balanced_local | random_token | 0.666 | −0.183 | 0.378 | −0.622 | 0.0% | fail |
| balanced_local | token | 0.634 | −0.216 | 0.374 | −0.626 | 1.0% | fail |

Runs: `outputs/conformer_interpolation/stage1_ground_truth_four_conditions/`
(random) and `stage1_ground_truth_balanced_local_four_conditions_gpu/`
(balanced-local). `stage1_ground_truth/` is the earlier two-condition run whose
`token` column uses the random-bit rule, and its numbers match `random_token`
above.

Reading:

- **Every method loses coverage.** Interpolated conformers do not reach parts of
  the ensemble the anchors miss; replacing half the anchors with interpolants
  strictly degrades COV-R.
- **The precision collapse is the real signal.** COV-P falls from 1.000 to
  0.17–0.56, so most interpolated conformers are not near any reference — they
  are off-manifold, not novel-but-valid.
- **Latent beats bit interpolation** at both pairings, and locality helps a lot
  (`balanced_local` latent recovers COV-R 0.733 vs. 0.572), but even the best
  configuration is 0.117 COV-R below simply keeping the anchors.
- **Bit interpolation is no better than its random control.** `token` scores at
  or below `random_token` everywhere, so the MSB-first rule carries no
  meaningful geometric ordering.
- Decode success and validity were 1.000 in all conditions, and duplicate rates
  were low — the failure is geometric quality, not mechanics.
