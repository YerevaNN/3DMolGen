# Bioactive Conformer Benchmark Analysis

## Short Thesis

The current CASF16 intersection runs support a publishable evaluation story, but the cleanest claim is not yet "geometrically diverse training data is better." The defensible current claim is:

> Energy-minimized conformer references are highly sample-efficient for near-native conformers, but they underrepresent the high-diversity tail needed to recover bound conformations for flexible ligands. Quality-filtered geometric diversity improves best-case bioactive-pose recovery, especially at high K and high rotatable-bond count, but naive diversity also wastes samples and can introduce high-energy artifacts.

This is a strong benchmark story because it separates three things that are often conflated:

- low-energy conformer plausibility
- ensemble diversity
- recovery of experimentally observed protein-bound ligand conformations

The current data already show that these objectives are not the same.

## Inputs Reviewed

Primary local artifacts:

- 94-ligand core intersection report: `/mnt/weka/mbedrosian/conformer_sets_core_chembl3d_exact_intersection_pb_full_dynamic_chembl_count/analysis/geometric_report.md`
- 1219-ligand reference intersection report: `/mnt/weka/mbedrosian/conformer_sets_ref_chembl3d_exact_intersection_pb_full_dynamic_chembl_count/analysis/geometric_report.md`
- Per-ligand metric tables under each run's `analysis/tables/geometric_per_ligand_metrics.csv`

External benchmark context checked:

- GEOM is a large energy-annotated conformer dataset intended for molecular property and conformer-generation modeling: https://www.nature.com/articles/s41597-022-01288-4
- CASF-2016 is the PDBbind comparative assessment benchmark: https://www.pdbbind-plus.org.cn/casf and https://pubmed.ncbi.nlm.nih.gov/30481020/
- PoseBusters was introduced to catch physical and chemical failures missed by RMSD-only docking evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/
- The Platinum Diverse Dataset is a relevant external protein-bound ligand conformation benchmark: https://pubs.acs.org/doi/10.1021/acs.jcim.6b00613

## Most Important Observations

### 1. The 94-ligand and 1219-ligand runs agree

The small core set and larger reference set tell the same story. This is useful experimentally: use the 94-ligand set for fast iteration and the 1219-ligand set as validation.

Key repeated patterns:

- Raw torsional perturbation produces the largest number of 1.0 A clusters, but also about 10 percent PoseBusters failure, almost entirely from `energy_ratio`.
- Minimization removes almost all PoseBusters failures but collapses diversity.
- RDKit raw fixed K=1000 gives the best or near-best CASF crystal recovery among generated sets.
- ChEMBL3D remains very competitive at sub-angstrom recovery despite far fewer conformers.

### 2. ChEMBL3D is not a weak baseline

On the 1219-ligand set, PoseBusters-passing ChEMBL3D has only about 68 conformers per ligand, but achieves:

| source | mean conformers | 1.0 A clusters | best RMSD | Hit@0.5 | Hit@0.75 | Hit@2.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ChEMBL3D PB | 68 | 7.4 | 0.554 | 0.591 | 0.789 | 0.975 |
| RDKit raw fixed | 973 | 38.6 | 0.482 | 0.590 | 0.813 | 0.998 |
| RDKit minimized fixed | 972 | 23.8 | 0.541 | 0.547 | 0.767 | 0.997 |
| torsion raw fixed | 863 | 62.4 | 0.495 | 0.576 | 0.763 | 0.993 |
| torsion minimized fixed | 973 | 24.3 | 0.553 | 0.528 | 0.759 | 0.993 |

Interpretation:

- ChEMBL3D is highly sample-efficient.
- RDKit raw fixed improves mean best RMSD and Hit@0.75/2.0, but needs roughly 14 times more conformers.
- If the paper says "energy-minimized conformers are bad," reviewers will reject the claim. The data say something subtler: energy-minimized conformers are compact and efficient, but high-K diverse ensembles recover additional bioactive-tail conformers.

### 3. Diversity helps most for flexible ligands

The rotatable-bond stratification is the clearest publishable result.

On the 1219-ligand set, for ligands with 9 or more rotatable bonds:

| source | mean conformers | 1.0 A clusters | best RMSD | Hit@0.5 | Hit@0.75 | Hit@2.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ChEMBL3D PB | 127 | 19.3 | 1.361 | 0.088 | 0.253 | 0.780 |
| RDKit raw fixed | 967 | 253.0 | 0.958 | 0.099 | 0.275 | 0.978 |
| RDKit minimized fixed | 967 | 168.8 | 1.077 | 0.110 | 0.308 | 0.967 |
| torsion raw fixed | 705 | 334.3 | 1.090 | 0.000 | 0.209 | 0.956 |
| torsion minimized fixed | 967 | 164.2 | 1.060 | 0.110 | 0.363 | 0.967 |

Per-ligand comparison against ChEMBL3D PB for 9+ rotatable-bond ligands:

| method | mean best RMSD delta vs ChEMBL3D | wins by >0.1 A | losses by >0.1 A | Hit@0.75 gain |
| --- | ---: | ---: | ---: | ---: |
| RDKit raw fixed | -0.403 | 60.4% | 23.1% | +2.2 pp |
| RDKit minimized fixed | -0.284 | 56.0% | 26.4% | +5.5 pp |
| torsion raw fixed | -0.271 | 53.8% | 34.1% | -4.4 pp |
| torsion minimized fixed | -0.301 | 58.2% | 23.1% | +11.0 pp |
| RDKit raw dynamic | -0.253 | 54.9% | 30.8% | -8.8 pp |

This is the strongest evidence for the hypothesis. For rigid ligands, ChEMBL3D already covers the relevant conformer space. For flexible ligands, the compact energy-minimized ensemble misses bound-pose tails, and large diverse ensembles improve best RMSD and 2.0 A coverage.

### 4. Minimization trades diversity for physicality and sample concentration

The current data show minimization is not a monotonic win.

On the 1219-ligand fixed K=1000 rows:

- RDKit raw: 38.6 clusters at 1.0 A, best RMSD 0.482, Hit@0.75 0.813.
- RDKit minimized: 23.8 clusters, best RMSD 0.541, Hit@0.75 0.767.
- torsion raw: 62.4 clusters, best RMSD 0.495, Hit@0.75 0.763.
- torsion minimized: 24.3 clusters, best RMSD 0.553, Hit@0.75 0.759.

Minimization reduces energy and removes PoseBusters failures, but it also moves the ensemble toward a smaller set of local basins. For bioactive-pose recovery, that collapse can hurt.

This is a good paper figure:

- x-axis: 1.0 A cluster count
- y-axis: CASF best RMSD or Hit@0.75
- point color: energy median or PoseBusters pass rate
- facet: rotatable-bond bin

The expected message: the useful frontier is not lowest energy or maximum diversity; it is quality-filtered diversity.

### 5. Torsion-raw diversity is too expensive in quality terms

Raw torsion perturbation is the diversity upper bound, but it is not a clean training target as implemented.

Across both intersections:

- raw torsion selected-pool PoseBusters failure is about 10 percent
- failures are dominated by `energy_ratio`
- generation-level steric clash rejection is about 16.5 percent
- fixed raw torsion needs a much larger generation pool to keep the target count

This makes raw torsion perturbation useful as an experimental probe, but not yet as the main dataset recipe. The paper should treat it as "high-diversity stress test" unless an energy-window or relaxation protocol fixes the high-energy tail.

### 6. PoseBusters filtering of ChEMBL3D reveals a dataset-quality issue

ChEMBL3D all-conformer references have high PoseBusters failure rates:

- 94-ligand set: 2999 / 10828 failures, 27.7 percent
- 1219-ligand set: 36022 / 118966 failures, 30.3 percent
- failures are overwhelmingly `tetrahedral_chirality`

However, filtering ChEMBL3D barely changes CASF hit rates. On the 1219-ligand set:

- ChEMBL3D all: Hit@0.5 0.593, Hit@0.75 0.791
- ChEMBL3D PB: Hit@0.5 0.591, Hit@0.75 0.789

This is important but dangerous. It may reflect real stereochemical inconsistencies, or it may reflect standardization/reference-mapping artifacts. Do not overclaim until the failed ChEMBL3D cases are manually audited by molecule.

## What This Says About The Original Hypothesis

Your hypothesis:

> Geometrically diverse conformers are better datasets to train generative models than usual energy-minimized conformers.

Current evidence supports a narrower statement:

> Geometrically diverse, quality-filtered ensembles contain bound-pose conformers that compact energy-minimized references often miss, especially for flexible ligands.

The current evidence does not yet prove:

- a model trained on diverse conformers will learn those useful tails
- the model will not waste probability mass on high-energy or strained geometries
- diversity improves generation at fixed sample budget
- diversity improves downstream docking, scoring, or virtual-screening outcomes

To prove the training-data claim, train matched models where only the conformer target distribution changes.

## Recommended Matched Training Experiment

Use identical molecule split, architecture, tokenizer, training tokens, and inference K. Change only the conformer source.

Training sets:

1. `chembl3d_min`: current ChEMBL3D/PB filtered set.
2. `rdkit_min`: RDKit ETKDG + MMFF/UFF, ChEMBL-count matched.
3. `rdkit_raw_pb`: raw RDKit, PB-filtered, ChEMBL-count matched.
4. `rdkit_mixed`: 50 percent minimized + 50 percent raw diverse, ChEMBL-count matched.
5. `torsion_relaxed_window`: torsion perturbations followed by constrained or short minimization, energy-window filtered.
6. `diverse_clustered`: select cluster medoids after PB and energy filtering, not arbitrary conformers.

Critical controls:

- Same molecules in every training set.
- Same number of conformers per molecule, or report a separate conformer-count ablation.
- Same total training tokens.
- Same train/validation/test split by canonical molecule identity.
- Hold out CASF/PDBBind/Platinum-like molecules by standardized InChIKey where possible.

Evaluation:

- CASF16 Hit@0.5, 0.75, 1.0, 2.0.
- rotatable-bond stratified performance.
- quality-filtered cluster count at K=100 and K=1000.
- local geometry and MMFF/xTB relaxation displacement.
- valid output rate and duplicate rate.
- model likelihood/calibration over conformer families if available.

The expected publishable result is not necessarily that the most diverse set wins. A more likely and more interesting result is a Pareto frontier:

- minimized data gives high precision and good local geometry
- raw diverse data gives broader bound-pose coverage but worse sample efficiency
- clustered mixed data gives the best tradeoff

## Better Metrics To Add

### 1. Diversity-conditioned bound-pose recovery

Report Hit@threshold as a function of the number of quality-filtered clusters, not only conformer count.

For each ligand:

- cluster generated conformers at 1.0 A
- select one representative per cluster
- compute CASF best RMSD using only representatives
- compare against random conformer subsampling with the same count

This tests whether new clusters are chemically useful or just redundant/noisy.

### 2. Energy-windowed diversity

Raw diversity alone is too easy to game. Report cluster counts inside energy windows relative to the molecule's best observed MMFF or xTB conformer:

- <= 5 kcal/mol
- <= 10 kcal/mol
- <= 20 kcal/mol
- all PB-passing

This would turn "quality-filtered diversity" into a stronger chemical metric.

### 3. Ligand strain to bioactive pose

For CASF crystal ligands:

- minimize the crystal ligand in isolation
- compute RMSD displacement after minimization
- estimate strain energy if feasible
- ask whether methods recover high-strain bound poses or only low-strain poses

This directly addresses whether protein-bound conformations live outside standard low-energy ensembles.

### 4. Conformer-count efficiency curves

For each method, subsample K:

- 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000

Plot Hit@0.5/0.75/2.0 versus K. ChEMBL3D probably wins early; raw/diverse methods may win late. That crossover is the paper.

### 5. Repeated-ligand multi-pose coverage

For ligands with multiple PDBBind/CASF-like complexes:

- map one standardized ligand to multiple bound poses
- evaluate how many distinct bound poses the ensemble covers

This is a direct biological argument for diversity: one molecule can bind different targets or pockets in different conformations.

### 6. Failure-mode audit by chemistry

Break out results by:

- rotatable bonds
- ring count and macrocycles
- formal charge
- number of stereocenters
- amide count
- aromatic heterocycles
- molecular weight
- flexible side-chain length

The paper becomes much stronger if the conclusion is not global but chemistry-specific.

## Additional Benchmarks Worth Adding

### PDBBind repeated-ligand slice

This is the most aligned extension. CASF is useful, but repeated ligands let you test whether one ligand-level ensemble covers multiple observed bound conformations.

### Platinum Diverse Dataset

Useful as an external protein-bound ligand conformation validation set. It is directly relevant because it was designed around experimentally observed protein-bound ligand conformations.

### PoseBusters benchmark set

Useful for physical validity and docking-like quality gates. Use it as validation for whether generated conformers remain chemically plausible when compared against protein-ligand reference structures.

### GEOM-DRUGS / GEOM-DRUGS-Revisited

Keep these as diagnostic vacuum-conformer benchmarks, not as the primary claim. They answer whether a method matches a finite energy-biased reference ensemble, not whether it recovers protein-bound conformations.

### xTB/CREST subset

For a smaller set, generate higher-quality semiempirical ensembles and compare:

- low-energy ensemble coverage
- CASF-bound pose proximity
- strain/relaxation displacement

This would anchor the analysis chemically and prevent the story from sounding like only an RDKit artifact.

## Candidate Paper Narratives

### Narrative A: "The finite-reference trap"

Standard conformer-generation benchmarks reward matching a finite low-energy reference ensemble. CASF-bound poses reveal that useful conformer generation requires a different objective: quality-filtered coverage of bioactive conformer tails.

Best if you add:

- GEOM-style COV/MAT for the same methods
- rank disagreement between GEOM recall and CASF Hit@K
- examples where GEOM/ChEMBL references miss a CASF-like conformation that diverse methods recover

### Narrative B: "Energy minimization improves chemistry but collapses bioactive diversity"

Minimization dramatically improves energy and PoseBusters validity, but it reduces conformer diversity and can move ensembles away from bound-like tails. The best conformer datasets should combine local geometry plausibility with controlled diversity.

Best if you add:

- energy-windowed diversity
- xTB relaxation displacement
- examples where minimization destroys a bound-like conformation

### Narrative C: "Flexible ligands need different conformer datasets"

Rigid molecules are solved by compact energy-minimized ensembles. Flexible molecules are not. Dataset construction should be conditional on molecular flexibility, with more diverse or cluster-balanced conformers for high-rotor ligands.

This is currently the strongest narrative from the data.

Best if you add:

- rotatable-bond stratified curves
- repeated-ligand multi-pose coverage
- matched model training by flexibility bin

### Narrative D: "Quality-filtered diversity is a training target"

Instead of training on arbitrary energy-minimized conformers, train models on conformer distributions selected by:

- PoseBusters pass
- bounded energy/strain
- cluster diversity
- bioactive-pose coverage where known

This is the most aligned with your generative-model goal, but it requires matched model training before it is proven.

## Recommended Next Analyses On Existing Artifacts

1. Build K-subsampling curves from the fixed 1000-conformer pools.
2. Run all metrics by rotatable-bond bins and include confidence intervals by bootstrap over ligands.
3. For every method, compute area under the Hit@K curve.
4. Compute cluster-representative Hit@K to test whether clusters are useful.
5. Audit the ChEMBL3D `tetrahedral_chirality` failures.
6. Produce 10 molecule case studies:
   - ChEMBL3D miss, RDKit/diverse hit
   - ChEMBL3D hit, diverse miss
   - flexible 9+ rotor cases
   - torsion raw high-energy failure cases
7. Add molecule-level overlap/leakage checks against any intended training set.

## Why Minimization Helps Only In Sampled Subsets

Follow-up inspection of the 1219-ligand per-ligand metrics and generation code suggests the sampled-subset minimization effect is real but has two different causes.

First, the `dynamic` and `chembl_count` rows are not paired raw-to-minimized subsets. Each stage builds its own fixed pool, then samples indices from that stage-specific pool before PoseBusters filtering. In `generate_casf_smiles_conformer_sets.py`, raw and minimized methods use different stable seed keys, and subset indices are sampled from the stage-specific fixed pool. Therefore subset-level raw/minimized comparisons include both minimization effects and stochastic pool/subset composition effects.

Second, CASF RMSD and clustering are computed on all loaded SDF conformers in `scripts/analyze_casf_conformer_sets.py`; only energy statistics use the PoseBusters-filtered subset. So the Hit@K change is a conformer-geometry/sampling effect, not simply a PoseBusters denominator artifact.

On the 1219-ligand set:

| family | tier | Hit@0.5 raw | Hit@0.5 minimized | delta | Hit@0.75 raw | Hit@0.75 minimized | delta | best RMSD delta, min - raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RDKit | dynamic | 0.482 | 0.495 | +0.013 | 0.719 | 0.712 | -0.007 | +0.015 |
| RDKit | ChEMBL-count | 0.424 | 0.461 | +0.037 | 0.684 | 0.684 | 0.000 | +0.003 |
| RDKit | fixed 1000 | 0.590 | 0.547 | -0.043 | 0.813 | 0.767 | -0.046 | +0.059 |
| torsion | dynamic | 0.427 | 0.478 | +0.051 | 0.642 | 0.691 | +0.048 | -0.028 |
| torsion | ChEMBL-count | 0.341 | 0.443 | +0.102 | 0.595 | 0.646 | +0.051 | -0.059 |
| torsion | fixed 1000 | 0.576 | 0.528 | -0.048 | 0.763 | 0.759 | -0.004 | +0.058 |

Interpretation:

- RDKit subset minimization is mostly a threshold effect. It creates enough new sub-0.5 A hits to increase Hit@0.5, but mean best RMSD does not improve and Hit@0.75 does not consistently improve. At fixed K=1000, raw RDKit wins because the full raw pool has enough coverage and minimization collapses useful diversity.
- Torsion subset minimization is stronger. Raw torsion is very diverse but high-energy and inefficient at small K. Minimization collapses extreme torsions into chemically plausible basins, improving both best RMSD and Hit@0.5/0.75 for sampled subsets. At fixed K=1000, raw torsion regains enough coverage that the diversity advantage offsets the minimization benefit.
- The effect is most convincing for torsion and for higher rotatable-bond bins. For torsion ChEMBL-count on 9+ rotor ligands, Hit@0.75 rises from 0.121 to 0.209; for torsion dynamic it rises from 0.077 to 0.253. For fixed K=1000, torsion minimization also helps 9+ rotor ligands, but hurts lower-flexibility bins enough that the aggregate fixed result is flat or worse.

The likely mechanism is a K-dependent precision/coverage tradeoff:

- At low K, minimization improves sample efficiency by pulling sampled structures into fewer, lower-energy basins, so more samples land near common bioactive-like conformations.
- At high K, raw ensembles have enough samples to cover rare bound-like tails; minimization can erase or merge those tail conformers, lowering aggregate hit rate.
- For RDKit, this is mostly cutoff churn near 0.5 A. For torsion perturbation, it is a real correction of an overly broad and high-energy proposal distribution.

This should be validated with paired K-subsampling curves from the same fixed pools. The current `dynamic` and `chembl_count` rows are useful but not the cleanest causal test because raw and minimized subsets are sampled from different stage-specific pools.

## Suggested Main Figures

1. CASF Hit@threshold by method, split by K regime.
2. Hit@0.75 versus 1.0 A cluster count, colored by energy median.
3. Rotatable-bond stratified best RMSD distributions.
4. K-subsampling curves for ChEMBL3D, RDKit raw, RDKit minimized, torsion raw, torsion minimized.
5. Energy-windowed cluster counts.
6. Example molecules showing bound-pose tail recovery.
7. Rank-disagreement heatmap across GEOM COV/MAT, diversity, CASF Hit@K, and local geometry.

## Bottom Line

The current result is promising, but the paper should be framed as an evaluation and dataset-design paper, not yet as proof that a particular generative model will improve. The strongest data-backed message is:

> Low-energy conformer sets are efficient but incomplete. For flexible ligands, quality-filtered geometric diversity recovers bound-pose conformations that compact minimized ensembles miss. The next step is to train matched generative models on energy-minimized, raw-diverse, and cluster-balanced datasets to test whether this bioactive-tail coverage transfers into learned sampling.
