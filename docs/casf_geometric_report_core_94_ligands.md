# CASF Geometric Analysis Revision (94 ligands)

- Root: `/mnt/weka/mbedrosian/pharma_generation_analysis/core_pb_full_dynamic_chembl_count`
- Generation: `/mnt/weka/mbedrosian/pharma_generation_analysis/core_pb_full_dynamic_chembl_count/generation`
- CASF crystal ligands: `/mnt/weka/mbedrosian/data/casf16/CASF16/core_chembl3d_exact_intersection_ligands`
- CASF optimized ligands: `/mnt/weka/mbedrosian/data/casf16/CASF16/ligands_opt`
- Ligands with ChEMBL3D map: 94 / 94
- Clash cutoff: 0.7

## Generation Pipeline Overview

Per-ligand rates are averaged across ligands. PoseBusters and kept-yield rates use the selected conformer pool for that method row (`pb_input_confs`), not the full fixed-pool generation count.

ChEMBL3D ground truth appears twice in comparison tables: `chembl3d_gt` uses all loaded conformers; `chembl3d_gt_pb` uses only conformers passing PoseBusters against the CASF crystal reference (fair comparison to filtered generation sets).

### Target Set Sizes And Selected-Pool Outcomes

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `target_confs_mean`: Mean per-ligand target set size: 1000 for fixed, dynamic formula (-20 + 22 * rotatable bonds), or ChEMBL3D conformer count.
- `selected_pool_total`: Total conformers in the selected set entering the reported filter stage (PoseBusters input pool).
- `kept_confs_total`: Final conformers kept after all filters for that method row.
- `pb_fail_total`: Total conformers failing PoseBusters in the selected pool.
- `pb_fail_rate_mean`: Per-ligand PoseBusters failure rate (`pb_fail / pb_input`), averaged across ligands.
- `kept_vs_target_rate_mean`: Per-ligand final yield (`kept / target_confs`), averaged across ligands.

| source | ligands_scope | target_confs_mean | selected_pool_total | kept_confs_total | pb_fail_total | pb_fail_rate_mean | kept_vs_target_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 1000.000 | 94000 | 93995 | 5 | 0.000 | 1.000 |
| rdkit_random_minimized_fixed | 94/1000 | 1000.000 | 94000 | 93997 | 3 | 0.000 | 1.000 |
| torsion_raw_fixed | 94/896 | 1000.000 | 94000 | 84266 | 9734 | 0.104 | 0.896 |
| torsion_minimized_fixed | 94/1000 | 1000.000 | 94000 | 93997 | 3 | 0.000 | 1.000 |
| rdkit_random_raw_dynamic | 94/80 | 79.936 | 7514 | 7513 | 1 | 0.000 | 1.000 |
| rdkit_random_minimized_dynamic | 94/80 | 79.936 | 7514 | 7513 | 1 | 0.000 | 1.000 |
| torsion_raw_dynamic | 94/70 | 79.936 | 7514 | 6562 | 952 | 0.095 | 0.905 |
| torsion_minimized_dynamic | 94/80 | 79.936 | 7514 | 7514 | 0 | 0.000 | 1.000 |
| rdkit_random_raw_chembl_count | 94/106 | 115.191 | 9917 | 9917 | 0 | 0.000 | 0.993 |
| rdkit_random_minimized_chembl_count | 94/106 | 115.191 | 9917 | 9917 | 0 | 0.000 | 0.993 |
| torsion_raw_chembl_count | 94/95 | 115.191 | 9917 | 8884 | 1033 | 0.103 | 0.891 |
| torsion_minimized_chembl_count | 94/106 | 115.191 | 9917 | 9917 | 0 | 0.000 | 0.993 |

### Fixed-Pool Generation Rejects

Only fixed-tier rows are shown here. Rates are computed against the generation/minimization input pool, not the dynamic or ChEMBL-count subsets.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `generation_pool_total`: Fixed-tier only: total conformers generated or entering minimization before subset selection.
- `finite_fail_total`: Fixed-tier only: conformers rejected for non-finite coordinates during generation.
- `clash_fail_total`: Fixed-tier only: conformers rejected for steric clash during generation.
- `geometry_fail_total`: Fixed-tier only: bond/stereo/RMSD rejects during generation.
- `finite_fail_rate_mean`: Fixed-tier only: finite rejects divided by the generation pool, averaged per ligand.
- `clash_fail_rate_mean`: Fixed-tier only: clash rejects divided by the generation pool, averaged per ligand.
- `geometry_fail_rate_mean`: Fixed-tier only: geometry rejects divided by the generation pool, averaged per ligand.

| source | ligands_scope | generation_pool_total | finite_fail_total | clash_fail_total | geometry_fail_total | finite_fail_rate_mean | clash_fail_rate_mean | geometry_fail_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 94000 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| rdkit_random_minimized_fixed | 94/1000 | 94000 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| torsion_raw_fixed | 94/896 | 167000 | 0 | 38143 | 0 | 0.000 | 0.168 | 0.000 |
| torsion_minimized_fixed | 94/1000 | 234000 | 0 | 52730 | 0 | 0.000 | 0.168 | 0.000 |

### Selected-Set Filter Counts

Dynamic and ChEMBL-count rows report only the selected subset entering PoseBusters. They do not repeat fixed-pool generation rejects.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `selected_pool_total`: Total conformers in the selected set entering the reported filter stage (PoseBusters input pool).
- `pb_fail_total`: Total conformers failing PoseBusters in the selected pool.
- `post_min_clash_fail_total`: Minimized methods only: conformers rejected for clash after minimization.
- `kept_confs_total`: Final conformers kept after all filters for that method row.

| source | ligands_scope | selected_pool_total | pb_fail_total | post_min_clash_fail_total | kept_confs_total |
| --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 94000 | 5 | - | 93995 |
| rdkit_random_minimized_fixed | 94/1000 | 94000 | 3 | 0 | 93997 |
| torsion_raw_fixed | 94/896 | 94000 | 9734 | - | 84266 |
| torsion_minimized_fixed | 94/1000 | 94000 | 3 | 123 | 93997 |
| rdkit_random_raw_dynamic | 94/80 | 7514 | 1 | - | 7513 |
| rdkit_random_minimized_dynamic | 94/80 | 7514 | 1 | 0 | 7513 |
| torsion_raw_dynamic | 94/70 | 7514 | 952 | - | 6562 |
| torsion_minimized_dynamic | 94/80 | 7514 | 0 | 123 | 7514 |
| rdkit_random_raw_chembl_count | 94/106 | 9917 | 0 | - | 9917 |
| rdkit_random_minimized_chembl_count | 94/106 | 9917 | 0 | 0 | 9917 |
| torsion_raw_chembl_count | 94/95 | 9917 | 1033 | - | 8884 |
| torsion_minimized_chembl_count | 94/106 | 9917 | 0 | 123 | 9917 |

### Selected-Set Filter Rates

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `pb_fail_rate_mean`: Per-ligand PoseBusters failure rate (`pb_fail / pb_input`), averaged across ligands.
- `post_min_clash_fail_rate_mean`: Minimized methods only: post-minimization clash rejects divided by minimization input, averaged per ligand.
- `kept_vs_target_rate_mean`: Per-ligand final yield (`kept / target_confs`), averaged across ligands.

| source | ligands_scope | pb_fail_rate_mean | post_min_clash_fail_rate_mean | kept_vs_target_rate_mean |
| --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 0.000 | - | 1.000 |
| rdkit_random_minimized_fixed | 94/1000 | 0.000 | 0.000 | 1.000 |
| torsion_raw_fixed | 94/896 | 0.104 | - | 0.896 |
| torsion_minimized_fixed | 94/1000 | 0.000 | 0.001 | 1.000 |
| rdkit_random_raw_dynamic | 94/80 | 0.000 | - | 1.000 |
| rdkit_random_minimized_dynamic | 94/80 | 0.000 | 0.000 | 1.000 |
| torsion_raw_dynamic | 94/70 | 0.095 | - | 0.905 |
| torsion_minimized_dynamic | 94/80 | 0.000 | 0.001 | 1.000 |
| rdkit_random_raw_chembl_count | 94/106 | 0.000 | - | 0.993 |
| rdkit_random_minimized_chembl_count | 94/106 | 0.000 | 0.000 | 0.993 |
| torsion_raw_chembl_count | 94/95 | 0.103 | - | 0.891 |
| torsion_minimized_chembl_count | 94/106 | 0.000 | 0.001 | 0.993 |

### Generation PoseBusters Failing Tests

Only tests with at least one failure are shown. Denominators are the selected PoseBusters input pool for each method row.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `selected_pool_total`: Total conformers in the selected set entering the reported filter stage (PoseBusters input pool).
- `pb_test`: PoseBusters boolean test name.
- `pb_fail_count`: Number of conformers failing this PoseBusters test in the selected pool.
- `pb_fail_rate`: Failures divided by the row selected-pool total (PoseBusters input conformers).

| source | ligands_scope | selected_pool_total | pb_test | pb_fail_count | pb_fail_rate |
| --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 94000 | planar_double_bonds | 4 | 0.000 |
| rdkit_random_raw_fixed | 94/1000 | 94000 | bond_lengths | 1 | 0.000 |
| rdkit_random_minimized_fixed | 94/1000 | 94000 | double_bond_stereochemistry | 3 | 0.000 |
| torsion_raw_fixed | 94/896 | 94000 | energy_ratio | 9734 | 0.104 |
| torsion_minimized_fixed | 94/1000 | 94000 | tetrahedral_chirality | 2 | 0.000 |
| torsion_minimized_fixed | 94/1000 | 94000 | double_bond_stereochemistry | 1 | 0.000 |
| rdkit_random_raw_dynamic | 94/80 | 7514 | planar_double_bonds | 1 | 0.000 |
| rdkit_random_minimized_dynamic | 94/80 | 7514 | double_bond_stereochemistry | 1 | 0.000 |
| torsion_raw_dynamic | 94/70 | 7514 | energy_ratio | 952 | 0.127 |
| torsion_raw_chembl_count | 94/95 | 9917 | energy_ratio | 1033 | 0.104 |

## Reference Filter Checks

### Reference Clash And PoseBusters Failures

`casf_crystal` uses crystal poses from the CASF ligand directory. `casf_opt` is included only when an optimized pose exists under `ligands_opt`.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `selected_pool_total`: Total conformers in the selected set entering the reported filter stage (PoseBusters input pool).
- `clash_fail_total`: Fixed-tier only: conformers rejected for steric clash during generation.
- `pb_fail_total`: Total conformers failing PoseBusters in the selected pool.
- `clash_fail_rate`: Clash failures divided by conformers checked for clashes.
- `pb_fail_rate`: Failures divided by the row selected-pool total (PoseBusters input conformers).

| source | ligands_scope | selected_pool_total | clash_fail_total | pb_fail_total | clash_fail_rate | pb_fail_rate |
| --- | --- | --- | --- | --- | --- | --- |
| casf_crystal | 94/1 | 94 | 0 | 1 | 0.000 | 0.011 |
| casf_opt | 94/1 | 94 | 1 | 3 | 0.011 | 0.032 |
| chembl3d_sdf | 94/1 | 94 | 0 | 4 | 0.000 | 0.043 |
| chembl3d_gt | 94/115 | 10828 | 0 | 2999 | 0.000 | 0.277 |
| chembl3d_gt_pb | 94/83 | 7829 | 0 | 0 | 0.000 | 0.000 |

### Reference PoseBusters Failing Tests

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `selected_pool_total`: Total conformers in the selected set entering the reported filter stage (PoseBusters input pool).
- `pb_test`: PoseBusters boolean test name.
- `pb_fail_count`: Number of conformers failing this PoseBusters test in the selected pool.
- `pb_fail_rate`: Failures divided by the row selected-pool total (PoseBusters input conformers).

| source | ligands_scope | selected_pool_total | pb_test | pb_fail_count | pb_fail_rate |
| --- | --- | --- | --- | --- | --- |
| casf_crystal | 94/1 | 94 | energy_ratio | 1 | 0.011 |
| casf_opt | 94/1 | 94 | bond_angles | 2 | 0.021 |
| casf_opt | 94/1 | 94 | inchi_convertible | 1 | 0.011 |
| casf_opt | 94/1 | 94 | sanitisation | 1 | 0.011 |
| chembl3d_sdf | 94/1 | 94 | tetrahedral_chirality | 4 | 0.043 |
| chembl3d_gt | 94/115 | 10828 | tetrahedral_chirality | 2999 | 0.277 |

## Clustering Diversity

### Mean Cluster Counts Per Ligand

Includes `chembl3d_gt` (all ChEMBL3D conformers) and `chembl3d_gt_pb` (PoseBusters-passing subset only).

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `mean_clusters_0p5`: Mean greedy cluster count per ligand at 0.5 Å.
- `mean_clusters_1p0`: Mean greedy cluster count per ligand at 1.0 Å.
- `mean_clusters_2p0`: Mean greedy cluster count per ligand at 2.0 Å.
- `mean_clusters_3p0`: Mean greedy cluster count per ligand at 3.0 Å.

| source | ligands_scope | mean_clusters_0p5 | mean_clusters_1p0 | mean_clusters_2p0 | mean_clusters_3p0 |
| --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 114.170 | 35.883 | 3.468 | 1.255 |
| rdkit_random_minimized_fixed | 94/1000 | 52.330 | 20.277 | 3.021 | 1.255 |
| torsion_raw_fixed | 94/896 | 296.755 | 74.149 | 4.330 | 1.426 |
| torsion_minimized_fixed | 94/1000 | 59.915 | 25.011 | 3.383 | 1.362 |
| rdkit_random_raw_dynamic | 94/80 | 37.149 | 17.309 | 2.872 | 1.245 |
| rdkit_random_minimized_dynamic | 94/80 | 25.298 | 12.574 | 2.649 | 1.202 |
| torsion_raw_dynamic | 94/70 | 51.277 | 23.968 | 3.223 | 1.287 |
| torsion_minimized_dynamic | 94/80 | 24.936 | 13.181 | 2.872 | 1.298 |
| rdkit_random_raw_chembl_count | 94/106 | 36.596 | 14.851 | 2.777 | 1.213 |
| rdkit_random_minimized_chembl_count | 94/106 | 22.181 | 11.085 | 2.638 | 1.223 |
| torsion_raw_chembl_count | 94/95 | 55.011 | 21.840 | 3.096 | 1.287 |
| torsion_minimized_chembl_count | 94/106 | 21.809 | 11.638 | 2.702 | 1.298 |
| chembl3d_gt | 94/115 | 23.404 | 7.840 | 2.128 | 1.202 |
| chembl3d_gt_pb | 94/83 | 22.181 | 7.766 | 2.117 | 1.202 |

### Normalized Cluster Density

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `clusters_per_100_0p5`: Mean per-ligand clusters per 100 conformers at 0.5 Å.
- `clusters_per_100_1p0`: Mean per-ligand clusters per 100 conformers at 1.0 Å.
- `clusters_per_100_2p0`: Mean per-ligand clusters per 100 conformers at 2.0 Å.
- `clusters_per_100_3p0`: Mean per-ligand clusters per 100 conformers at 3.0 Å.
- `cluster_entropy_1p0`: Mean normalized Shannon entropy of 1.0 Å cluster occupancies.
- `largest_cluster_fraction_1p0`: Mean fraction of conformers in the largest 1.0 Å cluster.

| source | ligands_scope | clusters_per_100_0p5 | clusters_per_100_1p0 | clusters_per_100_2p0 | clusters_per_100_3p0 | cluster_entropy_1p0 | largest_cluster_fraction_1p0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 11.418 | 3.589 | 0.347 | 0.126 | 0.739 | 0.394 |
| rdkit_random_minimized_fixed | 94/1000 | 5.233 | 2.028 | 0.302 | 0.126 | 0.709 | 0.446 |
| torsion_raw_fixed | 94/896 | 34.601 | 8.820 | 0.507 | 0.165 | 0.763 | 0.359 |
| torsion_minimized_fixed | 94/1000 | 5.992 | 2.501 | 0.338 | 0.136 | 0.593 | 0.539 |
| rdkit_random_raw_dynamic | 94/80 | 37.614 | 18.841 | 6.887 | 5.503 | 0.748 | 0.426 |
| rdkit_random_minimized_dynamic | 94/80 | 26.238 | 14.942 | 6.644 | 5.488 | 0.714 | 0.468 |
| torsion_raw_dynamic | 94/70 | 61.338 | 27.528 | 7.579 | 5.747 | 0.743 | 0.410 |
| torsion_minimized_dynamic | 94/80 | 25.829 | 15.684 | 6.972 | 5.579 | 0.601 | 0.557 |
| rdkit_random_raw_chembl_count | 94/106 | 51.671 | 32.080 | 18.475 | 16.524 | 0.700 | 0.458 |
| rdkit_random_minimized_chembl_count | 94/106 | 40.601 | 28.716 | 18.154 | 16.614 | 0.667 | 0.499 |
| torsion_raw_chembl_count | 94/95 | 75.602 | 40.906 | 20.789 | 18.032 | 0.687 | 0.438 |
| torsion_minimized_chembl_count | 94/106 | 40.960 | 28.930 | 18.358 | 16.718 | 0.579 | 0.571 |
| chembl3d_gt | 94/115 | 44.994 | 27.644 | 17.914 | 16.559 | 0.725 | 0.467 |
| chembl3d_gt_pb | 94/83 | 45.319 | 28.488 | 18.332 | 16.972 | 0.723 | 0.471 |

## Energy

### PB-Passing Conformer Energies

Energies for generation methods and `chembl3d_gt_pb` use PoseBusters-passing conformers. `chembl3d_gt` uses all loaded ChEMBL3D conformers.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `energy_min`: Mean per-ligand minimum PB-passing MMFF94s energy, averaged across ligands.
- `energy_max`: Mean per-ligand maximum PB-passing MMFF94s energy, averaged across ligands.
- `energy_median`: Mean per-ligand median PB-passing MMFF94s energy, averaged across ligands.
- `energy_std`: Mean per-ligand energy standard deviation, averaged across ligands.

| source | ligands_scope | energy_min | energy_max | energy_median | energy_std |
| --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 90/1000 | 58.753 | 142.093 | 80.854 | 10.286 |
| rdkit_random_minimized_fixed | 90/1000 | 24.017 | 42.899 | 27.159 | 2.753 |
| torsion_raw_fixed | 90/877 | 38.136 | 254.969 | 73.210 | 33.938 |
| torsion_minimized_fixed | 90/1000 | 24.099 | 34.675 | 26.654 | 2.563 |
| rdkit_random_raw_dynamic | 90/79 | 63.028 | 116.639 | 80.868 | 9.990 |
| rdkit_random_minimized_dynamic | 90/79 | 24.192 | 37.884 | 27.414 | 2.821 |
| torsion_raw_dynamic | 90/67 | 42.371 | 197.343 | 73.031 | 32.181 |
| torsion_minimized_dynamic | 90/79 | 24.137 | 33.460 | 26.631 | 2.552 |
| rdkit_random_raw_chembl_count | 90/79 | 64.830 | 113.076 | 80.958 | 9.343 |
| rdkit_random_minimized_chembl_count | 90/79 | 24.229 | 36.068 | 26.991 | 2.655 |
| torsion_raw_chembl_count | 90/67 | 47.196 | 188.526 | 75.288 | 30.231 |
| torsion_minimized_chembl_count | 90/79 | 24.225 | 32.992 | 26.824 | 2.402 |
| chembl3d_gt | 94/115 | 32.333 | 42.566 | 37.038 | 2.829 |
| chembl3d_gt_pb | 94/83 | 32.427 | 41.993 | 36.819 | 2.592 |

## CASF16 Crystal RMSD And Hits

### Crystal Ground Truth

RMSD and hits are computed against CASF crystal poses from the ligand directory. `chembl3d_gt_pb` rows compare only PoseBusters-passing ChEMBL3D conformers.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `casf_best_rmsd`: Mean per-ligand best heavy-atom aligned RMSD to CASF crystal (`ligands`).
- `casf_median_rmsd`: Mean per-ligand median heavy-atom aligned RMSD to CASF crystal.
- `casf_hit_0p25`: Fraction of ligands whose best CASF crystal RMSD is <= 0.25 Å.
- `casf_hit_0p5`: Fraction of ligands whose best CASF crystal RMSD is <= 0.5 Å.
- `casf_hit_0p75`: Fraction of ligands whose best CASF crystal RMSD is <= 0.75 Å.
- `casf_hit_2p0`: Fraction of ligands whose best CASF crystal RMSD is <= 2.0 Å.

| source | ligands_scope | casf_best_rmsd | casf_median_rmsd | casf_hit_0p25 | casf_hit_0p5 | casf_hit_0p75 | casf_hit_2p0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 0.447 | 1.334 | 0.309 | 0.628 | 0.809 | 1.000 |
| rdkit_random_minimized_fixed | 94/1000 | 0.528 | 1.348 | 0.234 | 0.543 | 0.766 | 1.000 |
| torsion_raw_fixed | 94/896 | 0.497 | 1.345 | 0.309 | 0.564 | 0.777 | 0.989 |
| torsion_minimized_fixed | 94/1000 | 0.554 | 1.299 | 0.181 | 0.511 | 0.755 | 0.989 |
| rdkit_random_raw_dynamic | 94/80 | 0.559 | 1.324 | 0.234 | 0.479 | 0.681 | 1.000 |
| rdkit_random_minimized_dynamic | 94/80 | 0.585 | 1.371 | 0.191 | 0.500 | 0.723 | 1.000 |
| torsion_raw_dynamic | 94/70 | 0.641 | 1.349 | 0.181 | 0.383 | 0.649 | 0.989 |
| torsion_minimized_dynamic | 94/80 | 0.631 | 1.311 | 0.138 | 0.415 | 0.670 | 0.989 |
| rdkit_random_raw_chembl_count | 94/106 | 0.604 | 1.342 | 0.191 | 0.426 | 0.660 | 1.000 |
| rdkit_random_minimized_chembl_count | 94/106 | 0.632 | 1.348 | 0.181 | 0.426 | 0.681 | 1.000 |
| torsion_raw_chembl_count | 94/95 | 0.694 | 1.345 | 0.106 | 0.319 | 0.649 | 0.989 |
| torsion_minimized_chembl_count | 94/106 | 0.663 | 1.286 | 0.149 | 0.436 | 0.638 | 0.989 |
| chembl3d_gt | 94/115 | 0.524 | 1.336 | 0.255 | 0.638 | 0.798 | 0.979 |
| chembl3d_gt_pb | 94/83 | 0.524 | 1.325 | 0.255 | 0.638 | 0.798 | 0.979 |

## CASF16 Optimized Ligand RMSD And Hits

### Optimized Ground Truth

RMSD and hits are computed against optimized poses from `ligands_opt` when available for a ligand.

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `casf_opt_best_rmsd`: Mean per-ligand best heavy-atom aligned RMSD to CASF optimized ligand (`ligands_opt`).
- `casf_opt_median_rmsd`: Mean per-ligand median heavy-atom aligned RMSD to CASF optimized ligand.
- `casf_opt_hit_0p25`: Fraction of ligands whose best CASF optimized RMSD is <= 0.25 Å.
- `casf_opt_hit_0p5`: Fraction of ligands whose best CASF optimized RMSD is <= 0.5 Å.
- `casf_opt_hit_0p75`: Fraction of ligands whose best CASF optimized RMSD is <= 0.75 Å.
- `casf_opt_hit_2p0`: Fraction of ligands whose best CASF optimized RMSD is <= 2.0 Å.

| source | ligands_scope | casf_opt_best_rmsd | casf_opt_median_rmsd | casf_opt_hit_0p25 | casf_opt_hit_0p5 | casf_opt_hit_0p75 | casf_opt_hit_2p0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 94/1000 | 0.504 | 1.363 | 0.319 | 0.596 | 0.830 | 0.989 |
| rdkit_random_minimized_fixed | 94/1000 | 0.569 | 1.387 | 0.213 | 0.500 | 0.777 | 0.989 |
| torsion_raw_fixed | 94/896 | 0.547 | 1.387 | 0.277 | 0.553 | 0.766 | 0.979 |
| torsion_minimized_fixed | 94/1000 | 0.598 | 1.341 | 0.170 | 0.511 | 0.766 | 0.979 |
| rdkit_random_raw_dynamic | 94/80 | 0.613 | 1.346 | 0.202 | 0.489 | 0.702 | 0.989 |
| rdkit_random_minimized_dynamic | 94/80 | 0.625 | 1.400 | 0.181 | 0.468 | 0.734 | 0.989 |
| torsion_raw_dynamic | 94/70 | 0.689 | 1.394 | 0.160 | 0.394 | 0.617 | 0.979 |
| torsion_minimized_dynamic | 94/80 | 0.664 | 1.350 | 0.128 | 0.426 | 0.702 | 0.979 |
| rdkit_random_raw_chembl_count | 94/106 | 0.665 | 1.369 | 0.149 | 0.426 | 0.670 | 0.989 |
| rdkit_random_minimized_chembl_count | 94/106 | 0.673 | 1.385 | 0.149 | 0.415 | 0.691 | 0.989 |
| torsion_raw_chembl_count | 94/95 | 0.754 | 1.388 | 0.096 | 0.298 | 0.617 | 0.979 |
| torsion_minimized_chembl_count | 94/106 | 0.693 | 1.320 | 0.138 | 0.426 | 0.670 | 0.979 |
| chembl3d_gt | 94/115 | 0.571 | 1.362 | 0.202 | 0.617 | 0.777 | 0.979 |
| chembl3d_gt_pb | 94/83 | 0.571 | 1.352 | 0.202 | 0.617 | 0.777 | 0.979 |
