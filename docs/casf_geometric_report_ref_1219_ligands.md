# CASF Geometric Analysis Revision (1219 ligands)

- Root: `/mnt/weka/mbedrosian/conformer_sets_ref_chembl3d_exact_intersection_pb_full_dynamic_chembl_count`
- Generation: `/mnt/weka/mbedrosian/conformer_sets_ref_chembl3d_exact_intersection_pb_full_dynamic_chembl_count/generation`
- CASF crystal ligands: `/mnt/weka/mbedrosian/code/3DMolGen/src/molgen3D/pharmacophore/data/CASF16_REF/ligands`
- CASF optimized ligands: `/mnt/weka/mbedrosian/code/3DMolGen/src/molgen3D/pharmacophore/data/CASF16/ligands_opt`
- Ligands with ChEMBL3D map: 1219 / 1219
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
| rdkit_random_raw_fixed | 1219/1000 | 1000.000 | 1219000 | 1218806 | 194 | 0.000 | 1.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 1000.000 | 1219000 | 1218516 | 484 | 0.000 | 1.000 |
| torsion_raw_fixed | 1219/898 | 1000.000 | 1219000 | 1094269 | 124731 | 0.102 | 0.898 |
| torsion_minimized_fixed | 1219/1000 | 1000.000 | 1219000 | 1218974 | 26 | 0.000 | 1.000 |
| rdkit_random_raw_dynamic | 1219/82 | 81.770 | 99678 | 99662 | 16 | 0.000 | 1.000 |
| rdkit_random_minimized_dynamic | 1219/82 | 81.770 | 99678 | 99638 | 40 | 0.000 | 1.000 |
| torsion_raw_dynamic | 1219/70 | 81.770 | 99678 | 85638 | 14040 | 0.103 | 0.897 |
| torsion_minimized_dynamic | 1219/82 | 81.770 | 99678 | 99677 | 1 | 0.000 | 1.000 |
| rdkit_random_raw_chembl_count | 1219/84 | 97.600 | 102864 | 102851 | 13 | 0.000 | 0.996 |
| rdkit_random_minimized_chembl_count | 1219/84 | 97.600 | 102864 | 102801 | 63 | 0.001 | 0.995 |
| torsion_raw_chembl_count | 1219/74 | 97.600 | 102864 | 90234 | 12630 | 0.102 | 0.894 |
| torsion_minimized_chembl_count | 1219/84 | 97.600 | 102864 | 102862 | 2 | 0.000 | 0.996 |

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
| rdkit_random_raw_fixed | 1219/1000 | 1219000 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 1219000 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| torsion_raw_fixed | 1219/898 | 2269000 | 0 | 565472 | 0 | 0.000 | 0.165 | 0.000 |
| torsion_minimized_fixed | 1219/1000 | 3055000 | 0 | 770786 | 0 | 0.000 | 0.165 | 0.000 |

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
| rdkit_random_raw_fixed | 1219/1000 | 1219000 | 194 | - | 1218806 |
| rdkit_random_minimized_fixed | 1219/1000 | 1219000 | 484 | 0 | 1218516 |
| torsion_raw_fixed | 1219/898 | 1219000 | 124731 | - | 1094269 |
| torsion_minimized_fixed | 1219/1000 | 1219000 | 26 | 17276 | 1218974 |
| rdkit_random_raw_dynamic | 1219/82 | 99678 | 16 | - | 99662 |
| rdkit_random_minimized_dynamic | 1219/82 | 99678 | 40 | 0 | 99638 |
| torsion_raw_dynamic | 1219/70 | 99678 | 14040 | - | 85638 |
| torsion_minimized_dynamic | 1219/82 | 99678 | 1 | 17276 | 99677 |
| rdkit_random_raw_chembl_count | 1219/84 | 102864 | 13 | - | 102851 |
| rdkit_random_minimized_chembl_count | 1219/84 | 102864 | 63 | 0 | 102801 |
| torsion_raw_chembl_count | 1219/74 | 102864 | 12630 | - | 90234 |
| torsion_minimized_chembl_count | 1219/84 | 102864 | 2 | 17276 | 102862 |

### Selected-Set Filter Rates

Column definitions:

- `source`: Generation method or reference source label.
- `ligands_scope`: Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).
- `pb_fail_rate_mean`: Per-ligand PoseBusters failure rate (`pb_fail / pb_input`), averaged across ligands.
- `post_min_clash_fail_rate_mean`: Minimized methods only: post-minimization clash rejects divided by minimization input, averaged per ligand.
- `kept_vs_target_rate_mean`: Per-ligand final yield (`kept / target_confs`), averaged across ligands.

| source | ligands_scope | pb_fail_rate_mean | post_min_clash_fail_rate_mean | kept_vs_target_rate_mean |
| --- | --- | --- | --- | --- |
| rdkit_random_raw_fixed | 1219/1000 | 0.000 | - | 1.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 0.000 | 0.000 | 1.000 |
| torsion_raw_fixed | 1219/898 | 0.102 | - | 0.898 |
| torsion_minimized_fixed | 1219/1000 | 0.000 | 0.009 | 1.000 |
| rdkit_random_raw_dynamic | 1219/82 | 0.000 | - | 1.000 |
| rdkit_random_minimized_dynamic | 1219/82 | 0.000 | 0.000 | 1.000 |
| torsion_raw_dynamic | 1219/70 | 0.103 | - | 0.897 |
| torsion_minimized_dynamic | 1219/82 | 0.000 | 0.009 | 1.000 |
| rdkit_random_raw_chembl_count | 1219/84 | 0.000 | - | 0.996 |
| rdkit_random_minimized_chembl_count | 1219/84 | 0.001 | 0.000 | 0.995 |
| torsion_raw_chembl_count | 1219/74 | 0.102 | - | 0.894 |
| torsion_minimized_chembl_count | 1219/84 | 0.000 | 0.009 | 0.996 |

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
| rdkit_random_raw_fixed | 1219/1000 | 1219000 | internal_steric_clash | 99 | 0.000 |
| rdkit_random_raw_fixed | 1219/1000 | 1219000 | planar_double_bonds | 68 | 0.000 |
| rdkit_random_raw_fixed | 1219/1000 | 1219000 | bond_lengths | 27 | 0.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 1219000 | double_bond_stereochemistry | 444 | 0.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 1219000 | planar_double_bonds | 34 | 0.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 1219000 | energy_ratio | 4 | 0.000 |
| rdkit_random_minimized_fixed | 1219/1000 | 1219000 | internal_steric_clash | 2 | 0.000 |
| torsion_raw_fixed | 1219/898 | 1219000 | energy_ratio | 124731 | 0.102 |
| torsion_minimized_fixed | 1219/1000 | 1219000 | tetrahedral_chirality | 21 | 0.000 |
| torsion_minimized_fixed | 1219/1000 | 1219000 | double_bond_stereochemistry | 3 | 0.000 |
| torsion_minimized_fixed | 1219/1000 | 1219000 | energy_ratio | 1 | 0.000 |
| torsion_minimized_fixed | 1219/1000 | 1219000 | planar_double_bonds | 1 | 0.000 |
| rdkit_random_raw_dynamic | 1219/82 | 99678 | planar_double_bonds | 8 | 0.000 |
| rdkit_random_raw_dynamic | 1219/82 | 99678 | bond_lengths | 5 | 0.000 |
| rdkit_random_raw_dynamic | 1219/82 | 99678 | internal_steric_clash | 3 | 0.000 |
| rdkit_random_minimized_dynamic | 1219/82 | 99678 | double_bond_stereochemistry | 38 | 0.000 |
| rdkit_random_minimized_dynamic | 1219/82 | 99678 | energy_ratio | 1 | 0.000 |
| rdkit_random_minimized_dynamic | 1219/82 | 99678 | planar_double_bonds | 1 | 0.000 |
| torsion_raw_dynamic | 1219/70 | 99678 | energy_ratio | 14040 | 0.141 |
| torsion_minimized_dynamic | 1219/82 | 99678 | tetrahedral_chirality | 1 | 0.000 |
| rdkit_random_raw_chembl_count | 1219/84 | 102864 | planar_double_bonds | 10 | 0.000 |
| rdkit_random_raw_chembl_count | 1219/84 | 102864 | bond_lengths | 2 | 0.000 |
| rdkit_random_raw_chembl_count | 1219/84 | 102864 | internal_steric_clash | 1 | 0.000 |
| rdkit_random_minimized_chembl_count | 1219/84 | 102864 | double_bond_stereochemistry | 51 | 0.000 |
| rdkit_random_minimized_chembl_count | 1219/84 | 102864 | planar_double_bonds | 12 | 0.000 |
| torsion_raw_chembl_count | 1219/74 | 102864 | energy_ratio | 12630 | 0.123 |
| torsion_minimized_chembl_count | 1219/84 | 102864 | planar_double_bonds | 1 | 0.000 |
| torsion_minimized_chembl_count | 1219/84 | 102864 | tetrahedral_chirality | 1 | 0.000 |

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
| casf_crystal | 1219/1 | 1219 | 10 | 24 | 0.008 | 0.020 |
| chembl3d_sdf | 1219/1 | 1219 | 0 | 33 | 0.000 | 0.027 |
| chembl3d_gt | 1219/98 | 118966 | 154 | 36022 | 0.001 | 0.303 |
| chembl3d_gt_pb | 1219/68 | 82944 | 78 | 0 | 0.001 | 0.000 |

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
| casf_crystal | 1219/1 | 1219 | energy_ratio | 12 | 0.010 |
| casf_crystal | 1219/1 | 1219 | internal_steric_clash | 10 | 0.008 |
| casf_crystal | 1219/1 | 1219 | bond_lengths | 1 | 0.001 |
| casf_crystal | 1219/1 | 1219 | planar_double_bonds | 1 | 0.001 |
| chembl3d_sdf | 1219/1 | 1219 | tetrahedral_chirality | 33 | 0.027 |
| chembl3d_gt | 1219/98 | 118966 | tetrahedral_chirality | 35867 | 0.301 |
| chembl3d_gt | 1219/98 | 118966 | internal_steric_clash | 154 | 0.001 |
| chembl3d_gt | 1219/98 | 118966 | energy_ratio | 1 | 0.000 |

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
| rdkit_random_raw_fixed | 1219/1000 | 126.796 | 38.648 | 4.512 | 1.433 |
| rdkit_random_minimized_fixed | 1219/1000 | 64.477 | 23.797 | 3.904 | 1.394 |
| torsion_raw_fixed | 1219/898 | 258.567 | 62.392 | 4.865 | 1.459 |
| torsion_minimized_fixed | 1219/1000 | 63.372 | 24.277 | 3.951 | 1.416 |
| rdkit_random_raw_dynamic | 1219/82 | 37.118 | 17.518 | 3.349 | 1.338 |
| rdkit_random_minimized_dynamic | 1219/82 | 26.507 | 12.992 | 3.067 | 1.334 |
| torsion_raw_dynamic | 1219/70 | 49.106 | 21.884 | 3.484 | 1.334 |
| torsion_minimized_dynamic | 1219/82 | 25.715 | 12.864 | 3.056 | 1.332 |
| rdkit_random_raw_chembl_count | 1219/84 | 34.714 | 15.385 | 3.030 | 1.303 |
| rdkit_random_minimized_chembl_count | 1219/84 | 23.757 | 11.528 | 2.875 | 1.313 |
| torsion_raw_chembl_count | 1219/74 | 43.614 | 19.267 | 3.125 | 1.303 |
| torsion_minimized_chembl_count | 1219/84 | 22.138 | 11.090 | 2.811 | 1.305 |
| chembl3d_gt | 1219/98 | 24.458 | 7.799 | 1.998 | 1.176 |
| chembl3d_gt_pb | 1219/68 | 22.331 | 7.409 | 1.948 | 1.170 |

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
| rdkit_random_raw_fixed | 1219/1000 | 12.681 | 3.866 | 0.451 | 0.143 | 0.724 | 0.403 |
| rdkit_random_minimized_fixed | 1219/1000 | 6.450 | 2.381 | 0.391 | 0.139 | 0.665 | 0.484 |
| torsion_raw_fixed | 1219/898 | 31.042 | 7.922 | 0.649 | 0.179 | 0.765 | 0.361 |
| torsion_minimized_fixed | 1219/1000 | 6.338 | 2.428 | 0.395 | 0.142 | 0.596 | 0.556 |
| rdkit_random_raw_dynamic | 1219/82 | 35.631 | 17.511 | 6.362 | 5.006 | 0.724 | 0.442 |
| rdkit_random_minimized_dynamic | 1219/82 | 25.413 | 13.710 | 6.192 | 5.006 | 0.664 | 0.510 |
| torsion_raw_dynamic | 1219/70 | 58.475 | 25.186 | 7.300 | 5.397 | 0.739 | 0.417 |
| torsion_minimized_dynamic | 1219/82 | 24.500 | 13.741 | 6.232 | 5.019 | 0.608 | 0.570 |
| rdkit_random_raw_chembl_count | 1219/84 | 51.946 | 30.783 | 16.931 | 14.992 | 0.702 | 0.466 |
| rdkit_random_minimized_chembl_count | 1219/84 | 40.850 | 26.632 | 16.746 | 14.985 | 0.655 | 0.524 |
| torsion_raw_chembl_count | 1219/74 | 72.848 | 38.942 | 18.398 | 15.826 | 0.704 | 0.447 |
| torsion_minimized_chembl_count | 1219/84 | 39.795 | 26.307 | 16.596 | 14.970 | 0.590 | 0.585 |
| chembl3d_gt | 1219/98 | 47.280 | 25.142 | 16.174 | 14.903 | 0.695 | 0.487 |
| chembl3d_gt_pb | 1219/68 | 47.390 | 25.281 | 16.256 | 14.976 | 0.693 | 0.492 |

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
| rdkit_random_raw_fixed | 1186/1000 | 53.718 | 137.849 | 76.934 | 10.771 |
| rdkit_random_minimized_fixed | 1186/999 | 16.919 | 34.746 | 20.301 | 2.938 |
| torsion_raw_fixed | 1186/887 | 31.557 | 237.141 | 66.729 | 31.884 |
| torsion_minimized_fixed | 1187/999 | 17.000 | 28.614 | 19.841 | 2.627 |
| rdkit_random_raw_dynamic | 1186/81 | 58.254 | 112.292 | 76.754 | 10.313 |
| rdkit_random_minimized_dynamic | 1186/81 | 17.066 | 29.580 | 20.345 | 2.842 |
| torsion_raw_dynamic | 1186/69 | 36.107 | 186.691 | 66.948 | 30.979 |
| torsion_minimized_dynamic | 1186/81 | 17.043 | 27.121 | 19.824 | 2.595 |
| rdkit_random_raw_chembl_count | 1186/66 | 60.790 | 108.246 | 77.001 | 9.758 |
| rdkit_random_minimized_chembl_count | 1186/66 | 17.223 | 28.563 | 20.467 | 2.599 |
| torsion_raw_chembl_count | 1185/55 | 39.354 | 171.527 | 67.831 | 29.111 |
| torsion_minimized_chembl_count | 1187/66 | 17.232 | 26.284 | 19.924 | 2.445 |
| chembl3d_gt | 1219/98 | 26.053 | 36.829 | 30.914 | 2.919 |
| chembl3d_gt_pb | 1219/68 | 26.132 | 36.630 | 30.897 | 2.893 |

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
| rdkit_random_raw_fixed | 1219/1000 | 0.482 | 1.351 | 0.262 | 0.590 | 0.813 | 0.998 |
| rdkit_random_minimized_fixed | 1219/1000 | 0.541 | 1.324 | 0.203 | 0.547 | 0.767 | 0.997 |
| torsion_raw_fixed | 1219/898 | 0.495 | 1.341 | 0.322 | 0.576 | 0.763 | 0.993 |
| torsion_minimized_fixed | 1219/1000 | 0.553 | 1.285 | 0.199 | 0.528 | 0.759 | 0.993 |
| rdkit_random_raw_dynamic | 1219/82 | 0.579 | 1.342 | 0.201 | 0.482 | 0.719 | 0.995 |
| rdkit_random_minimized_dynamic | 1219/82 | 0.594 | 1.333 | 0.184 | 0.495 | 0.712 | 0.989 |
| torsion_raw_dynamic | 1219/70 | 0.640 | 1.341 | 0.210 | 0.427 | 0.642 | 0.991 |
| torsion_minimized_dynamic | 1219/82 | 0.612 | 1.290 | 0.184 | 0.478 | 0.691 | 0.989 |
| rdkit_random_raw_chembl_count | 1219/84 | 0.629 | 1.342 | 0.159 | 0.424 | 0.684 | 0.989 |
| rdkit_random_minimized_chembl_count | 1219/84 | 0.632 | 1.321 | 0.158 | 0.461 | 0.684 | 0.989 |
| torsion_raw_chembl_count | 1219/74 | 0.716 | 1.341 | 0.120 | 0.341 | 0.595 | 0.984 |
| torsion_minimized_chembl_count | 1219/84 | 0.656 | 1.282 | 0.159 | 0.443 | 0.646 | 0.981 |
| chembl3d_gt | 1219/98 | 0.553 | 1.298 | 0.277 | 0.593 | 0.791 | 0.975 |
| chembl3d_gt_pb | 1219/68 | 0.554 | 1.292 | 0.277 | 0.591 | 0.789 | 0.975 |

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
| rdkit_random_raw_fixed | 1219/1000 | - | - | - | - | - | - |
| rdkit_random_minimized_fixed | 1219/1000 | - | - | - | - | - | - |
| torsion_raw_fixed | 1219/898 | - | - | - | - | - | - |
| torsion_minimized_fixed | 1219/1000 | - | - | - | - | - | - |
| rdkit_random_raw_dynamic | 1219/82 | - | - | - | - | - | - |
| rdkit_random_minimized_dynamic | 1219/82 | - | - | - | - | - | - |
| torsion_raw_dynamic | 1219/70 | - | - | - | - | - | - |
| torsion_minimized_dynamic | 1219/82 | - | - | - | - | - | - |
| rdkit_random_raw_chembl_count | 1219/84 | - | - | - | - | - | - |
| rdkit_random_minimized_chembl_count | 1219/84 | - | - | - | - | - | - |
| torsion_raw_chembl_count | 1219/74 | - | - | - | - | - | - |
| torsion_minimized_chembl_count | 1219/84 | - | - | - | - | - | - |
| chembl3d_gt | 1219/98 | - | - | - | - | - | - |
| chembl3d_gt_pb | 1219/68 | - | - | - | - | - | - |
