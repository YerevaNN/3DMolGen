# Manuscript wording: historical processing and the first corrected revision

Updated after reading all 19 pages of `neurips_2026_ai4dd.pdf` and creating the
first revised training CSV. The wording below must not be attached to the old
model results as though those models had already used the revised data.

## Historical processing — supported description

The collection pipeline aggregates coordinate-enriched molecular strings from
multiple sources. Historical molecular exclusion compares the remaining literal
strings after removing coordinate annotations; stereochemical markers encoded
in those strings are retained. A separate exact-sample deduplication stage
compares complete enriched strings, including coordinates. These operations do
not establish global molecule-level or scaffold-level split separation.

## Completed nabla-focused revision — draft methods paragraph

We audited the training corpus against the union of nablaDFT validation,
structure-test, and scaffold-test molecular identities using full Standard
InChIKeys. Keys were generated consistently from the decoded molecular text,
retaining encoded stereochemistry, isotopes, and charges, without inferring
stereochemistry from coordinates or applying custom salt or tautomer
normalization. Standard InChI normalization itself still applies. We excluded
matching training conformers across every source, including nablaDFT. The nabla
conformation-test subset was not used as a molecular exclusion list because it
intentionally evaluates unseen conformations of training molecules. This step
removed 288,741 conformers. We separately quarantined 13 records for which a
valid molecular identity could not be generated, yielding 189,925,494 training
conformers from the 190,214,248-row input. All exclusions and quarantine reasons
were recorded, and retained records preserve their original order and bytes.

This paragraph describes a **nabla-focused revision**, not complete global
holdout exclusion. It establishes neither scaffold-family disjointness nor
absence of repeated/near-identical test geometries. Those require separate audits.

## Recorded outcome

| Quantity | Rows |
| --- | ---: |
| Original training CSV | 190,214,248 |
| Nabla val/scaffold/structure key matches excluded | 288,741 |
| Of those, from nabla training itself | 503 |
| Conversion failures quarantined separately | 13 |
| Revised training CSV | 189,925,494 |

The overlap count contains 288,238 rows from other sources. Exact-text matching
against the same union identifies 274,636 rows, so key matching adds 14,105.
Counts refer to training samples/conformers, not unique molecules.

Output: `runs/nabla_holdouts_v1/merged_train.csv`.
SHA-256: `cc11a42b2c2bafafe501caef73c811b39eadf3c3edce6167f3c7d98d37381298`.
The writer regenerated all 42,018 distinct removal-text identities, reconciled
all source/text counts against the completed audit, and checked chunk hashes
while assembling the output. A separate full-file read-back verified the output
hash, byte size, 189,925,494 retained records, 13 quarantined records, and count
conservation. Its result is `runs/nabla_holdouts_v1/independent_verification.json`.
This is file verification, not an independent recomputation of molecular keys
on every retained record.

## Broader revision — not yet a completed-results claim

We are auditing all retained molecular validation/test references and the actual
GEOM downstream validation and benchmark identities against every training
source. The frozen registry has 42 molecular exclusion references and keeps the
nabla conformation reference as a separate evaluation regime. It explicitly
records three GEOM benchmark metadata/text key discrepancies. External PoseBench,
Platinum Diverse and GEOM-XL identity coverage and scaffold/conformation audits
remain release gates. The final global exclusion count must come from the
completed audit rather than summing overlapping subset counts.

Once all declared holdouts and identity policies are frozen, rebuild packed
training artifacts, retrain the tokenizer, and regenerate downstream tokenized
corpora under the same split policy. Table 1, Table 6 and downstream results must
be regenerated or explicitly labeled with their historical data/checkpoint
provenance. Filtering a CSV does not change an existing checkpoint's exposure.

## Changes to the final manuscript

1. **Section 2, data collection (PDF p.2, lines 69–75):** replace the blanket
   canonical-SMILES separation claim with the actual identity policy and scope.
   Distinguish the historical coordinate-stripped literal-string comparison from
   the new full Standard InChIKey exclusion. Cite the frozen release manifest.
2. **Appendix A.3 (p.11, lines 415–420):** replace the blanket nabla exception
   with three named regimes. Conformation tests allow shared molecules;
   structure tests require molecular separation; scaffold claims additionally
   require the benchmark's scaffold definition and a corpus-wide scaffold check.
   Exclusion applies to every training source, including nabla itself.
3. **Table 6:** regenerate train/validation/test counts from the final saved
   split manifests. Report overlap exclusions and quarantine separately, with
   sample counts and unique molecular identities. Count the union of exclusions;
   do not sum overlapping reference-specific counts.
4. **Table 1 and Table 6 reconstruction results:** identify exactly which nabla
   subset supplies each metric. Report the three regimes separately, and state
   the populations and weighting used for aggregates. Recompute metrics after
   retraining; do not attach new filtering claims to historical checkpoints.
5. **Appendices A.11–A.12:** describe the repaired downstream train/validation/test
   membership and exclusion of downstream holdouts from tokenizer pretraining.
   Replace the old overlap discussion only after the new splits and models have
   been independently checked. Keep historical comparisons clearly labeled.

### Final-release methods template — use only after the checks pass

The following paragraph is a template, not a completed-results claim. Fill the
bracketed fields from the final release reports, and retain only guarantees that
have actually been verified:

> We constructed a revised corpus from the historical 190,214,248-conformer
> training snapshot. We froze the validation and test references listed in
> [release manifest/version] and generated full Standard InChIKeys from molecular
> strings decoded using a consistent policy and RDKit [version]. Encoded
> stereochemistry and isotopes were retained; we applied no custom salt stripping,
> tautomer normalization, or stereochemistry assignment from coordinates. Standard
> InChI normalization applies. Across all training sources, we excluded conformers
> whose keys occurred in the union of the designated molecular holdouts. Records
> with unresolved conversion or structural-quality failures were quarantined and
> reported separately. This removed [N overlap samples; M unique training keys]
> and quarantined [Q samples], leaving [R training samples; U unique keys]. We
> resolved validation–test conflicts according to [documented ownership policy]
> while preserving the original benchmark files. Nabla conformation tests were
> treated as known-molecule evaluations and audited separately for geometry
> overlap under [atom-mapping/alignment/tolerance policy]. Scaffold separation was
> assessed using [benchmark definition and measured scope]. Independent audits
> verified [precisely stated separation guarantees]. All exclusions, split
> assignments, software versions, and dataset hashes accompany the release.

If enforcing scaffold exclusion removes additional samples, report that stage
and its union with molecular exclusions explicitly; include it in the final
count equation. Do not insert the intermediate 189,925,494 count as the final
global result. External benchmark coverage, stereo disagreements, scaffold
checks, geometry checks, and structural QC are still pending at this writing.
