#!/usr/bin/env bash
# Rounds 2..N of tokenizer refinement for the four GEOM baselines, with the
# metrics for each round hanging off that round's pass.
#
#   scripts/submit_baseline_iterations.sh            # rounds 2..10, all methods
#   METHODS=etflow ROUNDS=2 scripts/submit_baseline_iterations.sh
#
# One chain per method: round r's pass waits on round r-1's, because it eats
# r-1's output. PoseBusters and COV/MAT for round r wait only on round r's pass,
# so scoring never blocks the next pass.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
METHODS=${METHODS:-"torsional mcf_s mcf_b etflow"}
ROUNDS=${ROUNDS:-"2 3 4 5 6 7 8 9 10"}
PB_SHARDS=${PB_SHARDS:-4}
BASE="${REPO_ROOT}/outputs/baseline_refinement"

cd "${REPO_ROOT}"
printf '%-10s %-6s %-12s %-12s %-12s\n' method round pass posebusters covmat

for method in ${METHODS}; do
  prev_pass="${START_AFTER:-}"
  for round in ${ROUNDS}; do
    dir="${BASE}/${method}_r${round}"

    dep=()
    [[ -n "${prev_pass}" ]] && dep=(--dependency=afterok:"${prev_pass}" --kill-on-invalid-dep=yes)
    pass_id=$(METHOD="${method}" ROUND="${round}" \
      sbatch --parsable "${dep[@]}" scripts/run_baseline_iterate.sbatch)

    pb_id=$(RUN_DIR="${dir}" OUT_DIR="${BASE}/pb_${method}_r${round}" PB_CONDITIONS="after" \
      sbatch --parsable --array=0-$((PB_SHARDS - 1)) \
        --dependency=afterok:"${pass_id}" --kill-on-invalid-dep=yes \
        --job-name="pb-${method}-r${round}" --qos=researcher --partition=research_cpu \
        --cpus-per-task=16 --mem=48G \
        scripts/run_setup3_refine_posebusters.sbatch)

    cm_id=$(METHOD="${method}" OUTPUT_DIR="${dir}" \
      sbatch --parsable --dependency=afterok:"${pass_id}" --kill-on-invalid-dep=yes \
        --job-name="cm-${method}-r${round}" --qos=researcher --cpus-per-task=16 --mem=64G \
        scripts/run_baseline_covmat.sbatch)

    printf '%-10s %-6s %-12s %-12s %-12s\n' "${method}" "${round}" "${pass_id}" "${pb_id}" "${cm_id}"
    prev_pass="${pass_id}"
  done
done
