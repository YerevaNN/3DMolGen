#!/usr/bin/env bash
# Iterative gating, rounds 2..N: each round refines the previous round's GATED
# ensemble, so the chain is strictly sequential — refine, bust, gate, repeat.
#   METHOD=torsional ROUNDS="2 3 4 5 6 7 8 9 10" scripts/submit_gated_chain.sh
set -euo pipefail
REPO=${REPO:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
METHOD=${METHOD:-torsional}
ROUNDS=${ROUNDS:-"2 3 4 5 6 7 8 9 10"}
SHARDS=${SHARDS:-8}
cd "$REPO"
printf '%-6s %-12s %-12s %-12s %-12s\n' round refine posebusters assemble covmat
prev=""
for r in $ROUNDS; do
  dep=(); [[ -n "$prev" ]] && dep=(--dependency=afterok:"$prev" --kill-on-invalid-dep=yes)
  ref=$(METHOD=$METHOD ROUND=$r sbatch --parsable "${dep[@]}" scripts/run_gated_iterate.sbatch)
  pb=$(RUN_DIR="outputs/baseline_refinement/${METHOD}_g${r}" \
       OUT_DIR="outputs/baseline_refinement/pb_${METHOD}_g${r}" PB_CONDITIONS="after" \
       sbatch --parsable --array=0-$((SHARDS-1)) --dependency=afterok:"$ref" --kill-on-invalid-dep=yes \
         --job-name="pbg-${METHOD}-r${r}" --qos=researcher --partition=premium \
         --cpus-per-task=16 --mem=48G --time=03:00:00 scripts/run_setup3_refine_posebusters.sbatch)
  asm=$(METHOD=$METHOD ROUND=$r sbatch --parsable --dependency=afterok:"$pb" --kill-on-invalid-dep=yes \
         scripts/run_gated_assemble.sbatch)
  ev=$(METHOD=$METHOD ROUND=$r sbatch --parsable --dependency=afterok:"$asm" --kill-on-invalid-dep=yes \
         scripts/run_gated_eval.sbatch)
  printf '%-6s %-12s %-12s %-12s %-12s\n' "$r" "$ref" "$pb" "$asm" "$ev"
  prev=$asm      # only the assembled ensemble gates the next round; scoring runs alongside
done
