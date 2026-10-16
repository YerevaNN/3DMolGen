#!/usr/bin/env bash
#SBATCH --job-name=eval-direct
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/eval/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/eval/%j.err
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYBIN=${PYBIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/outputs/gen_results_user}
# Resolve the newest results dir matching the run name, or use RESULTS_DIR directly.
if [[ -n "${RESULTS_DIR:-}" ]]; then
  D="${RESULTS_DIR}"
else
  D=$(ls -dt "${RESULTS_ROOT}"/*_"${RUN_NAME}" 2>/dev/null | head -1)
fi
if [[ -z "${D}" || ! -d "${D}" ]]; then echo "ERROR: no results dir for RUN_NAME=${RUN_NAME} under ${RESULTS_ROOT}" >&2; exit 1; fi
echo "Evaluating: ${D}"
mkdir -p "${REPO_ROOT}/outputs/slurm_jobs/eval"
mkdir -p "${REPO_ROOT}/outputs/eval_results"
MOLGEN3D_REPO_ROOT="${REPO_ROOT}" \
MOLGEN3D_PROJECT_ROOT="${REPO_ROOT}" \
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
PYTHONNOUSERSITE=1 \
"${PYBIN}" "${REPO_ROOT}/src/molgen3D/evaluation/run_eval.py" \
  --device local --specific-dir "${D}" --test_set "${TEST_SET:-distinct}" \
  --num-workers "${NUM_WORKERS:-12}" --posebusters "${POSEBUSTERS:-None}" \
  --output-dir "${EVAL_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval_results}"
echo "EVAL_DONE ${RUN_NAME:-$D}"
