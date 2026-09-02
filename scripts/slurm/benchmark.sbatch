#!/usr/bin/env bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH -J cf-benchmark
#SBATCH -o /projects/prjs1859/users/%u/slurm/logs/%x_%j.out
#SBATCH -e /projects/prjs1859/users/%u/slurm/logs/%x_%j.err
# =====================================================================================
# One-shot cluster job for the multi-model benchmark (scripts/benchmark.py).
# Unlike train.sbatch.tmpl this is NOT rendered by sweep.py — submit it directly:
#
#     cd <repo on the cluster>
#     sbatch scripts/slurm/benchmark.sbatch                 # default env (load_exp_env.sh)
#     ENV=311 sbatch scripts/slurm/benchmark.sbatch         # Python 3.11 env (chronos)
#
# Every knob is an env var with a default; override at submit time, e.g.
#
#     CONFIG=experiments/experiment.yaml \
#     OVERRIDES="benchmark.top_quantile=0.1 wandb.job_type=benchmark" \
#     sbatch scripts/slurm/benchmark.sbatch
#
#   ENV          suffix picking ${ROOT_BASE}/load_exp_env_<ENV>.sh  (e.g. 311, moment)
#                empty -> ${ROOT_BASE}/load_exp_env.sh
#   ENV_SCRIPT   absolute path to an activation script; overrides ENV entirely
#   CONFIG       base experiment YAML                  (default experiments/experiment.yaml)
#   CONFIG_EXTRA extra "-c" fragments, space-sep       (default none)
#   OVERRIDES    trailing KEY=VALUE dot-list           (default none)
#   PROJECT_DIR  repo checkout on the cluster          (default $SLURM_SUBMIT_DIR)
#   WANDB_MODE   online | offline | disabled           (default online)
#
# The shared roster in experiment.yaml already skips any model whose backend is absent
# in the activated interpreter (is_model_available), so one submission per env covers
# that env's runnable subset. To benchmark a different env's models, submit again with a
# different ENV / ENV_SCRIPT; concat the per-env benchmark_summary.csv afterwards.
#
# GPU: partition is gpu_a100 (was gpu_h100). For H100 add `-p gpu_h100` at submit
# (`sbatch -p gpu_h100 ...`) and bump data.batch_size back up via OVERRIDES. Adjust the
# -t / --mem-per-gpu / -o / -e header for your account. Pre-create the log dir once:
#   mkdir -p /projects/prjs1859/users/$USER/slurm/logs
# =====================================================================================

# --- pin HF cache + offline mode (compute nodes have no internet) ---
export HF_HOME="/projects/prjs1859/users/xzhang4/.cache/huggingface"
export HF_HUB_OFFLINE=1

set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[cf-benchmark] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi

CONFIG="${CONFIG:-experiments/experiment.yaml}"
CONFIG_EXTRA="${CONFIG_EXTRA:-}"
OVERRIDES="${OVERRIDES:-}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
LOG_DIR="${LOG_DIR:-${ROOT_BASE}/users/${USER}/slurm/logs}"
mkdir -p "${LOG_DIR}"

# --- interpreter env: resolve the activation script -----------------------------------
#   ENV_SCRIPT=/abs/path.sh   -> used verbatim
#   ENV=<suffix>              -> ${ROOT_BASE}/load_exp_env_<suffix>.sh   (e.g. 311, moment)
#   neither set               -> ${ROOT_BASE}/load_exp_env.sh
ENV="${ENV:-}"
if [[ -z "${ENV_SCRIPT:-}" ]]; then
  if [[ -n "${ENV}" ]]; then
    ENV_SCRIPT="${ROOT_BASE}/load_exp_env_${ENV}.sh"
  else
    ENV_SCRIPT="${ROOT_BASE}/load_exp_env.sh"
  fi
fi
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "[cf-benchmark] activation script not found: ${ENV_SCRIPT}" >&2
  echo "[cf-benchmark] set ENV=<suffix> or ENV_SCRIPT=<abs path>. Available:" >&2
  ls -1 "${ROOT_BASE}"/load_exp_env*.sh 2>/dev/null >&2 || echo "  (none under ${ROOT_BASE})" >&2
  exit 1
fi
source "${ENV_SCRIPT}"

which python
python -c "import torch; print('torch ok', torch.__version__, '| cuda', torch.cuda.is_available())"

# --- business --------------------------------------------------------------------------
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

CMD=(python scripts/benchmark.py -c "${CONFIG}")
# shellcheck disable=SC2086
for f in ${CONFIG_EXTRA}; do CMD+=(-c "${f}"); done
# shellcheck disable=SC2206
CMD+=(${OVERRIDES})

echo "[cf-benchmark] PROJECT_DIR=${PROJECT_DIR}"
echo "[cf-benchmark] ENV_SCRIPT=${ENV_SCRIPT}"
echo "[cf-benchmark] ${CMD[*]}"
"${CMD[@]}"
