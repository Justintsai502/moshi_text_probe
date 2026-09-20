#!/usr/bin/env bash
# Run the probe on a compute node. Adjust the paths to your cluster.
set -euo pipefail

PERSONAPLEX_CODE="${PERSONAPLEX_CODE:-/home/kcire/personaplex/moshi}"
PERSONAPLEX_DIR="${PERSONAPLEX_DIR:-/bathrooms/kcire/datasets/models/hf/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NOTE: activate conda BEFORE any `module load`, otherwise the module's python
# shadows the env's python even though the prompt still shows the env name.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-moshi}"

export PYTHONPATH="${PERSONAPLEX_CODE}:${HERE}"
export PERSONAPLEX_DIR

python "${HERE}/probe_text_only.py" \
  --questions "${HERE}/questions.txt" \
  --style qa \
  --max-new-tokens 48 \
  --temp 0 \
  --out "${HERE}/report_qa_greedy.json" \
  "$@"

# Also record the raw, unmasked behaviour (how often it just wants to emit PAD):
python "${HERE}/probe_text_only.py" \
  --questions "${HERE}/questions.txt" \
  --style qa \
  --no-mask-pad \
  --max-new-tokens 48 \
  --temp 0 \
  --out "${HERE}/report_qa_nomask.json" \
  "$@"
