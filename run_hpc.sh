#!/usr/bin/env bash
# Run the probe on a compute node.
#
# Everything is overridable by env var:
#   PERSONAPLEX_CODE  path to the personaplex checkout's `moshi` package dir
#   PERSONAPLEX_DIR   path to the weights snapshot (model.safetensors + spm model)
#   CONDA_ENV         conda env to activate (default: current env, or "moshi")
#   SKIP_CONDA=1      don't touch conda at all (you activated it yourself)
#
# Example:
#   PERSONAPLEX_CODE=$HOME/personaplex/moshi \
#   PERSONAPLEX_DIR=$HOME/models/personaplex-7b-v1 \
#   ./run_hpc.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- conda ----
# NOTE: activate conda BEFORE any `module load`; a module's python silently
# shadows the env's python even though the prompt still shows the env name.
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
  if [[ -n "${CONDA_ENV:-}" ]]; then
    CONDA_SH="${CONDA_SH:-}"
    if [[ -z "${CONDA_SH}" ]] && command -v conda >/dev/null 2>&1; then
      CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    fi
    if [[ -f "${CONDA_SH}" ]]; then
      # shellcheck disable=SC1090
      source "${CONDA_SH}"
      conda activate "${CONDA_ENV}"
    else
      echo "warn: conda.sh not found (CONDA_SH=${CONDA_SH:-unset}); using current python" >&2
    fi
  else
    echo "info: CONDA_ENV not set; using the currently active python ($(command -v python))"
  fi
fi

# ------------------------------------------------------------- pythonpath ---
: "${PERSONAPLEX_CODE:=}"
: "${PERSONAPLEX_DIR:=}"

if [[ -z "${PERSONAPLEX_DIR}" || ! -d "${PERSONAPLEX_DIR}" ]]; then
  cat >&2 <<EOF
error: PERSONAPLEX_DIR is not a directory: '${PERSONAPLEX_DIR}'
       It must contain model.safetensors and tokenizer_spm_32k_3.model.
       Find it with, e.g.:
         ls ~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*
       or download it:
         huggingface-cli download nvidia/personaplex-7b-v1 --local-dir ./personaplex-7b-v1
EOF
  exit 2
fi

if [[ -n "${PERSONAPLEX_CODE}" ]]; then
  if [[ ! -d "${PERSONAPLEX_CODE}" ]]; then
    echo "error: PERSONAPLEX_CODE is not a directory: '${PERSONAPLEX_CODE}'" >&2
    exit 2
  fi
  export PYTHONPATH="${PERSONAPLEX_CODE}:${HERE}${PYTHONPATH:+:${PYTHONPATH}}"
else
  echo "info: PERSONAPLEX_CODE not set; relying on a pip-installed 'moshi'"
  export PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PERSONAPLEX_DIR

python - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("moshi") is None:
    sys.exit("error: cannot import 'moshi'. Set PERSONAPLEX_CODE to the checkout's "
             "moshi package dir, or `pip install moshi`.")
import torch
print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()}")
PY

# ------------------------------------------------------------------ runs ----
run() {  # run <name> <extra args...>
  local name="$1"; shift
  echo; echo "########## ${name}"
  python "${HERE}/probe_text_only.py" \
    --questions "${HERE}/questions.txt" \
    --style qa --temp 0 --max-new-tokens 48 \
    --out "${HERE}/report_${name}.json" "$@"
}

# 1. PAD/EPAD masked, audio init = -1
run qa_greedy --audio-init zero
# 2. same, but first frame uses moshi's own audio SOS instead of -1
run qa_special --audio-init special
# 3. raw behaviour: let it emit PAD if it wants to
run qa_nomask --audio-init zero --no-mask-pad

echo
echo "[done] reports: ${HERE}/report_qa_{greedy,special,nomask}.json"
