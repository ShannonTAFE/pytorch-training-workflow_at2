#!/usr/bin/env bash
#
# run_distributed.sh
#
# Launches train.py in a distributed or single-process configuration.
# Implementation path: Option 2 (PyTorch torchrun)
#
# Features:
# - Detect CUDA GPU count and choose torchrun or plain python.
# - Fall back gracefully to single-process (CPU or 1 GPU / MPS).
# - Provide clear header (device info, world size, run config).
# - Pass-through arguments to train.py (user overrides defaults).
# - Log all output to runs/ with timestamped log file.
# - Helpful error messages for missing deps / CUDA issues.
#

set -euo pipefail

# ------------- Helper: usage message -------------
show_help() {
  cat <<EOF
Usage: ./run_distributed.sh [train.py ARGS...]

Examples:
  ./run_distributed.sh
  ./run_distributed.sh --epochs 10 --batch-size 64
  ./run_distributed.sh --epochs 5 --batch-size 128 --lr 1e-3 --tracker wandb

Notes:
- This script auto-detects CUDA GPUs.
- If >= 2 CUDA GPUs are available, it will use torchrun with nproc_per_node = GPU count.
- Otherwise, it will run a single-process training job.
- Default training args are applied first; any args you pass override them.
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  show_help
  exit 0
fi

# ------------- Check for python -------------
if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] 'python' command not found. Please ensure Python is installed and on your PATH."
  exit 1
fi

# ------------- Check for torch availability -------------
if ! python - <<'PY' >/dev/null 2>&1
import torch
PY
then
  echo "[ERROR] PyTorch is not importable in this environment."
  echo "        Activate the correct conda env, e.g.:"
  echo "        conda activate pytorch_env_at2"
  exit 1
fi

# ------------- GPU / device detection -------------
CUDA_COUNT=$(python - <<'PY'
import torch
if torch.cuda.is_available():
    print(torch.cuda.device_count())
else:
    print(0)
PY
)

HAS_MPS=$(python - <<'PY'
import torch
print(int(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()))
PY
)

if [[ "$CUDA_COUNT" -ge 2 ]]; then
  WORLD_SIZE="$CUDA_COUNT"
  LAUNCH_MODE="torchrun-multi-gpu"
elif [[ "$CUDA_COUNT" -eq 1 ]]; then
  WORLD_SIZE=1
  LAUNCH_MODE="single-process-cuda"
elif [[ "$HAS_MPS" -eq 1 ]]; then
  WORLD_SIZE=1
  LAUNCH_MODE="single-process-mps"
else
  WORLD_SIZE=1
  LAUNCH_MODE="single-process-cpu"
fi

# ------------- Check for torchrun when needed -------------
if [[ "$WORLD_SIZE" -ge 2 ]]; then
  if ! command -v torchrun >/dev/null 2>&1; then
    echo "[WARN] Detected $CUDA_COUNT CUDA GPUs but 'torchrun' is not available."
    echo "       Falling back to single-process training."
    echo "       To enable multi-GPU, install a recent PyTorch with torchrun support."
    WORLD_SIZE=1
    LAUNCH_MODE="single-process-cuda"
  fi
fi

# ------------- Logging setup -------------
RUNS_DIR="runs"
mkdir -p "$RUNS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RUNS_DIR/run_${TIMESTAMP}.log"

# ------------- Default training arguments -------------
# These go FIRST so that any user-provided args override them.
# ------------- Default training arguments -------------
# These go FIRST so that any user-provided args override them.
DEFAULT_ARGS=(
  --epochs 5
  --batch-size 128
  --lr 1e-3
)


# All arguments the user passed to run_distributed.sh
USER_ARGS=( "$@" )

# ------------- Header -------------
echo "====================================================" | tee "$LOG_FILE"
echo "[RUN] $(date)" | tee -a "$LOG_FILE"
echo "[RUN] Host: $(hostname)" | tee -a "$LOG_FILE"
echo "[RUN] Launch mode: $LAUNCH_MODE" | tee -a "$LOG_FILE"
echo "[RUN] World size: $WORLD_SIZE" | tee -a "$LOG_FILE"
echo "[RUN] CUDA GPUs detected: $CUDA_COUNT" | tee -a "$LOG_FILE"
echo "[RUN] MPS available: $HAS_MPS" | tee -a "$LOG_FILE"
echo "[RUN] Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "[RUN] Default args: ${DEFAULT_ARGS[*]}" | tee -a "$LOG_FILE"
echo "[RUN] User args:    ${USER_ARGS[*]:-<none>}" | tee -a "$LOG_FILE"
echo "====================================================" | tee -a "$LOG_FILE"

# ------------- Error handling trap -------------
on_error() {
  local exit_code=$?
  echo "[ERROR] Training run failed with exit code $exit_code." | tee -a "$LOG_FILE"
  echo "[HINT] Common issues:" | tee -a "$LOG_FILE"
  echo "  - CUDA out of memory: try reducing --batch-size." | tee -a "$LOG_FILE"
  echo "  - Missing W&B login: run 'wandb login' in this environment." | tee -a "$LOG_FILE"
  echo "  - DDP mismatch: ensure train.py is compatible with torchrun if using multi-GPU." | tee -a "$LOG_FILE"
  exit "$exit_code"
}
trap on_error ERR

# ------------- Actual launch -------------
echo "[INFO] Starting training..." | tee -a "$LOG_FILE"

if [[ "$WORLD_SIZE" -ge 2 ]]; then
  # Multi-GPU launch with torchrun
  (
    set -x
    torchrun \
      --nproc_per_node="$WORLD_SIZE" \
      train.py \
      "${DEFAULT_ARGS[@]}" \
      "${USER_ARGS[@]}"
  ) 2>&1 | tee -a "$LOG_FILE"
else
  # Single-process launch (CPU, MPS, or single CUDA)
  (
    set -x
    python train.py \
      "${DEFAULT_ARGS[@]}" \
      "${USER_ARGS[@]}"
  ) 2>&1 | tee -a "$LOG_FILE"
fi

echo "====================================================" | tee -a "$LOG_FILE"
echo "[DONE] Training run completed successfully." | tee -a "$LOG_FILE"
echo "Logs saved to: $LOG_FILE"
echo "===================================================="
