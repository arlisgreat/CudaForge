#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin/python}"
TASK="${1:-KernelBench/level1/19_ReLU.py}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
SERVER_TYPE="${SERVER_TYPE:-openai}"
GPU_NAME="${GPU_NAME:-NVIDIA A800-SXM4-80GB}"
DEVICE="${DEVICE:-0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-1.0}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"
OUT_ROOT="${OUT_ROOT:-run/streamma_matrix}"
FIRST_N="${FIRST_N:-0}"
NUM_TASKS="${NUM_TASKS:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
SKIP_FIRST="${SKIP_FIRST:-0}"
SUBPROC_ID="${SUBPROC_ID:-0}"
PHASES="${PHASES:-all}"
INCLUDE_P1_NL="${INCLUDE_P1_NL:-0}"

cd "$ROOT"

if [[ "$SERVER_TYPE" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it in the shell; do not write it to the repo." >&2
  exit 2
fi

common=(
  "$TASK"
  --server_type "$SERVER_TYPE"
  --model_name "$MODEL_NAME"
  --gpu "$GPU_NAME"
  --device "$DEVICE"
  --max_tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --warmup "$WARMUP"
  --repeat "$REPEAT"
  --first_n "$FIRST_N"
  --num_tasks "$NUM_TASKS"
  --shuffle_seed "$SHUFFLE_SEED"
  --skip_first "$SKIP_FIRST"
  --subproc_id "$SUBPROC_ID"
)

failures=()

run_arm() {
  local name="$1"
  shift
  echo "===== $name ====="
  set +e
  "$PY" main.py "${common[@]}" "$@" --work_dir "$OUT_ROOT/$name"
  local status=$?
  set -e
  echo "$name exit_code=$status" | tee -a "$OUT_ROOT/matrix_status.txt"
  if [[ "$status" -ne 0 ]]; then
    failures+=("$name:$status")
  fi
}

run_phase_a() {
  # Phase A: seed generation only. Use this to evaluate compile/correctness failure.
  run_arm phaseA_P0_serial_full \
    --round 1 --comm_protocol serial_full --stream_phase none --gate_mode off

  run_arm phaseA_P1_serial_segmented_json \
    --round 1 --comm_protocol serial_segmented --stream_phase seed --gate_mode balanced

  run_arm phaseA_P2_stream_nl \
    --round 1 --comm_protocol stream_nl --stream_phase seed --gate_mode off

  run_arm phaseA_P3_stream_json_gate \
    --round 1 --comm_protocol stream_json_gate --stream_phase seed --gate_mode balanced

  if [[ "$INCLUDE_P1_NL" == "1" ]]; then
    # Optional P2-matched serial control without JSON gate.
    run_arm phaseA_P1_serial_segmented_nl \
      --round 1 --comm_protocol serial_segmented --stream_phase seed --gate_mode off
  fi
}

run_phase_b() {
  # Phase B: optimization generation only. Round 0 is original seed; round 1 uses
  # protocol only if round 0 produced a runnable kernel and NCU profiling succeeds.
  run_arm phaseB_P0_serial_full \
    --round 2 --comm_protocol serial_full --stream_phase none --gate_mode off

  run_arm phaseB_P1_serial_segmented_json \
    --round 2 --comm_protocol serial_segmented --stream_phase optimization --gate_mode balanced

  run_arm phaseB_P2_stream_nl \
    --round 2 --comm_protocol stream_nl --stream_phase optimization --gate_mode off

  run_arm phaseB_P3_stream_json_gate \
    --round 2 --comm_protocol stream_json_gate --stream_phase optimization --gate_mode balanced

  if [[ "$INCLUDE_P1_NL" == "1" ]]; then
    run_arm phaseB_P1_serial_segmented_nl \
      --round 2 --comm_protocol serial_segmented --stream_phase optimization --gate_mode off
  fi
}

case "$PHASES" in
  phaseA) run_phase_a ;;
  phaseB) run_phase_b ;;
  all) run_phase_a; run_phase_b ;;
  *)
    echo "PHASES must be one of: phaseA, phaseB, all" >&2
    exit 2
    ;;
esac

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "FAILED_ARMS ${failures[*]}" | tee -a "$OUT_ROOT/matrix_status.txt"
  exit 1
fi
