#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$ROOT_DIR/logs/videos/diagnostic_pack_20260226"
LOG_ROOT="$ROOT_DIR/logs/recovery"
SUMMARY="$LOG_ROOT/diagnostic_pack_20260226_summary.txt"
PYTHON_BIN="${PYTHON_BIN:-$HOME/isaaclab_venv/bin/python}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$SUMMARY"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not executable at $PYTHON_BIN" | tee -a "$SUMMARY"
  exit 1
fi

run_case() {
  local idx="$1"; shift
  local name="$1"; shift
  local cmd="$*"
  local out_dir="$OUT_ROOT/$(printf '%02d' "$idx")_${name}"
  local log_file="$LOG_ROOT/diag_$(printf '%02d' "$idx")_${name}.log"

  mkdir -p "$out_dir"
  echo "[$(printf '%02d' "$idx")] $name" | tee -a "$SUMMARY"
  echo "cmd: $cmd --output_dir $out_dir" > "$log_file"
  timeout 480 bash -lc "$cmd --output_dir '$out_dir'" >> "$log_file" 2>&1
  local rc=$?
  echo "rc=$rc log=$log_file" | tee -a "$SUMMARY"
}

BASE="cd '$ROOT_DIR' && export PYTHONPATH='$ROOT_DIR/rl_task:\$PYTHONPATH' && '$PYTHON_BIN' '$ROOT_DIR/scripts/record_motion_proof.py' --headless --num_envs 1 --camera_size 720 --record_views overhead,side,wrist"

run_case 1 collapse "$BASE --mode collapse --num_steps 420 --camera_zoom very_wide --collapse_no_apply"
run_case 2 sweep_all "$BASE --mode sweep --num_steps 420 --camera_zoom wide --env_action_scale 3.0 --action_gain 1.0"
run_case 3 sweep_base "$BASE --mode sweep_single --sweep_joint 0 --num_steps 360 --camera_zoom wide --env_action_scale 3.0"
run_case 4 sweep_shoulder "$BASE --mode sweep_single --sweep_joint 1 --num_steps 360 --camera_zoom wide --env_action_scale 3.0"
run_case 5 sweep_elbow "$BASE --mode sweep_single --sweep_joint 2 --num_steps 360 --camera_zoom wide --env_action_scale 3.0"
run_case 6 sweep_wrist_pitch "$BASE --mode sweep_single --sweep_joint 3 --num_steps 360 --camera_zoom wide --env_action_scale 3.0"
run_case 7 sweep_wrist_roll "$BASE --mode sweep_single --sweep_joint 4 --num_steps 360 --camera_zoom wide --env_action_scale 3.0"
run_case 8 ball_drop "$BASE --mode ball_drop --num_steps 420 --camera_zoom very_wide"
run_case 9 policy_state "$BASE --mode policy_state --checkpoint '$ROOT_DIR/logs/alice_ball_transfer/model_5749.pt' --num_steps 300 --camera_zoom wide"
run_case 10 policy_vision "$BASE --mode policy_vision --checkpoint '$ROOT_DIR/logs/alice_ball_transfer_vision/model_10000.pt' --num_steps 300 --camera_zoom wide"

echo "DONE" | tee -a "$SUMMARY"
