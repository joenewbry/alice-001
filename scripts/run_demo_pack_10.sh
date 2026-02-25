#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$ROOT_DIR/logs/videos/demo_pack_20260225"
LOG_ROOT="$ROOT_DIR/logs/recovery"
SUMMARY="$LOG_ROOT/demo_pack_20260225_summary.txt"
PYTHON_BIN="${PYTHON_BIN:-$HOME/isaaclab_venv/bin/python}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$SUMMARY"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not executable at $PYTHON_BIN" | tee -a "$SUMMARY"
  exit 1
fi

run_demo() {
  local idx="$1"; shift
  local name="$1"; shift
  local cmd="$*"

  local out_dir="$OUT_ROOT/$(printf '%02d' "$idx")_${name}"
  local log_file="$LOG_ROOT/demo_$(printf '%02d' "$idx")_${name}.log"
  mkdir -p "$out_dir"

  echo "[$(printf '%02d' "$idx")] $name" | tee -a "$SUMMARY"
  echo "cmd: $cmd --output_dir $out_dir" > "$log_file"

  timeout 420 bash -lc "$cmd --output_dir '$out_dir'" >> "$log_file" 2>&1
  local rc=$?
  echo "rc=$rc log=$log_file" | tee -a "$SUMMARY"
}

BASE="cd '$ROOT_DIR' && export PYTHONPATH='$ROOT_DIR/rl_task:\$PYTHONPATH' && '$PYTHON_BIN' '$ROOT_DIR/scripts/record_motion_proof.py' --headless --num_envs 1 --num_steps 360 --camera_size 720"

run_demo 1 sweep_tight "$BASE --mode sweep --camera_zoom tight --action_gain 1.0 --env_action_scale 1.0"
run_demo 2 sweep_wide "$BASE --mode sweep --camera_zoom wide --action_gain 1.0 --env_action_scale 2.0"
run_demo 3 sweep_very_wide "$BASE --mode sweep --camera_zoom very_wide --action_gain 1.0 --env_action_scale 2.5"
run_demo 4 ball_drop "$BASE --mode ball_drop --camera_zoom very_wide"
run_demo 5 collapse "$BASE --mode collapse --camera_zoom very_wide --collapse_no_apply"
run_demo 6 policy_state "$BASE --mode policy_state --checkpoint '$ROOT_DIR/logs/alice_ball_transfer/model_5749.pt' --camera_zoom wide --num_steps 300"
run_demo 7 policy_vision "$BASE --mode policy_vision --checkpoint '$ROOT_DIR/logs/alice_ball_transfer_vision/model_10000.pt' --camera_zoom wide --num_steps 300"
run_demo 8 policy_state_wider "$BASE --mode policy_state --checkpoint '$ROOT_DIR/logs/alice_ball_transfer/model_5749.pt' --camera_zoom very_wide --num_steps 420"
run_demo 9 sweep_fast "$BASE --mode sweep --camera_zoom wide --action_gain 1.0 --env_action_scale 3.0 --num_steps 420"
run_demo 10 ball_drop_long "$BASE --mode ball_drop --camera_zoom very_wide --num_steps 420"

echo "DONE" | tee -a "$SUMMARY"
