#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/recovery"
ISAACLAB_SH="${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}"
URDF_PATH="${URDF_PATH:-$ROOT_DIR/urdf/alice_001.urdf}"
USDA_PATH="${USDA_PATH:-$ROOT_DIR/usd/alice_001.usda}"

mkdir -p "$LOG_DIR"

fail() { echo "[FAIL] $1"; exit 1; }
pass() { echo "[PASS] $1"; }

[ -f "$ISAACLAB_SH" ] || fail "isaaclab.sh not found at $ISAACLAB_SH"
[ -f "$URDF_PATH" ] || fail "URDF not found: $URDF_PATH"
[ -f "$USDA_PATH" ] || fail "USDA not found: $USDA_PATH"

echo "== [1/4] URDF vs USD structural check =="
urdf_joints="$(
  grep -oE '<joint name="[^"]+"' "$URDF_PATH" \
    | sed -E 's/.*"([^"]+)"/\1/' \
    | sort -u
)"
usd_joints="$(
  grep -oE 'def Physics(Revolute|Prismatic|Spherical|Fixed)?Joint "[^"]+"' "$USDA_PATH" \
    | sed -E 's/.*"([^"]+)"/\1/' \
    | sort -u
)"

echo "$urdf_joints" >"$LOG_DIR/urdf_joints.txt"
echo "$usd_joints" >"$LOG_DIR/usd_joints.txt"
if diff -u "$LOG_DIR/urdf_joints.txt" "$LOG_DIR/usd_joints.txt" >"$LOG_DIR/triage_joint_name_diff.log"; then
  pass "URDF/USD joint names match"
else
  echo "[WARN] Joint name mismatch; see $LOG_DIR/triage_joint_name_diff.log"
fi

echo
echo "== [2/4] Runtime sanity =="
if [ -x "$ROOT_DIR/scripts/runtime_sanity_check.sh" ]; then
  if "$ROOT_DIR/scripts/runtime_sanity_check.sh" >"$LOG_DIR/triage_runtime_sanity.log" 2>&1; then
    pass "runtime_sanity_check.sh passed"
  else
    echo "[WARN] runtime sanity failed; see $LOG_DIR/triage_runtime_sanity.log"
  fi
else
  echo "[WARN] runtime_sanity_check.sh not executable; skipping"
fi

echo
echo "== [3/4] Deterministic articulation gates =="
if "$ISAACLAB_SH" -p "$ROOT_DIR/scripts/validate_articulation_motion.py" --headless --num_envs 1 --output_csv "$LOG_DIR/articulation_motion.csv" >"$LOG_DIR/triage_articulation.log" 2>&1; then
  pass "articulation gate passed"
else
  fail "articulation gate failed; see $LOG_DIR/triage_articulation.log"
fi

echo
echo "== [4/4] Optional motion-proof video =="
if "$ISAACLAB_SH" -p "$ROOT_DIR/scripts/record_motion_proof.py" --headless --mode sweep --num_steps 300 --output_dir "$ROOT_DIR/logs/videos/motion_proof" >"$LOG_DIR/triage_motion_proof.log" 2>&1; then
  pass "motion-proof capture completed"
else
  echo "[WARN] motion-proof capture failed; see $LOG_DIR/triage_motion_proof.log"
fi

echo
echo "== Summary =="
echo "Artifacts:"
echo "- $LOG_DIR/triage_joint_name_diff.log"
echo "- $LOG_DIR/triage_runtime_sanity.log"
echo "- $LOG_DIR/triage_articulation.log"
echo "- $LOG_DIR/articulation_motion.csv"
echo "- $ROOT_DIR/logs/videos/motion_proof/"
