#!/usr/bin/env bash
set -euo pipefail

VENV=~/isaaclab_venv
ISAACLAB=~/IsaacLab/isaaclab.sh
EXT_DIR=~/isaaclab_venv/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/exts/3

fail() { echo "[FAIL] $1"; exit 1; }
pass() { echo "[PASS] $1"; }

[ -d "$VENV" ] || fail "venv missing at $VENV"
[ -f "$ISAACLAB" ] || fail "isaaclab.sh missing at $ISAACLAB"
[ -d "$EXT_DIR" ] || fail "extension dir missing at $EXT_DIR"

source "$VENV/bin/activate"
python - <<'PY' || fail "Python imports failed"
import importlib.util
mods=["isaacsim","isaaclab","isaaclab_rl"]
for m in mods:
    if not importlib.util.find_spec(m):
        raise RuntimeError(f"missing module: {m}")
print("python imports ok")
PY
pass "Core python modules import"

fabric_count=$(ls -1 "$EXT_DIR" | grep -E '^omni\.physx\.fabric-' | wc -l | tr -d ' ')
[ "$fabric_count" -eq 1 ] || fail "Expected exactly 1 active omni.physx.fabric extension, found $fabric_count"
pass "Single active fabric extension"

if ls -1 "$EXT_DIR" | grep -E '^omni\.physx\.fabric-106\.3\.' >/dev/null; then
  fail "Stale 106.3 fabric extension still present in active ext dir"
fi
pass "No stale fabric 106.3 extension in active dir"

timeout 45 "$ISAACLAB" -p ~/Alice-001/rl_task/scripts/joint_test.py --num_envs 1 --headless >/tmp/runtime_sanity_joint_test.log 2>&1 || true
if grep -E "IPhysxFabric|IPhysxPrivate v0.2|Failed to acquire interface" /tmp/runtime_sanity_joint_test.log >/dev/null; then
  echo "---- joint_test excerpt ----"
  grep -nE "IPhysxFabric|IPhysxPrivate v0.2|Failed to acquire interface" /tmp/runtime_sanity_joint_test.log || true
  fail "Fabric interface mismatch persists"
fi
pass "No fabric interface mismatch in startup probe"

echo "[PASS] runtime_sanity_check complete"
