# Alice-001 Motion Recovery Report (Attempt 2)

Date: 2026-02-25
VM: alice-isaaclab-2 (us-central1-a)

## 1) Baseline + checkpoints
- No git repo is present in `~/Alice-001` (or local mirror), so commit/tag was not possible.
- Snapshot equivalents created:
  - `logs/recovery/alice001-first-attempt_20260225_071042.tar.gz`
  - `logs/recovery/alice001-attempt2-progress_20260225_072737.tar.gz`

## 2) Runtime repair status
### Changes applied
- Removed stale Fabric extension from active ext dir and moved it to backup:
  - moved: `.../exts/3/omni.physx.fabric-106.3.2+106.3.0.lx64.r.cp310.ub3f.disabled`
  - backup: `~/omni_ext_backup/`
- Added runtime sanity script:
  - `scripts/runtime_sanity_check.sh`

### Result
- `scripts/runtime_sanity_check.sh` PASS
- No Fabric interface mismatch seen in startup probe after cleanup.

## 3) USD rebuild status
- Attempted IsaacLab URDF importer rebuild path; blocked by package path issue:
  - `ModuleNotFoundError: isaacsim.asset`
- Attempted offline converter path; blocked unless started inside SimApp because `pxr` is unavailable in plain venv.
- Versioned rebuild file exists but is byte-identical to current asset:
  - `usd/alice_001.rebuild_v2.usd`
  - SHA256 (same as active): `5360329d420e9952763246f530c466927d1f9d6e23c99dc25d948235b8e7b6d9`

## 4) Deterministic articulation gates
Added script:
- `scripts/validate_articulation_motion.py`

Output:
- CSV: `logs/recovery/articulation_motion.csv` (930 rows)

Measured ranges:
- `joint_base`: 0.74993 rad
- `joint_shoulder`: 0.75000 rad
- `joint_elbow`: 0.75000 rad
- `joint_wrist_pitch`: 0.77550 rad
- `joint_wrist_roll`: 0.56084 rad
- `joint_left_finger`: 0.43001 rad
- `joint_right_finger`: 0.45833 rad
- `ee_disp_norm`: 0.13250 m

Gate verdict:
- Runtime gate: PASS
- Joint-range gate: PASS
- EE-displacement gate: PASS

## 5) Video artifacts
Available:
- `logs/videos/joint_sweep.mp4`
- `logs/videos/triage_policy_short/overhead.mp4`
- `logs/videos/triage_policy_vision_short/overhead.mp4`
- `logs/videos/triage_policy_vision_short/wrist_camera.mp4`

Note:
- Overhead/wrist framing can still look visually subtle despite confirmed motion telemetry.
- A longer `motion_proof` recorder attempt was unstable in this VM rendering path and was aborted.

## 6) New tooling added
- `scripts/runtime_sanity_check.sh`
- `scripts/validate_articulation_motion.py`
- `scripts/record_motion_proof.py` (experimental; rendering path unstable on this VM)

## 7) Current conclusion
- Robot articulation in sim is moving (confirmed numerically with large margins).
- Remaining confidence gap is visual-proof quality under this VM’s headless rendering path, not kinematic immobility.

## 8) Recommended next step
If visual-proof remains blocker, proceed to Attempt 3:
- Clean Isaac Sim/IsaacLab reinstall with pinned package set, then rerun the same gates and video capture scripts.
