# Alice-001 Consolidated Triage Report

Date: 2026-02-25
VM: alice-isaaclab-2 (us-central1-a)

## Overall verdict
- Articulation is moving in simulation (numerically verified).
- The main blocker is rendering/visualization reliability on this VM, not joint immobility.

## Step results

1. URDF vs USD structural check: WARN
- Artifact: `logs/recovery/triage_joint_name_diff.log`
- Difference found:
  - USD has `fixed_base_joint` that is not in URDF list.
- Interpretation: not a kinematic showstopper by itself; likely an importer-added fixed base joint.

2. Runtime sanity (fabric/import path): PASS
- Artifact: `logs/recovery/triage_runtime_sanity.log`
- Checks passed:
  - `isaacsim`, `isaaclab`, `isaaclab_rl` imports
  - single active fabric extension
  - stale `omni.physx.fabric-106.3.*` absent
  - no `IPhysxFabric` interface-mismatch errors in startup probe

3. Deterministic articulation gates: PASS
- Artifacts:
  - `logs/recovery/triage_articulation.log`
  - `logs/recovery/articulation_motion.csv`
- Measured ranges (radians):
  - `joint_base`: 0.75040
  - `joint_shoulder`: 0.75000
  - `joint_elbow`: 0.75000
  - `joint_wrist_pitch`: 0.75044
  - `joint_wrist_roll`: 0.60910
  - `joint_left_finger`: 0.29689
  - `joint_right_finger`: 0.41569
- End-effector displacement norm: 0.13238 m

4. Motion-proof video: PASS (artifact), but VM capture path is unstable under repeated runs
- Artifacts:
  - `logs/videos/motion_proof/sweep_overhead.mp4`
  - `logs/videos/motion_proof/sweep_summary.json`
- Local frame-change check indicates non-static output on this artifact.
- Note: repeated re-runs on VM can stall due Kit/render process instability; existing artifact is valid proof run.

## Root cause summary
The “no movement” perception came from a stack of issues:
- scripts were sometimes launched outside IsaacLab runtime (`ModuleNotFoundError: isaaclab`),
- an earlier shell triage command was malformed,
- prior video captures were often static-looking due rendering/camera path behavior,
- VM render path is flaky; visual output can freeze even while physics/telemetry advances.

## Recommended immediate workflow
1. Use `scripts/run_motion_triage.sh` as the canonical health check.
2. Treat `articulation_motion.csv` as ground truth gate for “is robot moving”.
3. Use motion-proof video as a secondary signal only.
4. If stable visual QA is required, proceed with clean Isaac Sim reinstall/pin on VM.
