# Alice-001 Isaac Sim Triage Report (Feb 25, 2026)

## Executive summary
- CAD -> USD conversion is structurally valid.
- Primary failure was Isaac Sim extension/version mismatch in Fabric plugin loading on VM.
- After removing stale extension version, Fabric interface crash was cleared.
- Joint-sweep script had a bug (`raw_env._robot`) and was fixed to `raw_env.robot`.
- Robot motion is present in simulation (confirmed by joint telemetry), but current overhead videos make motion hard to perceive due camera framing/scale.

## What was verified
1. URDF/USD structure consistency
- Joint names match exactly: 7 revolute joints.
- Link/rigid-body count matches: 8.

2. Runtime/plugin root cause on VM
- Error before fix:
  - `IPhysxPrivate v0.2 requested` vs `v1.2 available`
  - `RuntimeError: Failed to acquire interface: omni::physx::IPhysxFabric`
- Root issue: mixed extension versions under Isaac Sim ext directory, including stale `omni.physx.fabric-106.3.2`.

3. Fixes applied
- VM extension cleanup:
  - moved stale `omni.physx.fabric-106.3.2...` out of active ext path into `~/omni_ext_backup/`.
- Script bug fix on VM:
  - `~/Alice-001/rl_task/scripts/joint_test.py`: `_robot` -> `robot`.
- Local code fixes:
  - `rl_task/scripts/joint_test.py` `_robot` -> `robot`.
  - `rl_task/alice_ball_transfer/ball_transfer_env_cfg.py` USD path made portable via `Path(__file__).resolve()`.

## Motion evidence (policy run telemetry)
Source CSV:
- `~/Alice-001/logs/videos/triage_policy_short/eval_metrics.csv`

Observed ranges over 240 steps:
- `joint_base`: range `0.30923 rad`
- `joint_shoulder`: range `0.22468 rad`
- `joint_elbow`: range `0.59522 rad`
- `joint_wrist_pitch`: range `0.67919 rad`
- `joint_wrist_roll`: range `0.36516 rad`
- `joint_left_finger`: range `0.42662 rad`
- `joint_right_finger`: range `0.47694 rad`
- EE position changed by up to ~2.4 cm in XY, ~1.2 cm in Z.

Conclusion: joints are moving materially in sim.

## Recorded outputs
VM paths:
- Joint sweep (overhead): `~/Alice-001/logs/videos/joint_sweep.mp4`
- Policy short (state): `~/Alice-001/logs/videos/triage_policy_short/overhead.mp4`
- Policy short (vision):
  - overhead: `~/Alice-001/logs/videos/triage_policy_vision_short/overhead.mp4`
  - wrist: `~/Alice-001/logs/videos/triage_policy_vision_short/wrist_camera.mp4`

Local copied files:
- `/Users/joe/dev/Alice-001/logs/videos/joint_sweep.mp4`
- `/Users/joe/dev/Alice-001/logs/videos/triage_policy_vision_short_wrist.mp4`

## Why videos can look static even when sim moves
- Overhead view is wide and workspace is small; cm-scale motion can be visually subtle.
- Compression + downscaled render can hide tiny motion.
- Wrist camera is better for visible motion confirmation.

## Next recommended action
- Add a tighter side camera or overlay telemetry text on frames for unambiguous visible motion in every diagnostic video.
