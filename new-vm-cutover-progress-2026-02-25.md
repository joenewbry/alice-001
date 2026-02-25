# New VM Cutover Progress (2026-02-25)

## Completed
- Stopped old VM: `alice-isaaclab-2` (cost control).
- Created new VM: `alice-isaaclab-3` (`g2-standard-8`, `nvidia-l4`, `200GB`, `us-central1-a`).
- Synced clean project source to `~/Alice-001`.
- Installed base toolchain: `git`, `git-lfs`, `ffmpeg`, `python3.10-venv`.
- Installed Isaac Sim pip stack in `~/isaaclab_venv`:
  - `isaacsim[all,extscache]==4.5.0`
- Installed IsaacLab source stack + project package:
  - `isaaclab`, `isaaclab_tasks`, `isaaclab_rl` (editable)
  - `rl_task` package (editable)
- Resolved missing runtime shared libs encountered during bring-up:
  - `libsm6`, `libxext6`, `libxrender1`, `libgl1`, `libglib2.0-0`, `libxt6`, `libglu1-mesa`
- Upgraded VM runtime to new kernel via reboot:
  - now `6.8.0-1048-gcp`
- Installed Vulkan/driver userspace stack missing from base image:
  - `nvidia-driver-570-server`, `libnvidia-gl-570-server`, `mesa-vulkan-drivers`, `vulkan-tools`
- Verified Vulkan sees NVIDIA L4 after fix (`vulkaninfo --summary`).

## Current blocker
Camera-enabled IsaacLab runs on `alice-isaaclab-3` are still not completing to first simulation-step output in our scripts (`test_env.py`, `record_motion_proof.py`).

Observed behavior:
- Startup reaches Kit initialization and Vulkan device enumeration.
- In non-X mode: repeated GLFW/display warnings and startup stalls.
- In `xvfb-run` mode: X server is detected and Vulkan device is detected, but execution still stalls before script-level completion (no generated video artifact yet).

## Why this blocks requested deliverables
- The requested outputs require camera-rendered videos (ball drop + 10 demos + collapse check video).
- While physics stack now initializes much further than before, rendering-path startup does not consistently complete to produce mp4 artifacts on this new VM yet.

## Artifacts collected locally
- `logs/newvm/isaacsim_pip_install.log`
- `logs/newvm/isaaclab_install.log`
- `logs/newvm/isaaclab_core_install.log`
- `logs/newvm/test_env_smoke_newvm.log`
- `logs/newvm/newvm_motion_smoke.log`
- `logs/newvm/newvm_motion_xvfb.log`

## Next concrete fix to execute
1. Force a software/egl-safe renderer path in launch args for smoke scripts (explicit non-RTX path) and re-test with `xvfb`.
2. If still stalled, pin to the exact image family used by the previously working VM instead of generic DLVM image and rerun bring-up.
3. Once one short camera run succeeds, execute the 10-demo batch and copy videos local immediately.
