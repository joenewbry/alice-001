# Alice-001 Training Report — February 24, 2026

## Overview

Overnight training run of the Alice-001 sim-to-real pipeline on GCP VM `alice-isaaclab-2` (NVIDIA L4 GPU, 24GB VRAM, us-central1-a). Two stages completed: State-only RL (Stage 1) and Vision RL (Stage 2, all 3 curriculum phases).

**Total training time**: ~13 hours
**Total checkpoints**: 50 (26 state + 24 vision)
**VM cost**: ~$13 (g2-standard-8 @ ~$1/hr)

---

## Stage 1: State-Only RL

**Config**: 256 envs, PPO, 28-dim obs (joint pos/vel + EE pos + ball pos + distances), 7-dim actions
**Duration**: ~2 hours, 5750 iterations, ~1.3s/iter
**Checkpoint**: `model_5749.pt`

### Final Metrics
| Metric | Value |
|--------|-------|
| Grasping rate | **95.8%** |
| Grasp reward | **0.93** |
| Lift rate | 0% |
| EE-to-ball distance | ~1.8cm |
| Mean reward/step | 12.5 |

### Evaluation (600 steps, 4 envs)
- Converges to ball within ~30 steps
- Maintains contact and closed gripper (grasp_reward = 1.0)
- No lift behavior — arm reaches and grasps but doesn't lift

---

## Stage 2: Vision RL

**Config**: 32 envs (limited by L4 descriptor sets), 4-frame stacked 224x224 wrist camera, frozen ResNet-18 backbone (512-dim features), asymmetric actor-critic (actor: 526-dim vision+proprio, critic: 28-dim privileged state)

### Phase 1: Reach Only (1500 iterations, 1.7 hours)
- Learns to visually locate ball and move EE toward it
- Best reach: 2.3cm EE-to-ball distance
- Emergent grasping: 19% (not rewarded, but happens naturally)

### Phase 2: Reach + Grasp (1500 iterations, 3.5 hours)
- Grasp reward activated; lift/transport/drop disabled
- Grasping peaks at 57%, then oscillates 20-40%
- Reach stabilizes at 2-3cm
- Gripper closing remains inconsistent (sparse reward challenge)

### Phase 3: Full Pipeline (6000 iterations, 7.7 hours)
- All rewards enabled (reach + grasp + lift + transport + drop)
- Late breakthrough: grasp_reward finally appears at iteration ~9000
- Best grasping: **82.3%** at 1.35cm distance

### Final Vision Metrics (model_10000)
| Metric | Value |
|--------|-------|
| Grasping rate | **78%** |
| Grasp reward | **0.126** |
| Lift rate | 0% |
| EE-to-ball distance | ~1.6cm |
| Mean reward/step | ~2.4 |

### Evaluation (600 steps, 4 envs, model_10000)
- Wrist camera video saved (`logs/videos/model_10000/wrist_camera.mp4`)
- Reaches ball within ~50 steps from camera vision alone
- Maintains proximity (1.0cm by end of episode)
- Grasping at 100% by episode end
- No lift behavior

---

## Comparison: State vs Vision

| Metric | State (5749) | Vision (10000) |
|--------|-------------|----------------|
| EE-to-ball | 1.8cm | 1.6cm |
| Grasping | 95.8% | 78% |
| Grasp reward | 0.93 | 0.126 |
| Lift | 0% | 0% |
| Reward/step | 12.5 | 2.4 |
| Training time | 2h | 13h |

The vision policy achieves comparable reach precision but weaker grasping than the state policy, which is expected given the harder observation space (raw pixels vs privileged state).

---

## Technical Issues & Fixes

1. **GCP zone exhaustion**: T4 GPUs unavailable in us-west1-b. Migrated via disk snapshot to us-central1-a with L4 GPU.

2. **libGLU.so.1 missing**: Headless camera rendering requires `libglu1-mesa`. Installed with apt-get.

3. **RTX descriptor set exhaustion**: 128 camera environments overflows the L4's descriptor pool. Reduced to 32 envs.

4. **usdrt.hierarchy missing**: Isaac Sim 4.5's `usdrt` module lacks the `hierarchy` attribute needed by Fabric mode. Fixed with `use_fabric=False` in SimulationCfg.

5. **RSL-RL resume bug**: Config-based resume (`load_run`/`resume`) doesn't work reliably. Use `runner.load(checkpoint_path)` directly.

6. **Checkpoint sort bug**: Lexicographic sort puts `model_750.pt` after `model_5000.pt`. Fixed with numeric key extraction.

---

## Key Observations

### Why No Lift?
Neither the state nor vision policy learned to lift the ball. Likely causes:
- **Gravity disabled**: The sim runs with gravity=0 (PhysX drives can't hold this lightweight arm). Without gravity, "lifting" doesn't have the same meaning — the ball doesn't fall when released.
- **Reward shaping**: The lift reward requires `ball_lifted AND has_contact`, which is a conjunction of two conditions. The policy needs to discover both simultaneously.
- **Kinematic mode**: The arm uses position targets, not torque. The gripper may not generate enough holding force in the sim to reliably lift.

### Vision Policy Challenges
- **32 envs**: Far fewer than the 256 used for state RL, leading to higher variance per iteration
- **Sparse grasp reward**: Requires 3 simultaneous conditions (contact + gripper closed + proximity). Hard to discover with limited parallel exploration.
- **Late learning**: Grasp reward only appeared at iteration ~9000 of Phase 3, suggesting the policy needed many iterations to stumble on the right gripper closing behavior.

---

## Files & Artifacts

### Checkpoints (downloaded locally)
- `logs/alice_ball_transfer/model_5749.pt` — Stage 1 final (state-only)
- `logs/alice_ball_transfer_vision/model_0.pt` through `model_10497.pt` — Stage 2 all phases

### Evaluation
- `logs/videos/model_10000/wrist_camera.mp4` — Vision policy wrist camera view
- `logs/videos/model_10000/eval_metrics.csv` — Vision policy step-by-step metrics
- `logs/videos/model_5749/eval_metrics.csv` — State policy step-by-step metrics

### VM Status
- `alice-isaaclab-2` (us-central1-a): **TERMINATED**
- `alice-isaaclab` (us-west1-b): **TERMINATED**

---

## Next Steps

1. **Fix lift reward**: Investigate gravity/force model. May need to enable gravity with compensating joint stiffness, or change lift detection to use ball displacement rather than height.
2. **Domain randomization (Stage 3)**: Run with `--domain_rand` flag using the current best vision checkpoint as starting point.
3. **ONNX export**: Export vision policy to ONNX for TensorRT deployment on Jetson.
4. **Hardware setup**: Connect xArm 1S + wrist camera to Prometheus (Jetson Orin Nano) for real-world deployment testing.
5. **Fine-tune on real data**: Use teleop to collect 50-100 real demonstrations, then fine-tune with Octo or the overnight learning loop.
