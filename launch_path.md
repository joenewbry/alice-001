# Launch Path: Data & Training Estimates for Alice-001

> HiWonder S1 — 5-DOF arm + half-ball gripper, desktop workspace, basic manipulation tasks.

---

## System Profile

| Parameter | Value |
|-----------|-------|
| Degrees of Freedom | 5 (base, shoulder, elbow, wrist) + 1 gripper |
| Gripper type | Two half-balls (simplified grasp geometry) |
| Workspace | ~30cm radius hemisphere on desktop |
| Control frequency | 10-20 Hz |
| Observation | Joint angles (6-dim) + 1-2 cameras (RGB 128x128) |
| Action space | Joint position deltas (5-dim) + gripper binary (1-dim) |

This is a **low-DOF, constrained-workspace** system — one of the simpler real-world manipulation setups. That works in our favor: data requirements are significantly lower than for 7-DOF arms with dexterous hands.

---

## Data Requirements Summary

### Per Task

| Data Source | Amount | Episodes | Time to Collect |
|-------------|--------|----------|-----------------|
| **Sim demos (scripted/planned)** | 10,000-50,000 episodes | Auto-generated | 2-8 hours (parallelized) |
| **Sim demos (with domain rand)** | 50,000-100,000 episodes | Auto-generated | 4-16 hours (parallelized) |
| **Real-world teleoperation demos** | 50-100 episodes | Manual | 2-4 hours per task |
| **DAgger corrections (per round)** | 10-20 episodes | Manual | 30-60 min per round |
| **DAgger total (3-5 rounds)** | 30-100 additional episodes | Manual | 2-5 hours per task |

### Total Across 5 Starter Tasks

| Data Source | Total Episodes | Total Collection Time |
|-------------|---------------|----------------------|
| Sim (auto-generated) | 250,000-500,000 | 1-3 days (GPU-parallelized) |
| Real teleoperation | 250-500 | 10-20 hours |
| Real DAgger corrections | 150-500 | 10-25 hours |
| **Real-world total** | **400-1,000** | **20-45 hours** |

### Why These Numbers

**Sim data (10,000-100,000 per task):**
- Diffusion Policy and ACT papers show sim pre-training on 1,000-100,000 demos is effective
- More data with domain randomization = better transfer robustness
- 5-DOF is simple enough that 10,000 clean demos may suffice; 50,000+ with randomization adds robustness
- Cost: nearly free (just compute time)

**Real data (50-100 per task):**
- Diffusion Policy achieves 85%+ success with ~50 human demos on similar tabletop tasks
- ACT works well with as few as 25-50 demos for simple pick-and-place
- Half-ball grippers simplify grasping, reducing the variety of grasp strategies needed
- 50 is a minimum; 100 gives a comfortable margin

**DAgger (10-20 per round, 3-5 rounds):**
- Addresses distribution shift — the biggest failure mode of behavioral cloning
- Each round specifically targets the current policy's failure modes
- 3-5 rounds is typically enough to converge for tabletop tasks

---

## Training Compute Estimates

### Per Task

| Training Stage | GPU | Duration | Notes |
|----------------|-----|----------|-------|
| State-based BC in sim | 1x RTX 3090/4090 | 1-2 hours | Fast iteration, no images |
| Vision-based Diffusion Policy in sim | 1x RTX 3090/4090 | 4-8 hours | ResNet encoder + diffusion head |
| Vision-based ACT in sim | 1x RTX 3090/4090 | 3-6 hours | Transformer-based |
| Real-world fine-tuning | 1x RTX 3090/4090 | 30-60 min | Small dataset, few epochs |
| DAgger retrain (per round) | 1x RTX 3090/4090 | 15-30 min | Warm-start from previous |

### Total for 5 Tasks (parallelized)

| Scenario | GPUs | Wall-clock Time |
|----------|------|-----------------|
| Sequential (1 GPU) | 1 | 3-7 days |
| Parallel (5 GPUs, 1 per task) | 5 | 8-16 hours |
| Parallel + sweeps | 8-10 | 12-24 hours |

### Can You Use a Laptop?

- **State-based policy**: Yes — trains on CPU in a few hours, GPU in minutes
- **Vision policy**: Needs a GPU. A laptop 3060/3070 works but 2-3x slower
- **Apple Silicon (M1/M2/M3)**: MPS backend works for PyTorch, ~2x slower than 3090
- **Cloud**: Most cost-effective at scale — rent A100s on Lambda Labs / RunPod (~$1-2/hr)

---

## Sim-to-Real Transfer Strategy

### Why Sim-to-Real Works Here

Our setup is near-ideal for sim-to-real:
1. **Low DOF** — fewer dimensions to get wrong
2. **Simple gripper** — half-balls have predictable contact geometry
3. **Rigid objects** — no deformable items
4. **Constrained workspace** — desktop bounds limit state space
5. **Slow motion** — servo-driven, no high-speed dynamics

### Transfer Gap Sources & Mitigations

| Gap Source | Impact | Mitigation |
|------------|--------|------------|
| Servo response time | High — real servos have lag | Measure and model delay in sim (typically 50-200ms) |
| Joint backlash | Medium — real gears have play | Add random deadzone noise in sim |
| Gripper contact | Medium — half-ball contact differs | Randomize friction, restitution in sim |
| Visual appearance | Low-Medium — sim looks different | Domain randomization on textures, lighting |
| Camera calibration | Low — slight position mismatch | Randomize camera pose ±2cm, ±5deg in sim |
| Table friction | Low-Medium — sliding behavior differs | Randomize friction coefficient ±30% |

### Expected Transfer Performance

| Approach | Expected Success Rate | Data Needed |
|----------|----------------------|-------------|
| Zero-shot (sim only, no randomization) | 10-30% | 0 real demos |
| Zero-shot (sim + domain randomization) | 30-60% | 0 real demos |
| Fine-tuned (50 real demos) | 60-80% | 50 real demos |
| Fine-tuned + DAgger (3 rounds) | 80-95% | 80-130 real demos |

---

## Parallelization Strategy

### Parallel Sim Data Generation

```
Machine/GPU 1: Task A (pick-and-place) — 50,000 episodes
Machine/GPU 2: Task B (push)           — 50,000 episodes
Machine/GPU 3: Task C (stack)          — 50,000 episodes
Machine/GPU 4: Task D (handover)       — 50,000 episodes
Machine/GPU 5: Task E (sort)           — 50,000 episodes
```

Each task's sim data is independent — perfect parallelism. MuJoCo is CPU-bound, so you can also run 8-16 envs per CPU core using vectorized environments.

**Single machine throughput (8-core CPU):**
- ~500-2,000 episodes/hour (depends on episode length)
- 50,000 episodes = 25-100 hours single-threaded, 3-12 hours with 8 workers

### Parallel Training

```
GPU 1: Train pick-and-place Diffusion Policy
GPU 2: Train push Diffusion Policy
GPU 3: Train stack Diffusion Policy
GPU 4: Train handover Diffusion Policy
GPU 5: Train sort Diffusion Policy
GPU 6-10: Hyperparameter sweeps (LR, batch size, architecture)
```

### Parallel Real-World Collection

This is the bottleneck — you only have one robot. Strategies:

1. **Batch by task**: Collect all 50-100 demos for Task A, then B, then C...
2. **Interleave**: Collect 10 demos per task, train, identify weakest task, collect more for that one
3. **AI-prioritized**: Use VLM to evaluate which task needs more data, focus collection there
4. **Multi-camera**: Mount 2-3 cameras, get multiple viewpoints per demo for free
5. **Future**: Buy a second arm and collect for 2 tasks simultaneously

### Parallel Evaluation

After training, evaluate all 5 task policies concurrently in sim (one process per task). Only the best-performing policies move to real-world evaluation.

---

## Using AI to Accelerate Every Stage

### 1. Automated Reward Engineering (Sim)

Instead of hand-crafting reward functions:
- Describe the task to GPT-4 / Claude: "Write a MuJoCo reward function for pick-and-place with a 5-DOF arm"
- Generate 10 reward function variants
- Evaluate each in sim for 1,000 episodes
- Select the one that produces highest-quality demonstrations
- **Speedup: 5-10x faster reward iteration** vs. manual tuning

### 2. Automated Demo Generation (Sim)

- Use LLM-generated motion plans as seed demonstrations
- Use RL (SAC/PPO) with AI-designed rewards to generate optimal demos
- Curriculum: LLM generates progressively harder task variants
  - "Place block at center" → "Place block at edge" → "Place block on small target"
- **Speedup: eliminates manual scripting** of demo trajectories

### 3. Automated Quality Control (Real Data)

- Feed each demonstration video to a VLM (GPT-4V / Claude Vision)
- Ask: "Did the robot successfully pick up the object and place it in the target zone?"
- Auto-label demos as success/failure/partial
- Flag bad demos for removal
- **Speedup: 10-50x faster than manual review** for large batches

### 4. Automated Failure Analysis

After each real-world evaluation:
- Record video of all policy rollouts
- Feed failures to VLM: "Why did this grasp fail?"
- Get structured failure categorization:
  - "Gripper opened too early" → adjust timing
  - "Approach angle too steep" → need more varied demos
  - "Object was outside reach" → expand training distribution
- Use this to prioritize what to fix next
- **Speedup: turns hours of manual analysis into minutes**

### 5. Foundation Model Starting Point

Instead of training from scratch:
- Start from **Octo** (open-source robot foundation model) or **OpenVLA**
- These models were pre-trained on 800,000+ robot episodes across many embodiments
- Fine-tune on Alice-001's embodiment with just **10-20 real demos**
- **Speedup: reduces real-world data needs by 3-10x**
- Caveat: need to verify the model supports 5-DOF action spaces (may need adapter)

### 6. Synthetic Data Augmentation

- Use image generation models to augment visual observations:
  - Change background textures
  - Add/remove distractor objects
  - Vary lighting conditions
- Use trajectory augmentation:
  - Mirror demos (left ↔ right)
  - Add Gaussian noise to actions, check if still successful in sim
  - Time-stretch demos (faster/slower)
- **Speedup: 2-5x effective dataset size from same collection effort**

### 7. LLM-Guided Curriculum for Multi-Task

- Have LLM design training curriculum:
  - "Given these 5 tasks sorted by difficulty, suggest training order and data mixing ratios"
  - "The push task has 40% success, pick-and-place has 85%. Recommend next steps."
- Auto-adjust training data mixing based on per-task performance
- **Speedup: faster convergence** to good multi-task performance

---

## Timeline Estimate

### Milestone 1: Single Task Working in Sim (Weeks 1-2)

| Step | Time | Parallel? |
|------|------|-----------|
| URDF model of arm | 2-3 days | No |
| MuJoCo environment + Gym wrapper | 2-3 days | After URDF |
| Domain randomization | 1-2 days | After env |
| Scripted demo generation (10,000 eps) | 0.5 days | After env |
| Train state-based Diffusion Policy | 0.5 days | After data |
| Train vision-based Diffusion Policy | 1 day | After state-based |
| Achieve >85% sim success rate | Included above | — |

### Milestone 2: Sim-to-Real Transfer (Weeks 3-4)

| Step | Time | Parallel? |
|------|------|-----------|
| Camera mounting + calibration | 1 day | — |
| Teleoperation interface | 2-3 days | Parallel with camera |
| Collect 50 real demos (pick-and-place) | 1 day | After teleop |
| System identification (servo delays) | 1 day | Parallel with demos |
| Fine-tune sim policy on real data | 0.5 days | After data |
| Evaluate on real robot | 0.5 days | After fine-tune |
| DAgger rounds (3x) | 2-3 days | Sequential |

### Milestone 3: 5 Tasks Working (Weeks 5-8)

| Step | Time | Parallel? |
|------|------|-----------|
| Sim envs for 4 more tasks | 3-4 days | Parallel (1 day each) |
| Sim data generation (4 tasks) | 1 day | Fully parallel |
| Sim training (4 tasks) | 1-2 days | Parallel on GPUs |
| Real demos (50 each × 4 tasks) | 4-5 days | Sequential (1 robot) |
| Fine-tuning + DAgger (4 tasks) | 4-6 days | Training parallel, collection sequential |
| Integration with voice system | 2-3 days | After policies work |

### Milestone 4: Production Deployment (Weeks 9-10)

| Step | Time | Parallel? |
|------|------|-----------|
| ONNX export + inference optimization | 1-2 days | — |
| Voice command → policy trigger integration | 1-2 days | After export |
| Safety testing + edge cases | 2-3 days | After integration |
| Multi-task evaluation (20 trials each) | 1 day | — |

### Total: ~10 weeks to 5 working tasks

Aggressive timeline assumes:
- 1 person working ~full time
- 1 GPU available (cloud is fine)
- No major hardware issues

---

## Data Storage Estimates

| Data Type | Per Episode | Per Task (50k sim + 100 real) | 5 Tasks |
|-----------|-------------|-------------------------------|---------|
| State-only demos | ~10 KB | 500 MB | 2.5 GB |
| Vision demos (128x128) | ~5 MB | 250 GB (sim) + 500 MB (real) | 1.25 TB |
| Vision demos (64x64) | ~1.5 MB | 75 GB (sim) + 150 MB (real) | 375 GB |
| Model checkpoints | ~200 MB | 1-2 GB (with sweeps) | 5-10 GB |

**Recommendation**: Start with state-based policies (joint angles only) to validate the pipeline. Add vision later. This cuts storage by 100-500x and training time by 5-10x.

---

## Cost Estimates

| Item | Cost | Notes |
|------|------|-------|
| Cloud GPU (A100, 100 hrs) | $150-200 | Lambda Labs / RunPod |
| OpenAI API (reward eng + quality control) | $20-50 | GPT-4 calls for automation |
| Camera(s) | $30-100 | USB webcam, 1080p fine |
| Teleoperation device | $0-60 | Keyboard free; gamepad ~$30; 3D mouse ~$60 |
| Additional objects for tasks | $10-30 | Wooden blocks, small cups |
| **Total** | **~$200-450** | Excluding the robot itself |

---

## Risk Factors & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Servo backlash too high for precise placement | Medium | High | Use force-based grasp detection; increase placement target tolerance |
| Half-ball gripper can't grasp target objects | Low | High | Test gripper on actual objects early (Phase 0); choose grippable objects |
| Sim-to-real visual gap too large | Medium | Medium | Start with state-based policy (no vision); add vision later |
| Not enough GPU for training | Low | Medium | Use cloud; state-based policies train on CPU |
| Teleoperation demos are low quality | Medium | Medium | Use VLM quality control; re-collect bad demos |

---

## Quick Reference: What to Do First

1. **Today**: Order a USB webcam if you don't have one
2. **This week**: Build the MuJoCo URDF model of the arm
3. **This week**: Get 5 manual demonstrations recorded to understand the task difficulty
4. **Next week**: Scripted sim demos + first Diffusion Policy training run
5. **Week 3**: First real-world fine-tuning attempt

The single most impactful thing is getting the sim environment working. Everything else builds on top of it.
