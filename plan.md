# Alice-001 Training Loop Plan

> Goal: Build a sim-to-real training pipeline for the HiWonder S1 5-DOF arm with half-ball grippers, enabling learned manipulation policies for basic tasks.

---

## Phase 0: Audit Current System & Define Tasks

- [ ] Inventory current rule-based commands and their success rate on hardware
- [ ] Define the 3-5 starter tasks for learned policies:
  - [ ] **Pick and place** — grasp object at A, place at B
  - [ ] **Push** — push object from A to B
  - [ ] **Stack** — place one object on top of another
  - [ ] **Handover** — pick up object, move to handoff zone
  - [ ] **Sort** — move objects to correct bins by color/shape
- [ ] Measure and document the physical workspace envelope (reachable area, table dims)
- [ ] Characterize the half-ball gripper: max grip diameter, grip force, slip behavior
- [ ] Record 5-10 teleoperated demonstrations by hand to establish baseline task difficulty
- [ ] Decide on observation space:
  - [ ] Joint angles (5 values) — from servos
  - [ ] Gripper state (1 value) — open/close percentage
  - [ ] Wrist camera RGB (if adding one) — 224x224 or 128x128
  - [ ] Top-down camera RGB (if adding one) — for workspace overview
- [ ] Decide on action space:
  - [ ] Joint velocity (5-dim continuous) vs. delta joint position vs. end-effector delta
  - [ ] Gripper open/close (1-dim binary or continuous)

---

## Phase 1: Simulation Environment Setup

- [ ] Choose sim framework:
  - [ ] **MuJoCo** (recommended — fast, accurate contact physics, free, great ecosystem)
  - [ ] Alternative: PyBullet, Isaac Sim (if GPU-rich)
- [ ] Model the HiWonder S1 arm in URDF/MJCF:
  - [ ] Measure link lengths, masses, joint limits from physical robot
  - [ ] Model the half-ball gripper geometry (two hemispheres)
  - [ ] Validate forward kinematics against real arm (< 5mm error at end-effector)
- [ ] Model the workspace:
  - [ ] Desk surface with friction
  - [ ] Target objects (cubes, cylinders, small items the gripper can handle)
  - [ ] Camera viewpoints matching where real cameras will go
- [ ] Build a Gymnasium (OpenAI Gym) compatible environment wrapper:
  - [ ] `reset()` — randomize object positions, arm starting pose
  - [ ] `step(action)` — apply action, return obs, reward, done, info
  - [ ] `render()` — RGB rendering for visual policy training
- [ ] Implement domain randomization:
  - [ ] Object mass, friction, size (uniform ±20%)
  - [ ] Lighting color and direction
  - [ ] Camera position jitter (±2cm, ±5deg)
  - [ ] Table texture randomization
  - [ ] Joint friction/damping variation
  - [ ] Action delay/noise (simulating real servo latency)
- [ ] Implement task-specific reward functions:
  - [ ] Pick-and-place: sparse (1 if placed correctly) + shaped (distance to object, distance to goal)
  - [ ] Push: distance of object to target
  - [ ] Stack: height of stacked objects
- [ ] Validate sim: run the existing rule-based commands in sim, compare to real behavior
- [ ] Set up parallel environment runner (vectorized envs with `gymnasium.vector`)

---

## Phase 2: Data Collection Infrastructure

### Sim Data Collection

- [ ] Build automated demonstration generator:
  - [ ] Use motion planning (RRT/RRT*) or scripted policies to generate expert trajectories
  - [ ] Use AI-assisted reward shaping: have an LLM generate reward function variants, evaluate which produces best demos
  - [ ] Target: **50,000 demonstrations per task** in sim (with domain randomization)
- [ ] Record demonstration format (standardize early):
  - [ ] `observations`: joint angles, gripper state, camera images (if visual)
  - [ ] `actions`: joint deltas or velocities + gripper command
  - [ ] `rewards`: per-step reward signal
  - [ ] `metadata`: task ID, domain randomization params, success/fail
  - [ ] Storage format: HDF5 or Zarr (chunked, compressed)
- [ ] Build data pipeline:
  - [ ] Write demonstrations to disk in parallel across sim workers
  - [ ] Data validation: filter out failed/corrupted episodes
  - [ ] Compute dataset statistics (mean, std for normalization)
- [ ] Set up data versioning (DVC or Weights & Biases Artifacts)

### Real-World Data Collection

- [ ] Build a teleoperation interface:
  - [ ] Option A: **Leader-follower** — second arm mirrors movements (best quality)
  - [ ] Option B: **Keyboard/gamepad** — manual joint control
  - [ ] Option C: **VR controller** — 6-DOF tracking mapped to end-effector
  - [ ] Option D: **Kinesthetic teaching** — physically move the arm while recording
  - [ ] **AI-assisted**: Use a VLM (GPT-4V / Claude vision) to auto-label task stages and detect errors in demos
- [ ] Record real-world demonstrations:
  - [ ] Mount camera(s) in same position as sim
  - [ ] Record joint angles at 10-30 Hz (match sim frequency)
  - [ ] Record camera frames synchronized with joint data
  - [ ] Target: **50-100 demonstrations per task** for initial fine-tuning
- [ ] Build real-world data quality checker:
  - [ ] Auto-detect gripper slip, missed grasps, out-of-bounds
  - [ ] Use VLM to classify demo quality (good/marginal/bad)
  - [ ] Flag demos where object wasn't successfully manipulated

---

## Phase 3: Training Pipeline

### Architecture Selection

- [ ] Evaluate and select policy architecture:
  - [ ] **Diffusion Policy** (recommended for manipulation — state of the art, works with ~50 demos)
  - [ ] **ACT (Action Chunking Transformers)** — great with small real-world datasets
  - [ ] **BC-Z / RT-1 style** — if going multi-task
  - [ ] **OpenVLA / Octo** — foundation model fine-tuning (if want to leverage pre-training)
- [ ] Implement training loop:
  - [ ] Dataloader for HDF5/Zarr demonstration data
  - [ ] Policy network (takes obs → outputs action chunk)
  - [ ] Loss function (MSE for BC, diffusion loss for diffusion policy)
  - [ ] Optimizer (AdamW, cosine LR schedule)
  - [ ] Validation split (80/20)
  - [ ] Checkpoint saving and evaluation hooks
- [ ] Set up experiment tracking (Weights & Biases or MLflow)

### Sim Pre-training

- [ ] Train base policy on sim data:
  - [ ] Start with state-based policy (joint angles only — faster iteration)
  - [ ] Then train vision-based policy (camera images → actions)
  - [ ] Batch size: 256-1024 (state), 64-128 (vision)
  - [ ] Training: ~50-200 epochs or until validation loss plateaus
  - [ ] Evaluate in sim: measure success rate across 100 rollouts per task
  - [ ] Target: **>85% success rate in sim** before moving to real
- [ ] Parallelize training across tasks:
  - [ ] One GPU per task (or time-share on single GPU)
  - [ ] Shared encoder, task-specific heads (if multi-task)
  - [ ] Hyperparameter sweep with Optuna or W&B Sweeps
- [ ] AI-accelerated training:
  - [ ] Use LLMs to auto-generate curriculum (easy → hard task variants)
  - [ ] Use VLMs to auto-evaluate policy rollouts in sim (did it succeed?)
  - [ ] Use code-generation AI to iterate on reward functions

### Sim-to-Real Transfer

- [ ] Domain adaptation strategy:
  - [ ] **Domain randomization** (primary — randomize visual and physics params in sim)
  - [ ] **System identification** — measure real robot servo response times, friction, backlash and match in sim
  - [ ] **Action space calibration** — map sim joint deltas to real servo commands
- [ ] Run sim policy on real robot (zero-shot transfer test):
  - [ ] Measure success rate
  - [ ] Record failure modes (overshoot, miss-grasp, collision)
  - [ ] Log all rollouts for analysis
- [ ] If zero-shot transfer < 50% success, proceed to fine-tuning

---

## Phase 4: Real-World Fine-Tuning

- [ ] Fine-tune sim-pretrained policy on real demonstrations:
  - [ ] Load sim-pretrained weights
  - [ ] Train on 50-100 real demos with lower learning rate (0.1x sim LR)
  - [ ] Training: 20-50 epochs (small dataset, careful not to overfit)
  - [ ] Use EMA (exponential moving average) for stable policy
- [ ] DAgger (Dataset Aggregation) loop — iterative improvement:
  - [ ] Deploy current policy on real robot
  - [ ] Have human correct failures (intervene and demonstrate)
  - [ ] Add corrective demonstrations to training set
  - [ ] Retrain policy
  - [ ] Repeat 3-5 rounds
  - [ ] Target: **10-20 new demos per DAgger round**
- [ ] AI-assisted real-world data augmentation:
  - [ ] Use image augmentation (crop, color jitter, blur) on visual observations
  - [ ] Use trajectory perturbation (add noise to successful demos, relabel)
  - [ ] Use a VLM to generate synthetic language instructions for demos
- [ ] Evaluate on held-out test scenarios:
  - [ ] 20 trials per task, measure success rate
  - [ ] Target: **>80% success rate on basic tasks**
- [ ] Save final model checkpoints and deployment configs

---

## Phase 5: Deployment & Integration

- [ ] Integrate learned policy with existing voice-control system:
  - [ ] Add new command category: "autonomous" tasks (e.g., "pick up the block")
  - [ ] Voice command triggers policy inference instead of rule-based motion
  - [ ] Keep rule-based commands for direct joint control
- [ ] Build inference pipeline:
  - [ ] Model runs on-device (Raspberry Pi / Jetson) or streams to laptop
  - [ ] Inference latency target: < 50ms per action (10-20 Hz control)
  - [ ] ONNX export or TorchScript for deployment
- [ ] Safety wrapper:
  - [ ] Joint limit enforcement (already exists — reuse)
  - [ ] Force/torque limits
  - [ ] Workspace boundary enforcement
  - [ ] Emergency stop integration (already exists)
- [ ] Monitoring:
  - [ ] Log all policy actions and outcomes
  - [ ] Detect policy failures automatically (VLM-based or heuristic)
  - [ ] Alert if success rate drops below threshold

---

## Phase 6: Scaling & Multi-Task

- [ ] Add new tasks by collecting 50-100 real demos + DAgger
- [ ] Train multi-task policy:
  - [ ] Shared vision encoder + language-conditioned heads
  - [ ] Use voice command text as task conditioning input
  - [ ] "Pick up the red block" → policy selects correct behavior
- [ ] Continuous learning loop:
  - [ ] Every successful real-world execution → add to training set
  - [ ] Periodic retraining (weekly batch)
  - [ ] Track per-task success rate over time
- [ ] Explore foundation model fine-tuning:
  - [ ] Fine-tune OpenVLA or Octo on Alice-001's embodiment
  - [ ] Leverage pre-trained manipulation knowledge
  - [ ] May reduce real-world data needs to ~10-20 demos per task
