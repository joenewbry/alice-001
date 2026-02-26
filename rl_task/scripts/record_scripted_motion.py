"""Record a scripted motion proof: arm moves ball from source to target.

Uses the optimal joint config found by sweep_workspace.py to interpolate
a smooth trajectory, demonstrating physics, kinematics, and camera rendering.
"""
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_frames", type=int, default=300)
parser.add_argument("--output_dir", type=str, default="/home/joe/motion_proof")
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import os
import torch
import numpy as np
from PIL import Image

from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from alice_ball_transfer.ball_transfer_env_cfg import (
    ALICE_001_CFG, BALL_CFG, PEDESTAL_CFG, TABLE_CFG, BallTransferEnvCfg,
)

# ── Setup sim ──
sim_cfg = SimulationCfg(
    dt=1/120, gravity=(0, 0, -9.81), use_fabric=False,
    render_interval=1,
)
sim = sim_utils.SimulationContext(sim_cfg)

# Ground + light
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)
light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
light.func("/World/Light", light)

# Scene
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

# Robot
robot = Articulation(ALICE_001_CFG)
scene.articulations["robot"] = robot

# Table
table = RigidObject(TABLE_CFG)
scene.rigid_objects["table"] = table

# Pedestal
pedestal = RigidObject(PEDESTAL_CFG)
scene.rigid_objects["pedestal"] = pedestal

# Ball
ball = RigidObject(BALL_CFG)
scene.rigid_objects["ball"] = ball

# Cameras
env_cfg = BallTransferEnvCfg()
overhead_cam = Camera(env_cfg.overhead_camera)
scene.sensors["overhead_cam"] = overhead_cam
side_cam = Camera(env_cfg.side_camera)
scene.sensors["side_cam"] = side_cam

# Clone and reset
scene.clone_environments(copy_from_source=False)
sim.reset()
scene.reset()

# Output dir
os.makedirs(args.output_dir, exist_ok=True)

# ── Joint configs ──
init_joints = torch.tensor(
    [[0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0]], device=robot.device
)
# Target: from sweep_workspace.py — sh=-0.8, el=-0.9, wp=-0.1
# EE reaches approximately (-0.022, 0, 0.185) — near our target
target_joints = torch.tensor(
    [[0.0, -0.8, -0.9, -0.1, 0.0, 0.0, 0.0]], device=robot.device
)

# ── Find body indices ──
ee_idx = robot.find_bodies("gripper_base")[0][0]

# ── PD gains (applied as effort targets — PhysX position drives are broken) ──
PD_STIFFNESS = torch.tensor([100, 100, 100, 100, 100, 2000, 2000],
                            device=robot.device, dtype=torch.float32)
PD_DAMPING = torch.tensor([10, 10, 10, 10, 10, 100, 100],
                          device=robot.device, dtype=torch.float32)
MAX_EFFORT = torch.tensor([100, 100, 100, 100, 100, 200, 200],
                          device=robot.device, dtype=torch.float32)

def apply_pd(target_pos):
    """Compute and apply PD torques as effort targets."""
    current_pos = robot.data.joint_pos[0:1]
    current_vel = robot.data.joint_vel[0:1]
    error = target_pos - current_pos
    torque = PD_STIFFNESS * error - PD_DAMPING * current_vel
    torque = torch.clamp(torque, -MAX_EFFORT, MAX_EFFORT)
    robot.set_joint_effort_target(torque)

# ── Set initial pose ──
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))

# Let initial pose settle with effort-based PD
print("Settling initial pose...")
for _ in range(120):
    apply_pd(init_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)

# Snap ball to EE
ee_pos = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
ball_quat = ball.data.root_quat_w
ball_pose = torch.cat([ee_pos, ball_quat], dim=-1)
ball.write_root_pose_to_sim(ball_pose, torch.tensor([0], device=robot.device))
ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=robot.device), torch.tensor([0], device=robot.device))

ee0 = ee_pos[0].cpu().numpy()
print(f"Initial EE: ({ee0[0]:.4f}, {ee0[1]:.4f}, {ee0[2]:.4f})")

def save_frame(frame_idx, output_dir, overhead_cam, side_cam):
    """Save camera frames."""
    if overhead_cam.data.output and "rgb" in overhead_cam.data.output:
        img = overhead_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/overhead_{frame_idx:04d}.png")
    if side_cam.data.output and "rgb" in side_cam.data.output:
        img = side_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/side_{frame_idx:04d}.png")

env_ids = torch.tensor([0], device=robot.device)
SIM_STEPS_PER_FRAME = 4  # 4 steps at 120Hz = 30fps rendering

# Track trajectory for matplotlib fallback
trajectory = {"ee_x": [], "ee_z": [], "ball_x": [], "ball_z": [], "frame": []}

def step_and_render(num_frames, joint_target, frame_offset, label=""):
    """Apply PD effort targets and step sim, saving frames."""
    for f in range(num_frames):
        # Keep ball at EE (kinematic grasp)
        ee_pos_w = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
        ball_pose = torch.cat([ee_pos_w, ball.data.root_quat_w], dim=-1)
        ball.write_root_pose_to_sim(ball_pose, env_ids)
        ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=robot.device), env_ids)

        for _ in range(SIM_STEPS_PER_FRAME):
            apply_pd(joint_target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=1/120)

        frame = frame_offset + f
        save_frame(frame, args.output_dir, overhead_cam, side_cam)

        # Track trajectory
        ep = robot.data.body_pos_w[0, ee_idx, :].cpu().numpy()
        bp = ball.data.root_pos_w[0].cpu().numpy()
        trajectory["ee_x"].append(ep[0])
        trajectory["ee_z"].append(ep[2])
        trajectory["ball_x"].append(bp[0])
        trajectory["ball_z"].append(bp[2])
        trajectory["frame"].append(frame)

        if f % 30 == 0:
            jp = robot.data.joint_pos[0].cpu().numpy()
            print(f"  [{label}] Frame {frame}: EE=({ep[0]:.4f},{ep[1]:.4f},{ep[2]:.4f}) "
                  f"joints={jp.round(3)}")

# ── Phase 1: Hold at init (60 frames = 2s) ──
print("Phase 1: Hold at init...")
step_and_render(60, init_joints, 0, "HOLD")

# ── Phase 2: Move to target via PD (300 frames = 10s) ──
# Set target joints and let PD controller drive the arm there
print("Phase 2: PD-driving to target...")
step_and_render(300, target_joints, 60, "MOVE")

# ── Phase 3: Hold at target (60 frames = 2s) ──
print("Phase 3: Hold at target...")
step_and_render(60, target_joints, 360, "HOLD")

# Final position
ep = robot.data.body_pos_w[0, ee_idx, :].cpu().numpy()
bp = ball.data.root_pos_w[0].cpu().numpy()
print(f"\nFinal EE: ({ep[0]:.4f}, {ep[1]:.4f}, {ep[2]:.4f})")
print(f"Final Ball: ({bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f})")
target_pos = np.array([-0.025, 0.0, 0.185])
dist = np.linalg.norm(bp[:3] - target_pos)
print(f"Ball-to-target distance: {dist:.4f}m")

# ── Save trajectory plot ──
print("\nGenerating trajectory plot...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    frames = trajectory["frame"]
    # Plot 1: X-Z trajectory
    ax1 = axes[0]
    ax1.plot(trajectory["ee_x"], trajectory["ee_z"], 'b-', label='EE path', linewidth=2)
    ax1.plot(trajectory["ball_x"], trajectory["ball_z"], 'r--', label='Ball path', linewidth=1.5)
    ax1.plot(trajectory["ee_x"][0], trajectory["ee_z"][0], 'go', markersize=10, label='Start')
    ax1.plot(trajectory["ee_x"][-1], trajectory["ee_z"][-1], 'rs', markersize=10, label='End')
    ax1.plot(-0.025, 0.185, 'k*', markersize=15, label='Target')
    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Z position (m)')
    ax1.set_title('EE & Ball Trajectory (X-Z plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Position over time
    ax2 = axes[1]
    ax2.plot(frames, trajectory["ee_x"], 'b-', label='EE X')
    ax2.plot(frames, trajectory["ee_z"], 'b--', label='EE Z')
    ax2.plot(frames, trajectory["ball_x"], 'r-', label='Ball X')
    ax2.plot(frames, trajectory["ball_z"], 'r--', label='Ball Z')
    ax2.axhline(y=-0.025, color='gray', linestyle=':', label='Target X')
    ax2.axhline(y=0.185, color='gray', linestyle='-.', label='Target Z')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/trajectory_plot.png", dpi=150)
    print(f"Saved trajectory plot: {args.output_dir}/trajectory_plot.png")
except Exception as e:
    print(f"Could not generate plot: {e}")

# Count frames
overhead_count = len([f for f in os.listdir(args.output_dir) if f.startswith("overhead_") and f.endswith(".png")])
side_count = len([f for f in os.listdir(args.output_dir) if f.startswith("side_") and f.endswith(".png")])
print(f"Saved {overhead_count} overhead frames, {side_count} side frames")

simulation_app.close()
