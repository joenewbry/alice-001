"""Verify arm motion with kinematic interpolation. No cameras — just trajectory plot."""
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--output_dir", type=str, default="/home/joe/motion_proof")
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import os
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from alice_ball_transfer.ball_transfer_env_cfg import (
    ALICE_001_CFG, BALL_CFG, PEDESTAL_CFG, TABLE_CFG,
)

# ── Sim setup ──
sim_cfg = SimulationCfg(dt=1/120, gravity=(0.0, 0.0, -9.81), render_interval=1)
sim = sim_utils.SimulationContext(sim_cfg)

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)
light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
light.func("/World/Light", light)

scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

robot = Articulation(ALICE_001_CFG)
scene.articulations["robot"] = robot
table = RigidObject(TABLE_CFG)
scene.rigid_objects["table"] = table
pedestal = RigidObject(PEDESTAL_CFG)
scene.rigid_objects["pedestal"] = pedestal
ball = RigidObject(BALL_CFG)
scene.rigid_objects["ball"] = ball

scene.clone_environments(copy_from_source=False)
sim.reset()
scene.reset()

os.makedirs(args.output_dir, exist_ok=True)

ee_idx = robot.find_bodies("gripper_base")[0][0]
device = robot.device
env_ids = torch.tensor([0], device=device)

# Joint configs
init_joints = torch.tensor([[0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0]], device=device)
target_joints = torch.tensor([[0.0, -0.8, -0.9, -0.1, 0.0, 0.0, 0.0]], device=device)

# ── Initialize ──
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
for _ in range(10):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)

jp = robot.data.joint_pos[0].cpu().numpy()
ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
print(f"Init: joints={jp.round(3)} EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f})")

# Snap ball to EE
ee_pos = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
ball_pose = torch.cat([ee_pos, ball.data.root_quat_w], dim=-1)
ball.write_root_pose_to_sim(ball_pose, env_ids)
ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids)

# ── Trajectory tracking ──
traj = {"frame": [], "ee_x": [], "ee_z": [], "ball_x": [], "ball_z": [],
        "sh": [], "el": [], "wp": [], "target_sh": [], "target_el": [], "target_wp": []}

ALPHA = 0.5  # Smoothing factor — high for tight gravity-compensation tracking

def step_sim(num_steps, target, label=""):
    """Step sim with kinematic interpolation and track trajectory."""
    for s in range(num_steps):
        # Keep ball at EE
        ee_w = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
        bp = torch.cat([ee_w, ball.data.root_quat_w], dim=-1)
        ball.write_root_pose_to_sim(bp, env_ids)
        ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids)

        # Kinematic interpolation toward target
        current = robot.data.joint_pos[0:1].clone()
        interp = current + ALPHA * (target - current)
        robot.write_joint_state_to_sim(interp, torch.zeros_like(interp))

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=1/120)

        # Record every 4th step (30fps equivalent)
        if s % 4 == 0:
            jp = robot.data.joint_pos[0].cpu().numpy()
            ep = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
            bp_np = ball.data.root_pos_w[0].cpu().numpy()
            traj["frame"].append(len(traj["frame"]))
            traj["ee_x"].append(ep[0]); traj["ee_z"].append(ep[2])
            traj["ball_x"].append(bp_np[0]); traj["ball_z"].append(bp_np[2])
            traj["sh"].append(jp[1]); traj["el"].append(jp[2]); traj["wp"].append(jp[3])
            traj["target_sh"].append(target[0, 1].item())
            traj["target_el"].append(target[0, 2].item())
            traj["target_wp"].append(target[0, 3].item())

        if s % 120 == 0:
            jp = robot.data.joint_pos[0].cpu().numpy()
            ep = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
            print(f"  [{label}] step {s}: sh={jp[1]:.3f} el={jp[2]:.3f} wp={jp[3]:.3f} "
                  f"EE=({ep[0]:.4f},{ep[2]:.4f})")

# ── Phase 1: Hold at init (120 steps = 1s) ──
print("Phase 1: Hold at init...")
step_sim(120, init_joints, "HOLD")

# ── Phase 2: Move to target (480 steps = 4s) ──
print("Phase 2: Kinematic interpolation to target...")
step_sim(480, target_joints, "MOVE")

# ── Phase 3: Hold at target (120 steps = 1s) ──
print("Phase 3: Hold at target...")
step_sim(120, target_joints, "HOLD")

# ── Final state ──
ep = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
bp = ball.data.root_pos_w[0].cpu().numpy()
target_pos = np.array([-0.154, 0.0, 0.243])
dist = np.linalg.norm(bp[:3] - target_pos)
print(f"\nFinal EE: ({ep[0]:.4f}, {ep[1]:.4f}, {ep[2]:.4f})")
print(f"Final Ball: ({bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f})")
print(f"Ball-to-target distance: {dist:.4f}m")

jp = robot.data.joint_pos[0].cpu().numpy()
target_jp = target_joints[0].cpu().numpy()
print(f"Joint errors: sh={abs(jp[1]-target_jp[1]):.4f} el={abs(jp[2]-target_jp[2]):.4f} wp={abs(jp[3]-target_jp[3]):.4f}")

if dist < 0.02:
    print("SUCCESS: Ball reached target!")
elif dist < 0.05:
    print("CLOSE: Ball near target.")
else:
    print("FAIL: Ball did not reach target.")

# ── Generate trajectory plot ──
print("\nGenerating trajectory plot...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
frames = traj["frame"]

ax1 = axes[0, 0]
ax1.plot(traj["ee_x"], traj["ee_z"], 'b-', label='EE path', linewidth=2)
ax1.plot(traj["ball_x"], traj["ball_z"], 'r--', label='Ball path', linewidth=1.5)
ax1.plot(traj["ee_x"][0], traj["ee_z"][0], 'go', markersize=10, label='Start')
ax1.plot(traj["ee_x"][-1], traj["ee_z"][-1], 'rs', markersize=10, label='End')
ax1.plot(-0.154, 0.243, 'k*', markersize=15, label='Target')
ax1.set_xlabel('X position (m)')
ax1.set_ylabel('Z position (m)')
ax1.set_title('EE & Ball Trajectory (X-Z plane)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

ax2 = axes[0, 1]
ax2.plot(frames, traj["ee_x"], 'b-', label='EE X')
ax2.plot(frames, traj["ee_z"], 'b--', label='EE Z')
ax2.axhline(y=-0.154, color='gray', linestyle=':', label='Target X')
ax2.axhline(y=0.243, color='gray', linestyle='-.', label='Target Z')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Position (m)')
ax2.set_title('EE Position vs Time')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(frames, traj["sh"], 'b-', label='Shoulder', linewidth=2)
ax3.plot(frames, traj["el"], 'r-', label='Elbow', linewidth=2)
ax3.plot(frames, traj["wp"], 'g-', label='Wrist Pitch', linewidth=2)
ax3.plot(frames, traj["target_sh"], 'b:', label='Target Sh', alpha=0.5)
ax3.plot(frames, traj["target_el"], 'r:', label='Target El', alpha=0.5)
ax3.plot(frames, traj["target_wp"], 'g:', label='Target WP', alpha=0.5)
ax3.set_xlabel('Frame')
ax3.set_ylabel('Angle (rad)')
ax3.set_title('Joint Angles vs Time (actual + target)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
sh_err = [a - t for a, t in zip(traj["sh"], traj["target_sh"])]
el_err = [a - t for a, t in zip(traj["el"], traj["target_el"])]
wp_err = [a - t for a, t in zip(traj["wp"], traj["target_wp"])]
ax4.plot(frames, sh_err, 'b-', label='Shoulder err')
ax4.plot(frames, el_err, 'r-', label='Elbow err')
ax4.plot(frames, wp_err, 'g-', label='Wrist err')
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax4.set_xlabel('Frame')
ax4.set_ylabel('Error (rad)')
ax4.set_title('Tracking Error')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Alice-001: Kinematic Interpolation Motion Proof', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{args.output_dir}/trajectory_plot.png", dpi=150)
print(f"Saved: {args.output_dir}/trajectory_plot.png")

simulation_app.close()
