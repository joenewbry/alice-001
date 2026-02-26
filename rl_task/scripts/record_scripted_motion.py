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

# ── Set initial pose ──
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)

# Let initial pose settle
print("Settling initial pose...")
for _ in range(120):
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

# ── Phase 1: Hold at init (30 frames) ──
print("Phase 1: Hold at init...")
for frame in range(30):
    robot.set_joint_position_target(init_joints)
    # Keep ball at EE
    ee_pos_w = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
    ball_pose = torch.cat([ee_pos_w, ball.data.root_quat_w], dim=-1)
    ball.write_root_pose_to_sim(ball_pose, torch.tensor([0], device=robot.device))
    ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=robot.device), torch.tensor([0], device=robot.device))

    for _ in range(2):  # 2 sim steps per frame at 120Hz → 60fps
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=1/120)

    # Save frames
    if overhead_cam.data.output and "rgb" in overhead_cam.data.output:
        img = overhead_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args.output_dir}/overhead_{frame:04d}.png")
    if side_cam.data.output and "rgb" in side_cam.data.output:
        img = side_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args.output_dir}/side_{frame:04d}.png")

# ── Phase 2: Interpolate init → target (180 frames) ──
print("Phase 2: Interpolating to target...")
for frame_in_phase in range(180):
    t = min(1.0, frame_in_phase / 120.0)  # Linear interp over 2 seconds
    current_joints = init_joints + t * (target_joints - init_joints)
    robot.set_joint_position_target(current_joints)

    # Keep ball at EE
    ee_pos_w = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
    ball_pose = torch.cat([ee_pos_w, ball.data.root_quat_w], dim=-1)
    ball.write_root_pose_to_sim(ball_pose, torch.tensor([0], device=robot.device))
    ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=robot.device), torch.tensor([0], device=robot.device))

    for _ in range(2):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=1/120)

    frame = 30 + frame_in_phase
    if overhead_cam.data.output and "rgb" in overhead_cam.data.output:
        img = overhead_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args.output_dir}/overhead_{frame:04d}.png")
    if side_cam.data.output and "rgb" in side_cam.data.output:
        img = side_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args.output_dir}/side_{frame:04d}.png")

    if frame_in_phase % 30 == 0:
        ep = robot.data.body_pos_w[0, ee_idx, :].cpu().numpy()
        bp = ball.data.root_pos_w[0].cpu().numpy()
        print(f"  Frame {frame}: t={t:.2f} EE=({ep[0]:.4f},{ep[1]:.4f},{ep[2]:.4f}) "
              f"Ball=({bp[0]:.4f},{bp[1]:.4f},{bp[2]:.4f})")

# ── Phase 3: Hold at target (90 frames) ──
print("Phase 3: Hold at target...")
for frame_in_phase in range(90):
    robot.set_joint_position_target(target_joints)
    ee_pos_w = robot.data.body_pos_w[0, ee_idx, :].unsqueeze(0)
    ball_pose = torch.cat([ee_pos_w, ball.data.root_quat_w], dim=-1)
    ball.write_root_pose_to_sim(ball_pose, torch.tensor([0], device=robot.device))
    ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=robot.device), torch.tensor([0], device=robot.device))

    for _ in range(2):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=1/120)

    frame = 210 + frame_in_phase
    if overhead_cam.data.output and "rgb" in overhead_cam.data.output:
        img = overhead_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args.output_dir}/overhead_{frame:04d}.png")
    if side_cam.data.output and "rgb" in side_cam.data.output:
        img = side_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args.output_dir}/side_{frame:04d}.png")

# Final position
ep = robot.data.body_pos_w[0, ee_idx, :].cpu().numpy()
bp = ball.data.root_pos_w[0].cpu().numpy()
print(f"\nFinal EE: ({ep[0]:.4f}, {ep[1]:.4f}, {ep[2]:.4f})")
print(f"Final Ball: ({bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f})")
target_pos = np.array([-0.025, 0.0, 0.185])
dist = np.linalg.norm(bp[:3] - target_pos)
print(f"Ball-to-target distance: {dist:.4f}m")

# ── Assemble videos with ffmpeg ──
print("\nAssembling videos...")
os.system(f"ffmpeg -y -framerate 30 -i {args.output_dir}/overhead_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p {args.output_dir}/overhead_motion.mp4 2>/dev/null")
os.system(f"ffmpeg -y -framerate 30 -i {args.output_dir}/side_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p {args.output_dir}/side_motion.mp4 2>/dev/null")

# Count frames
overhead_count = len([f for f in os.listdir(args.output_dir) if f.startswith("overhead_") and f.endswith(".png")])
side_count = len([f for f in os.listdir(args.output_dir) if f.startswith("side_") and f.endswith(".png")])
print(f"Saved {overhead_count} overhead frames, {side_count} side frames")
print(f"Videos: {args.output_dir}/overhead_motion.mp4, {args.output_dir}/side_motion.mp4")

simulation_app.close()
