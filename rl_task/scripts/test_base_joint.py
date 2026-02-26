"""Quick diagnostic: can the base joint actually move?"""
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from alice_ball_transfer.ball_transfer_env_cfg import ALICE_001_CFG

# Minimal sim
sim_cfg = SimulationCfg(dt=1/120, gravity=(0, 0, -9.81), use_fabric=False)
sim = sim_utils.SimulationContext(sim_cfg)

# Ground
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)

# Light
light = sim_utils.DomeLightCfg(intensity=2000)
light.func("/World/Light", light)

# Scene with 1 env
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

# Robot
robot = Articulation(ALICE_001_CFG)
scene.articulations["robot"] = robot
scene.clone_environments(copy_from_source=False)

# Reset sim
sim.reset()
scene.reset()

print(f"Joint names: {robot.joint_names}")
print(f"Num joints: {robot.num_joints}")
print(f"Joint limits (soft): lower={robot.data.soft_joint_pos_limits[0,:,0].cpu().numpy().round(3)}")
print(f"Joint limits (soft): upper={robot.data.soft_joint_pos_limits[0,:,1].cpu().numpy().round(3)}")

# Initial position
jp = robot.data.joint_pos[0].cpu().numpy()
print(f"\n=== INITIAL joint positions: {jp.round(4)}")

# Test 1: Set base_joint target to 1.0 rad
print("\n=== TEST: Setting base_joint target to 1.0 rad ===")
targets = robot.data.joint_pos.clone()
targets[0, 0] = 1.0  # base_joint = 1.0 rad
robot.set_joint_position_target(targets)

# Step 200 times
for i in range(200):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if i % 50 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        print(f"  Step {i:3d}: joints={jp.round(4)}, vel={jv.round(4)}")

jp = robot.data.joint_pos[0].cpu().numpy()
print(f"\n=== AFTER 200 steps: joints={jp.round(4)}")
print(f"=== base_joint moved from 0.0 to {jp[0]:.4f} (target was 1.0)")

# Test 2: Set ALL joints to target [0.5, 0.5, -1.0, 0.5, 0.5, 0, 0]
print("\n=== TEST: Setting all arm joints to non-zero targets ===")
targets2 = torch.tensor([[0.5, 0.5, -1.0, 0.5, 0.5, 0.0, 0.0]], device=robot.device)
robot.set_joint_position_target(targets2)

for i in range(200):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if i % 50 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        print(f"  Step {i:3d}: joints={jp.round(4)}")

jp = robot.data.joint_pos[0].cpu().numpy()
print(f"\n=== FINAL: joints={jp.round(4)}")
print(f"=== Targets were: [0.5, 0.5, -1.0, 0.5, 0.5, 0, 0]")

simulation_app.close()
