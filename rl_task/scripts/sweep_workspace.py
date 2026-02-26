"""Sweep joint angles to map workspace and find config that reaches target."""
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import numpy as np
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from alice_ball_transfer.ball_transfer_env_cfg import ALICE_001_CFG

sim_cfg = SimulationCfg(dt=1/120, gravity=(0, 0, -9.81), use_fabric=False)
sim = sim_utils.SimulationContext(sim_cfg)

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)
light = sim_utils.DomeLightCfg(intensity=2000)
light.func("/World/Light", light)

scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

robot = Articulation(ALICE_001_CFG)
scene.articulations["robot"] = robot
scene.clone_environments(copy_from_source=False)

sim.reset()
scene.reset()

# Find EE body
ee_idx = robot.find_bodies("gripper_base")[0][0]
print(f"Joint names: {robot.joint_names}")
print(f"EE body index: {ee_idx}")

# Target position
target = torch.tensor([-0.05, 0.0, 0.20])
source = torch.tensor([-0.091, 0.0, 0.159])
print(f"Target: {target.numpy()}")
print(f"Source: {source.numpy()}")

# Init joint positions
init_joints = torch.tensor([0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0])

# Sweep: shoulder, elbow, wrist_pitch
# Coarse grid first
best_dist = 999.0
best_config = None
results = []

shoulder_range = np.linspace(-1.5, 0.5, 21)
elbow_range = np.linspace(-2.5, -0.5, 21)
wrist_range = np.linspace(-1.5, 0.5, 21)

total = len(shoulder_range) * len(elbow_range) * len(wrist_range)
print(f"Sweeping {total} configurations...")

count = 0
for sh in shoulder_range:
    for el in elbow_range:
        for wp in wrist_range:
            count += 1

            # Set joint positions
            jp = init_joints.clone().unsqueeze(0)
            jp[0, 1] = sh   # shoulder
            jp[0, 2] = el   # elbow
            jp[0, 3] = wp   # wrist_pitch
            jv = torch.zeros_like(jp)

            robot.write_joint_state_to_sim(jp, jv)
            robot.set_joint_position_target(jp)

            # Step to let PD settle
            for _ in range(50):
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt=1/120)

            # Read EE position
            ee_pos = robot.data.body_pos_w[0, ee_idx, :].cpu()
            # Subtract env origin (should be 0 for env 0)
            dist_to_target = torch.norm(ee_pos - target).item()

            if dist_to_target < best_dist:
                best_dist = dist_to_target
                best_config = (sh, el, wp)
                print(f"  [{count}/{total}] NEW BEST: sh={sh:.2f} el={el:.2f} wp={wp:.2f} "
                      f"→ EE=({ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}) "
                      f"dist={dist_to_target:.4f}")

            if count % 500 == 0:
                print(f"  [{count}/{total}] best_dist={best_dist:.4f}")

print(f"\n=== BEST CONFIG ===")
print(f"Shoulder={best_config[0]:.2f}, Elbow={best_config[1]:.2f}, WristPitch={best_config[2]:.2f}")
print(f"Distance to target: {best_dist:.4f}")

# Fine-tune around best
print(f"\n=== FINE-TUNING around best config ===")
sh_fine = np.linspace(best_config[0]-0.2, best_config[0]+0.2, 11)
el_fine = np.linspace(best_config[1]-0.2, best_config[1]+0.2, 11)
wp_fine = np.linspace(best_config[2]-0.2, best_config[2]+0.2, 11)

for sh in sh_fine:
    for el in el_fine:
        for wp in wp_fine:
            jp = init_joints.clone().unsqueeze(0)
            jp[0, 1] = sh
            jp[0, 2] = el
            jp[0, 3] = wp
            jv = torch.zeros_like(jp)

            robot.write_joint_state_to_sim(jp, jv)
            robot.set_joint_position_target(jp)

            for _ in range(50):
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt=1/120)

            ee_pos = robot.data.body_pos_w[0, ee_idx, :].cpu()
            dist_to_target = torch.norm(ee_pos - target).item()

            if dist_to_target < best_dist:
                best_dist = dist_to_target
                best_config = (sh, el, wp)
                print(f"  FINE BEST: sh={sh:.2f} el={el:.2f} wp={wp:.2f} "
                      f"→ EE=({ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}) "
                      f"dist={dist_to_target:.4f}")

print(f"\n=== FINAL BEST CONFIG ===")
print(f"Shoulder={best_config[0]:.3f}, Elbow={best_config[1]:.3f}, WristPitch={best_config[2]:.3f}")
print(f"Distance to target: {best_dist:.4f}")
print(f"Action offsets from init: sh={best_config[0]-(-0.3):.3f}, el={best_config[1]-(-1.8):.3f}, wp={best_config[2]-(-0.5):.3f}")

simulation_app.close()
