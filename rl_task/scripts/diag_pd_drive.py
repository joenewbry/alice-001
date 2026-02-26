"""Minimal diagnostic: does set_joint_position_target actually move the arm?

No cameras, no ball, no complexity. Just robot + PD target + step.
"""
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from alice_ball_transfer.ball_transfer_env_cfg import ALICE_001_CFG

# ── Setup sim ──
sim_cfg = SimulationCfg(dt=1/120, gravity=(0, 0, -9.81), render_interval=1)
sim = sim_utils.SimulationContext(sim_cfg)

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)
light = sim_utils.DomeLightCfg(intensity=3000.0)
light.func("/World/Light", light)

scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

robot = Articulation(ALICE_001_CFG)
scene.articulations["robot"] = robot

scene.clone_environments(copy_from_source=False)
sim.reset()
scene.reset()

ee_idx = robot.find_bodies("gripper_base")[0][0]
device = robot.device

# ── Read PhysX drive properties ──
print("=== PhysX Drive Properties ===")
print(f"  max forces:  {robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy().round(2)}")
print(f"  stiffnesses: {robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy().round(2)}")
print(f"  dampings:    {robot.root_physx_view.get_dof_dampings()[0].cpu().numpy().round(2)}")
print(f"  joint names: {robot.joint_names}")

# ── Check drive types (are they position drives?) ──
try:
    # Try to get drive type info
    import omni.physics
    print(f"  drive types: (check USD)")
except:
    pass

# ── Test 1: Set init position, let settle ──
init_joints = torch.tensor([[0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0]], device=device)
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)

print("\n=== Test 1: Settling from init position ===")
print(f"  Target:  {init_joints[0].cpu().numpy().round(3)}")

for step in range(120):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        print(f"  Step {step:3d}: joints={jp.round(3)} EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f})")

jp_settled = robot.data.joint_pos[0].cpu().numpy()
print(f"  Settled: {jp_settled.round(3)}")

# ── Test 2: Change target, see if arm moves ──
target_joints = torch.tensor([[0.0, -0.8, -0.9, -0.1, 0.0, 0.0, 0.0]], device=device)
print("\n=== Test 2: Change PD target (set_joint_position_target) ===")
print(f"  New target: {target_joints[0].cpu().numpy().round(3)}")

robot.set_joint_position_target(target_joints)
for step in range(300):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 60 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        print(f"  Step {step:3d}: joints={jp.round(3)} EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f})")

jp_after = robot.data.joint_pos[0].cpu().numpy()
print(f"  After PD: {jp_after.round(3)}")
moved = abs(jp_after[1] - jp_settled[1]) > 0.01
print(f"  Shoulder moved? {moved} (delta={jp_after[1] - jp_settled[1]:.4f})")

# ── Test 3: Set target EVERY step (not just once) ──
print("\n=== Test 3: Set target EVERY step ===")
jp_before3 = robot.data.joint_pos[0].cpu().numpy().copy()
for step in range(300):
    robot.set_joint_position_target(target_joints)  # Set every step
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 60 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        print(f"  Step {step:3d}: joints={jp.round(3)} EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f})")

jp_after3 = robot.data.joint_pos[0].cpu().numpy()
print(f"  After: {jp_after3.round(3)}")
moved3 = abs(jp_after3[1] - jp_before3[1]) > 0.01
print(f"  Shoulder moved? {moved3} (delta={jp_after3[1] - jp_before3[1]:.4f})")

# ── Test 4: Direct PhysX API (bypass Isaac Lab) ──
print("\n=== Test 4: Direct PhysX set_dof_position_targets ===")
jp_before4 = robot.data.joint_pos[0].cpu().numpy().copy()
target_direct = target_joints.clone()
for step in range(300):
    robot.root_physx_view.set_dof_position_targets(target_direct, torch.tensor([0], device=device))
    sim.step()
    scene.update(dt=1/120)
    if step % 60 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        print(f"  Step {step:3d}: joints={jp.round(3)} EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f})")

jp_after4 = robot.data.joint_pos[0].cpu().numpy()
print(f"  After: {jp_after4.round(3)}")
moved4 = abs(jp_after4[1] - jp_before4[1]) > 0.01
print(f"  Shoulder moved? {moved4} (delta={jp_after4[1] - jp_before4[1]:.4f})")

# ── Test 5: Kinematic teleport (write_joint_state_to_sim) ──
print("\n=== Test 5: Kinematic teleport (write_joint_state_to_sim) ===")
jp_before5 = robot.data.joint_pos[0].cpu().numpy().copy()
robot.write_joint_state_to_sim(target_joints, torch.zeros_like(target_joints))
for step in range(60):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 20 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        print(f"  Step {step:3d}: joints={jp.round(3)} EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f})")

jp_after5 = robot.data.joint_pos[0].cpu().numpy()
print(f"  After: {jp_after5.round(3)}")
moved5 = abs(jp_after5[1] - jp_before5[1]) > 0.01
print(f"  Shoulder moved? {moved5} (delta={jp_after5[1] - jp_before5[1]:.4f})")

# ── Test 6: Check actual joint forces being applied ──
print("\n=== Test 6: Read applied forces ===")
try:
    forces = robot.root_physx_view.get_dof_projected_joint_forces()
    print(f"  Applied forces: {forces[0].cpu().numpy().round(3)}")
except Exception as e:
    print(f"  Could not read forces: {e}")

try:
    torques = robot.data.applied_torque
    print(f"  Applied torques: {torques[0].cpu().numpy().round(3)}")
except Exception as e:
    print(f"  Could not read torques: {e}")

print("\n=== Summary ===")
print(f"  Test 2 (PD target once):       {'PASS' if abs(jp_after[1] - jp_settled[1]) > 0.01 else 'FAIL'}")
print(f"  Test 3 (PD target every step):  {'PASS' if moved3 else 'FAIL'}")
print(f"  Test 4 (Direct PhysX):          {'PASS' if moved4 else 'FAIL'}")
print(f"  Test 5 (Kinematic teleport):    {'PASS' if moved5 else 'FAIL'}")

simulation_app.close()
