"""Minimal diagnostic: does set_joint_position_target actually move the arm?

Tests multiple gain levels and zero-gravity to isolate the issue.
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
import numpy as np
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

init_joints = torch.tensor([[0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0]], device=device)
target_joints = torch.tensor([[0.0, -0.8, -0.9, -0.1, 0.0, 0.0, 0.0]], device=device)

def read_state(label=""):
    jp = robot.data.joint_pos[0].cpu().numpy()
    jv = robot.data.joint_vel[0].cpu().numpy()
    ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    return jp, jv, ee

def print_state(step, jp, jv, ee, label=""):
    print(f"  [{label}] Step {step:3d}: sh={jp[1]:.3f} el={jp[2]:.3f} wp={jp[3]:.3f} "
          f"EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}) "
          f"vel_sh={jv[1]:.2f}")

# ── Read initial PhysX drive properties ──
print("=== PhysX Drive Properties (as configured) ===")
print(f"  max forces:  {robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy().round(2)}")
print(f"  stiffnesses: {robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy().round(2)}")
print(f"  dampings:    {robot.root_physx_view.get_dof_dampings()[0].cpu().numpy().round(2)}")
print(f"  joint names: {robot.joint_names}")

# ── Check link masses ──
print("\n=== Link Masses ===")
for i, name in enumerate(robot.body_names):
    try:
        mass = robot.root_physx_view.get_link_masses()[0, i].item()
        print(f"  {name}: {mass:.4f} kg")
    except:
        pass
total_mass = robot.root_physx_view.get_link_masses()[0].sum().item()
print(f"  TOTAL: {total_mass:.4f} kg")

# ── Test A: Current gains (stiffness=100, maxForce=100) ──
print("\n=== Test A: Current gains (stiffness=100, maxForce=100) ===")
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)
scene.write_data_to_sim()
sim.step()
scene.update(dt=1/120)

for step in range(60):
    robot.set_joint_position_target(init_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 20 == 0:
        jp, jv, ee = read_state()
        print_state(step, jp, jv, ee, "A")

jp_a, _, _ = read_state()
print(f"  Result: shoulder settled at {jp_a[1]:.3f} (target was {init_joints[0,1].item():.1f})")

# ── Test B: Crank up gains (stiffness=5000, maxForce=50000) ──
print("\n=== Test B: HIGH gains (stiffness=5000, maxForce=50000) ===")
new_stiff = torch.tensor([[5000, 5000, 5000, 5000, 5000, 2000, 2000]], dtype=torch.float32, device=device)
new_damp = torch.tensor([[500, 500, 500, 500, 500, 100, 100]], dtype=torch.float32, device=device)
new_force = torch.tensor([[50000, 50000, 50000, 50000, 50000, 200, 200]], dtype=torch.float32, device=device)
robot.root_physx_view.set_dof_stiffnesses(new_stiff, torch.tensor([0], device=device))
robot.root_physx_view.set_dof_dampings(new_damp, torch.tensor([0], device=device))
robot.root_physx_view.set_dof_max_forces(new_force, torch.tensor([0], device=device))

# Verify
print(f"  Verified stiffnesses: {robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy().round(0)}")
print(f"  Verified maxForces:   {robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy().round(0)}")

robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)
scene.write_data_to_sim()
sim.step()
scene.update(dt=1/120)

for step in range(60):
    robot.set_joint_position_target(init_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 20 == 0:
        jp, jv, ee = read_state()
        print_state(step, jp, jv, ee, "B")

jp_b, _, _ = read_state()
print(f"  Result: shoulder settled at {jp_b[1]:.3f} (target was {init_joints[0,1].item():.1f})")

# Now try to move to target
print("  Moving to target...")
for step in range(120):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp, jv, ee = read_state()
        print_state(step, jp, jv, ee, "B-move")

jp_b2, _, ee_b2 = read_state()
print(f"  Result: shoulder={jp_b2[1]:.3f} elbow={jp_b2[2]:.3f} (target: sh={target_joints[0,1].item():.1f} el={target_joints[0,2].item():.1f})")

# ── Test C: Zero gravity ──
print("\n=== Test C: Zero gravity + original gains ===")
# Reset gains to original
orig_stiff = torch.tensor([[100, 100, 100, 100, 100, 2000, 2000]], dtype=torch.float32, device=device)
orig_damp = torch.tensor([[10, 10, 10, 10, 10, 100, 100]], dtype=torch.float32, device=device)
orig_force = torch.tensor([[100, 100, 100, 100, 100, 200, 200]], dtype=torch.float32, device=device)
robot.root_physx_view.set_dof_stiffnesses(orig_stiff, torch.tensor([0], device=device))
robot.root_physx_view.set_dof_dampings(orig_damp, torch.tensor([0], device=device))
robot.root_physx_view.set_dof_max_forces(orig_force, torch.tensor([0], device=device))

# Change gravity to zero
try:
    from omni.isaac.core.utils.prims import get_prim_at_path
    import omni.physx
    from pxr import UsdPhysics, Gf
    stage = sim._backend_utils.get_current_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.SceneAPI):
            scene_api = UsdPhysics.SceneAPI(prim)
            scene_api.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, 0))
            scene_api.GetGravityMagnitudeAttr().Set(0.0)
            print(f"  Set gravity to zero on {prim.GetPath()}")
except Exception as e:
    print(f"  Could not change gravity: {e}")
    print("  Trying alternative method...")
    try:
        sim._physics_sim_view.set_gravity(torch.tensor([0, 0, 0], dtype=torch.float32, device=device))
        print("  Set gravity via physics_sim_view")
    except Exception as e2:
        print(f"  Alternative also failed: {e2}")

robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)
scene.write_data_to_sim()
sim.step()
scene.update(dt=1/120)

for step in range(60):
    robot.set_joint_position_target(init_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 20 == 0:
        jp, jv, ee = read_state()
        print_state(step, jp, jv, ee, "C-hold")

jp_c, _, _ = read_state()
print(f"  Result (hold): shoulder settled at {jp_c[1]:.3f} (target was {init_joints[0,1].item():.1f})")

# Move in zero gravity
print("  Moving to target in zero gravity...")
for step in range(120):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp, jv, ee = read_state()
        print_state(step, jp, jv, ee, "C-move")

jp_c2, _, _ = read_state()
print(f"  Result (move): shoulder={jp_c2[1]:.3f} elbow={jp_c2[2]:.3f}")

# ── Summary ──
print("\n=== Summary ===")
print(f"  Test A (current gains, gravity):     shoulder holds at {jp_a[1]:.3f} (target: -0.3)")
print(f"  Test B (high gains, gravity):         shoulder holds at {jp_b[1]:.3f} (target: -0.3)")
hold_ok = abs(jp_b[1] - (-0.3)) < 0.1
print(f"  Test B hold: {'PASS' if hold_ok else 'FAIL'}")
move_ok = abs(jp_b2[1] - target_joints[0,1].item()) < 0.2
print(f"  Test B move: {'PASS' if move_ok else 'FAIL'}")
print(f"  Test C (original gains, zero-g):      shoulder holds at {jp_c[1]:.3f} (target: -0.3)")
move_zg = abs(jp_c2[1] - target_joints[0,1].item()) < 0.2
print(f"  Test C move: {'PASS' if move_zg else 'FAIL'}")

simulation_app.close()
