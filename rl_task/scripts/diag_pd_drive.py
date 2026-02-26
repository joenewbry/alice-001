"""Minimal diagnostic: test ALL drive modes to find what actually moves the arm.

Tests position targets, velocity targets, effort targets, and direct PhysX forces.
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
sim_cfg = SimulationCfg(dt=1/120, gravity=(0.0, 0.0, 0.0), render_interval=1)
sim = sim_utils.SimulationContext(sim_cfg)

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)

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

print("=== PhysX Drive Properties ===")
print(f"  stiffnesses: {robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy().round(2)}")
print(f"  dampings:    {robot.root_physx_view.get_dof_dampings()[0].cpu().numpy().round(2)}")
print(f"  max forces:  {robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy().round(2)}")

def reset_to_init():
    robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
    robot.set_joint_position_target(init_joints)
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=1/120)

def print_joint(step, label):
    jp = robot.data.joint_pos[0].cpu().numpy()
    jv = robot.data.joint_vel[0].cpu().numpy()
    ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    print(f"  [{label}] Step {step:3d}: sh={jp[1]:.3f} el={jp[2]:.3f} wp={jp[3]:.3f} "
          f"vel_sh={jv[1]:.3f} EE=({ee[0]:.4f},{ee[2]:.4f})")

# ── Test 1: Position targets (Isaac Lab API) ──
print("\n=== Test 1: set_joint_position_target (Isaac Lab) ===")
reset_to_init()
for step in range(120):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        print_joint(step, "pos_target")
jp = robot.data.joint_pos[0].cpu().numpy()
t1_err = abs(jp[1] - (-0.8))
print(f"  Result: sh_err={t1_err:.3f} {'PASS' if t1_err < 0.2 else 'FAIL'}")

# ── Test 2: Velocity targets ──
print("\n=== Test 2: set_joint_velocity_target ===")
reset_to_init()
vel_target = torch.zeros_like(init_joints)
vel_target[0, 1] = -1.0  # 1 rad/s toward -0.8
vel_target[0, 2] = 1.0   # 1 rad/s toward -0.9
for step in range(120):
    robot.set_joint_velocity_target(vel_target)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        print_joint(step, "vel_target")
jp = robot.data.joint_pos[0].cpu().numpy()
t2_moved = abs(jp[1] - (-0.3)) > 0.05
print(f"  Result: sh moved {abs(jp[1]-(-0.3)):.3f} rad {'PASS' if t2_moved else 'FAIL'}")

# ── Test 3: Effort targets (direct torque) ──
print("\n=== Test 3: set_joint_effort_target (direct torque) ===")
reset_to_init()
effort = torch.zeros_like(init_joints)
effort[0, 1] = -10.0  # 10 N·m toward -0.8
effort[0, 2] = 10.0   # 10 N·m toward -0.9
for step in range(120):
    robot.set_joint_effort_target(effort)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        print_joint(step, "effort")
jp = robot.data.joint_pos[0].cpu().numpy()
t3_moved = abs(jp[1] - (-0.3)) > 0.05
print(f"  Result: sh moved {abs(jp[1]-(-0.3)):.3f} rad {'PASS' if t3_moved else 'FAIL'}")

# ── Test 4: Direct PhysX DOF targets ──
print("\n=== Test 4: root_physx_view.set_dof_position_targets (direct PhysX) ===")
reset_to_init()
idx = torch.tensor([0], device=device)
for step in range(120):
    robot.root_physx_view.set_dof_position_targets(target_joints, idx)
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        print_joint(step, "phx_pos")
jp = robot.data.joint_pos[0].cpu().numpy()
t4_err = abs(jp[1] - (-0.8))
print(f"  Result: sh_err={t4_err:.3f} {'PASS' if t4_err < 0.2 else 'FAIL'}")

# ── Test 5: Direct PhysX actuation forces ──
print("\n=== Test 5: root_physx_view.set_dof_actuation_forces (direct PhysX force) ===")
reset_to_init()
forces = torch.zeros_like(init_joints)
forces[0, 1] = -10.0  # 10 N·m
forces[0, 2] = 10.0
for step in range(120):
    robot.root_physx_view.set_dof_actuation_forces(forces, idx)
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        print_joint(step, "phx_force")
jp = robot.data.joint_pos[0].cpu().numpy()
t5_moved = abs(jp[1] - (-0.3)) > 0.05
print(f"  Result: sh moved {abs(jp[1]-(-0.3)):.3f} rad {'PASS' if t5_moved else 'FAIL'}")

# ── Test 6: Kinematic teleport + PD (position + velocity write) ──
print("\n=== Test 6: write_joint_state_to_sim interpolation ===")
reset_to_init()
n_interp = 60
for step in range(n_interp):
    t = (step + 1) / n_interp
    interp = init_joints * (1 - t) + target_joints * t
    robot.write_joint_state_to_sim(interp, torch.zeros_like(interp))
    robot.set_joint_position_target(interp)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 15 == 0:
        print_joint(step, "interp")
jp = robot.data.joint_pos[0].cpu().numpy()
t6_err = abs(jp[1] - (-0.8))
print(f"  Result: sh_err={t6_err:.3f} {'PASS' if t6_err < 0.2 else 'FAIL'}")

# ── Summary ──
print("\n=== SUMMARY ===")
print(f"  Test 1 (position target):     {'PASS' if t1_err < 0.2 else 'FAIL'}")
print(f"  Test 2 (velocity target):     {'PASS' if t2_moved else 'FAIL'}")
print(f"  Test 3 (effort target):       {'PASS' if t3_moved else 'FAIL'}")
print(f"  Test 4 (PhysX pos target):    {'PASS' if t4_err < 0.2 else 'FAIL'}")
print(f"  Test 5 (PhysX force):         {'PASS' if t5_moved else 'FAIL'}")
print(f"  Test 6 (kinematic interp):    {'PASS' if t6_err < 0.2 else 'FAIL'}")

simulation_app.close()
