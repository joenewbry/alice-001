"""Minimal diagnostic: does set_joint_position_target actually move the arm?

Tests multiple configurations to find working PD gains.
"""
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--stiffness", type=float, default=100.0)
parser.add_argument("--damping", type=float, default=10.0)
parser.add_argument("--max_force", type=float, default=100.0)
parser.add_argument("--no_gravity", action="store_true")
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import copy
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from alice_ball_transfer.ball_transfer_env_cfg import ALICE_001_CFG

# ── Override actuator config BEFORE creating the robot ──
# This is how Isaac Lab applies gains — through the config, not at runtime
arm_cfg = ALICE_001_CFG.actuators["arm"]
arm_cfg.stiffness = args.stiffness
arm_cfg.damping = args.damping
arm_cfg.effort_limit_sim = args.max_force
print(f"\n=== Config: stiffness={args.stiffness}, damping={args.damping}, maxForce={args.max_force}, gravity={'OFF' if args.no_gravity else 'ON'} ===\n")

# ── Setup sim ──
gravity = (0.0, 0.0, 0.0) if args.no_gravity else (0.0, 0.0, -9.81)
sim_cfg = SimulationCfg(dt=1/120, gravity=gravity, render_interval=1)
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

# ── Verify PhysX drive properties ──
print("=== PhysX Drive Properties (after creation) ===")
print(f"  max forces:  {robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy().round(2)}")
print(f"  stiffnesses: {robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy().round(2)}")
print(f"  dampings:    {robot.root_physx_view.get_dof_dampings()[0].cpu().numpy().round(2)}")

# ── Link masses ──
inv_masses = robot.root_physx_view.get_inv_masses()[0]
for i, name in enumerate(robot.body_names):
    inv_m = inv_masses[i].item()
    mass = 1.0 / inv_m if inv_m > 0 else float('inf')
    print(f"  {name}: {mass:.4f} kg")

init_joints = torch.tensor([[0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0]], device=device)
target_joints = torch.tensor([[0.0, -0.8, -0.9, -0.1, 0.0, 0.0, 0.0]], device=device)

# ── Test 1: Hold at init ──
print("\n=== Test 1: Hold at init position ===")
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)

for step in range(120):
    robot.set_joint_position_target(init_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        print(f"  Step {step:3d}: sh={jp[1]:.3f} el={jp[2]:.3f} wp={jp[3]:.3f} "
              f"EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}) vel_sh={jv[1]:.2f}")

jp = robot.data.joint_pos[0].cpu().numpy()
hold_error = abs(jp[1] - init_joints[0, 1].item())
print(f"  Shoulder error: {hold_error:.3f} rad ({'PASS' if hold_error < 0.1 else 'FAIL'})")

# ── Test 2: Move to target ──
print("\n=== Test 2: Move to target ===")

# Debug: read PhysX targets BEFORE and AFTER setting them
print("  Before set_joint_position_target:")
try:
    pre_targets = robot.root_physx_view.get_dof_position_targets()[0].cpu().numpy()
    print(f"    PhysX drive targets: {pre_targets.round(3)}")
except Exception as e:
    print(f"    Could not read: {e}")

robot.set_joint_position_target(target_joints)
print(f"  Called set_joint_position_target({target_joints[0].cpu().numpy().round(3)})")
print(f"  _has_joint_pos_target: {robot._has_joint_pos_target if hasattr(robot, '_has_joint_pos_target') else 'N/A'}")
print(f"  data.joint_pos_target: {robot.data.joint_pos_target[0].cpu().numpy().round(3)}")

scene.write_data_to_sim()
print("  After write_data_to_sim:")
try:
    post_targets = robot.root_physx_view.get_dof_position_targets()[0].cpu().numpy()
    print(f"    PhysX drive targets: {post_targets.round(3)}")
except Exception as e:
    print(f"    Could not read: {e}")

for step in range(300):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 60 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        # Also read back PhysX target
        phx_tgt = robot.root_physx_view.get_dof_position_targets()[0].cpu().numpy()
        print(f"  Step {step:3d}: sh={jp[1]:.3f} el={jp[2]:.3f} wp={jp[3]:.3f} "
              f"EE=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}) vel_sh={jv[1]:.2f} "
              f"phx_tgt_sh={phx_tgt[1]:.3f}")

jp = robot.data.joint_pos[0].cpu().numpy()
ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
sh_err = abs(jp[1] - target_joints[0, 1].item())
el_err = abs(jp[2] - target_joints[0, 2].item())
print(f"\n=== Results ===")
print(f"  Shoulder: actual={jp[1]:.3f} target={target_joints[0,1].item():.1f} error={sh_err:.3f} {'PASS' if sh_err < 0.2 else 'FAIL'}")
print(f"  Elbow:    actual={jp[2]:.3f} target={target_joints[0,2].item():.1f} error={el_err:.3f} {'PASS' if el_err < 0.2 else 'FAIL'}")
print(f"  EE pos:   ({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f})")
print(f"  All joints: {jp.round(3)}")

simulation_app.close()
