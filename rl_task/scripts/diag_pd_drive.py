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
# CRITICAL: test with use_fabric=True — use_fabric=False may break PhysX drives
sim_cfg = SimulationCfg(dt=1/120, gravity=gravity, render_interval=1, use_fabric=True)
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

# ── Test 2: Move to target (check sleep state) ──
print("\n=== Test 2: Move to target ===")

# Check if body is sleeping
try:
    sleep_states = robot.root_physx_view.get_sleep_states()
    print(f"  Sleep states before: {sleep_states}")
except Exception as e:
    print(f"  Could not read sleep state: {e}")

# Check PhysX targets
pre_targets = robot.root_physx_view.get_dof_position_targets()[0].cpu().numpy()
print(f"  PhysX targets before: {pre_targets.round(3)}")

# Set new target
robot.set_joint_position_target(target_joints)
scene.write_data_to_sim()
post_targets = robot.root_physx_view.get_dof_position_targets()[0].cpu().numpy()
print(f"  PhysX targets after:  {post_targets.round(3)}")
print(f"  data.joint_pos_target: {robot.data.joint_pos_target[0].cpu().numpy().round(3)}")

for step in range(120):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        phx_tgt = robot.root_physx_view.get_dof_position_targets()[0].cpu().numpy()
        try:
            sleep_st = robot.root_physx_view.get_sleep_states()
            sl = sleep_st.cpu().numpy() if hasattr(sleep_st, 'cpu') else sleep_st
        except:
            sl = "?"
        print(f"  Step {step:3d}: sh={jp[1]:.3f} vel_sh={jv[1]:.2f} phx_tgt={phx_tgt[1]:.3f} sleep={sl}")

jp_a = robot.data.joint_pos[0].cpu().numpy()
print(f"  Result A: shoulder error={abs(jp_a[1] - target_joints[0,1].item()):.3f} {'PASS' if abs(jp_a[1] - target_joints[0,1].item()) < 0.2 else 'FAIL'}")

# ── Test 3: Wake up body then try to move ──
print("\n=== Test 3: Wake body with velocity impulse, then move ===")
# Write a tiny velocity to wake PhysX
current_pos = robot.data.joint_pos.clone()
wake_vel = torch.zeros_like(current_pos)
wake_vel[0, 1] = 0.01  # tiny shoulder velocity
robot.write_joint_state_to_sim(current_pos, wake_vel)
robot.set_joint_position_target(target_joints)

for step in range(120):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        try:
            sleep_st = robot.root_physx_view.get_sleep_states()
            sl = sleep_st.cpu().numpy() if hasattr(sleep_st, 'cpu') else sleep_st
        except:
            sl = "?"
        print(f"  Step {step:3d}: sh={jp[1]:.3f} vel_sh={jv[1]:.2f} sleep={sl}")

jp_b = robot.data.joint_pos[0].cpu().numpy()
print(f"  Result B: shoulder error={abs(jp_b[1] - target_joints[0,1].item()):.3f} {'PASS' if abs(jp_b[1] - target_joints[0,1].item()) < 0.2 else 'FAIL'}")

# ── Test 4: Disable sleep threshold via PhysX view ──
print("\n=== Test 4: Reset state + disable sleep + move ===")
robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
robot.set_joint_position_target(init_joints)
# Try to disable sleeping
try:
    robot.root_physx_view.set_sleep_thresholds(torch.tensor([[0.0]], device=device))
    print("  Set sleep threshold to 0")
except Exception as e:
    print(f"  Could not set sleep threshold: {e}")
    try:
        # Alternative: wake the body each step
        robot.root_physx_view.wake_up()
        print("  Called wake_up()")
    except Exception as e2:
        print(f"  Could not wake: {e2}")

# Settle with disabled sleep
for step in range(60):
    robot.set_joint_position_target(init_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)

jp = robot.data.joint_pos[0].cpu().numpy()
print(f"  After hold: sh={jp[1]:.3f}")

# Now move
for step in range(120):
    robot.set_joint_position_target(target_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=1/120)
    if step % 30 == 0:
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        try:
            sleep_st = robot.root_physx_view.get_sleep_states()
            sl = sleep_st.cpu().numpy() if hasattr(sleep_st, 'cpu') else sleep_st
        except:
            sl = "?"
        print(f"  Step {step:3d}: sh={jp[1]:.3f} vel_sh={jv[1]:.2f} sleep={sl}")

jp_c = robot.data.joint_pos[0].cpu().numpy()
print(f"  Result C: shoulder error={abs(jp_c[1] - target_joints[0,1].item()):.3f} {'PASS' if abs(jp_c[1] - target_joints[0,1].item()) < 0.2 else 'FAIL'}")

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
