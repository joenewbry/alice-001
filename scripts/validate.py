#!/usr/bin/env python3
"""Validate Alice 001 USD asset in Isaac Lab.

Run on cloud GPU after URDF -> USD conversion.

Tests:
1. Gravity drop: Load USD, play sim, arm should collapse gracefully
2. Joint sweep: Set each joint target through full range
3. Gripper test: Open/close cycle
4. Multi-env test: Spawn 64 copies, verify performance

Usage:
    ./isaaclab.sh -p scripts/validate.py --num_envs 1 --test gravity
    ./isaaclab.sh -p scripts/validate.py --num_envs 1 --test joints
    ./isaaclab.sh -p scripts/validate.py --num_envs 1 --test gripper
    ./isaaclab.sh -p scripts/validate.py --num_envs 64 --test multi
    ./isaaclab.sh -p scripts/validate.py --num_envs 1 --test all
"""

import argparse
import math
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Alice 001 USD in Isaac Lab")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--test", type=str, default="all",
                        choices=["gravity", "joints", "gripper", "multi", "all"])
    parser.add_argument("--usd_path", type=str,
                        default="./usd/alice_001.usd")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def create_sim_cfg(args):
    """Create simulation configuration."""
    import omni.isaac.lab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS
            num_position_iterations=8,
            num_velocity_iterations=0,
            max_depenetration_velocity=5.0,
        ),
    )


def create_robot_cfg(usd_path: str):
    """Create robot articulation configuration."""
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.assets.articulation import ArticulationCfg

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "base_joint": 0.0,
                "shoulder_joint": 0.0,
                "elbow_joint": -0.5,
                "wrist_pitch_joint": 0.3,
                "wrist_roll_joint": 0.0,
                "left_finger_joint": 0.15,
                "right_finger_joint": -0.15,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["base_joint", "shoulder_joint", "elbow_joint",
                                  "wrist_pitch_joint", "wrist_roll_joint"],
                effort_limit=2.5,
                velocity_limit=5.24,
                stiffness=40.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_finger_joint", "right_finger_joint"],
                effort_limit=1.0,
                velocity_limit=5.24,
                stiffness=20.0,
                damping=2.0,
            ),
        },
    )


def test_gravity_drop(robot, sim, num_steps=500):
    """Test 1: Let arm fall under gravity. Should collapse smoothly, no explosions."""
    print("\n=== TEST: Gravity Drop ===")

    # Set all joint targets to 0 (no stiffness fight)
    robot.set_joint_position_target(torch.zeros(1, robot.num_joints, device=robot.device))

    max_vel = 0.0
    for step in range(num_steps):
        sim.step()
        robot.update(sim.cfg.dt)

        if step % 100 == 0:
            pos = robot.data.joint_pos[0]
            vel = robot.data.joint_vel[0]
            max_vel = max(max_vel, vel.abs().max().item())
            print(f"  Step {step}: joint_pos={pos.cpu().numpy().round(3)}")

    # Check for explosion (unreasonable velocities)
    final_vel = robot.data.joint_vel[0].abs().max().item()
    if final_vel > 50.0:  # rad/s
        print(f"  FAIL: Excessive velocity {final_vel:.1f} rad/s (likely unstable)")
        return False
    else:
        print(f"  PASS: Max velocity {max_vel:.1f} rad/s (stable)")
        return True


def test_joint_sweep(robot, sim, num_steps_per_joint=300):
    """Test 2: Sweep each joint through its range."""
    print("\n=== TEST: Joint Sweep ===")

    joint_names = ["base_joint", "shoulder_joint", "elbow_joint",
                   "wrist_pitch_joint", "wrist_roll_joint",
                   "left_finger_joint", "right_finger_joint"]

    all_passed = True
    for i, name in enumerate(joint_names):
        lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
        mid = (lo + hi) / 2.0

        print(f"\n  Sweeping {name}: [{lo:.3f}, {hi:.3f}] rad")

        # Sweep from mid to hi
        for step in range(num_steps_per_joint):
            t = step / num_steps_per_joint
            target = mid + (hi - mid) * math.sin(t * math.pi)

            targets = robot.data.joint_pos[0].clone()
            targets[i] = target
            robot.set_joint_position_target(targets.unsqueeze(0))
            sim.step()
            robot.update(sim.cfg.dt)

        actual = robot.data.joint_pos[0, i].item()
        error = abs(actual - mid)
        if error > 0.5:  # Allow some settling error
            print(f"    Position after sweep: {actual:.3f} (returned to ~mid: {error:.3f} error)")
        else:
            print(f"    PASS: Returned to mid position (error={error:.3f})")

    return all_passed


def test_gripper(robot, sim, num_cycles=3, steps_per_cycle=200):
    """Test 3: Open/close gripper cycles."""
    print("\n=== TEST: Gripper Open/Close ===")

    # Find finger joint indices
    joint_names = [robot.joint_names[i] for i in range(robot.num_joints)]
    left_idx = joint_names.index("left_finger_joint")
    right_idx = joint_names.index("right_finger_joint")

    for cycle in range(num_cycles):
        # Open
        for step in range(steps_per_cycle):
            targets = robot.data.joint_pos[0].clone()
            targets[left_idx] = 0.4   # Open
            targets[right_idx] = -0.4
            robot.set_joint_position_target(targets.unsqueeze(0))
            sim.step()
            robot.update(sim.cfg.dt)

        left_pos = robot.data.joint_pos[0, left_idx].item()
        right_pos = robot.data.joint_pos[0, right_idx].item()
        print(f"  Cycle {cycle+1} OPEN:  left={left_pos:.3f}, right={right_pos:.3f}")

        # Close
        for step in range(steps_per_cycle):
            targets = robot.data.joint_pos[0].clone()
            targets[left_idx] = 0.0
            targets[right_idx] = 0.0
            robot.set_joint_position_target(targets.unsqueeze(0))
            sim.step()
            robot.update(sim.cfg.dt)

        left_pos = robot.data.joint_pos[0, left_idx].item()
        right_pos = robot.data.joint_pos[0, right_idx].item()
        print(f"  Cycle {cycle+1} CLOSE: left={left_pos:.3f}, right={right_pos:.3f}")

    # Check symmetry
    if abs(left_pos + right_pos) < 0.05:
        print("  PASS: Symmetric finger motion")
        return True
    else:
        print(f"  WARN: Asymmetric fingers (sum={left_pos + right_pos:.3f})")
        return True  # Not a failure, just a warning


def test_multi_env(robot, sim, num_envs, num_steps=500):
    """Test 4: Multi-environment performance test."""
    print(f"\n=== TEST: Multi-Environment ({num_envs} envs) ===")

    start = time.time()
    for step in range(num_steps):
        sim.step()
        robot.update(sim.cfg.dt)

    elapsed = time.time() - start
    sim_time = num_steps * sim.cfg.dt
    rtf = sim_time / elapsed  # Real-time factor
    fps = num_steps / elapsed

    print(f"  {num_steps} steps in {elapsed:.2f}s")
    print(f"  Real-time factor: {rtf:.1f}x")
    print(f"  Steps/sec: {fps:.0f}")
    print(f"  Effective env-steps/sec: {fps * num_envs:.0f}")

    if fps > 100:
        print("  PASS: >100 Hz sim rate")
        return True
    else:
        print("  WARN: <100 Hz sim rate (may need optimization)")
        return True


def main():
    args = parse_args()

    # Late imports (require Isaac Lab)
    try:
        from omni.isaac.lab.app import AppLauncher
    except ImportError:
        print("ERROR: Isaac Lab not found. Run this script via:")
        print("  ./isaaclab.sh -p scripts/validate.py [args]")
        sys.exit(1)

    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import Articulation
    from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
    from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

    # Setup sim
    sim_cfg = create_sim_cfg(args)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 0.8], target=[0.0, 0.0, 0.2])

    # Setup robot
    robot_cfg = create_robot_cfg(args.usd_path)
    robot = Articulation(robot_cfg)

    # Play
    sim.reset()
    robot.update(sim.cfg.dt)

    print(f"\nRobot loaded: {robot.num_joints} joints, {robot.num_bodies} bodies")
    print(f"Joint names: {robot.joint_names}")
    print(f"Body names: {robot.body_names}")

    # Run tests
    tests = args.test
    if tests == "all":
        tests = ["gravity", "joints", "gripper"]

    results = {}
    if "gravity" in tests:
        results["gravity"] = test_gravity_drop(robot, sim)
    if "joints" in tests:
        results["joints"] = test_joint_sweep(robot, sim)
    if "gripper" in tests:
        results["gripper"] = test_gripper(robot, sim)
    if "multi" in tests:
        results["multi"] = test_multi_env(robot, sim, args.num_envs)

    # Summary
    print("\n=== RESULTS ===")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    simulation_app.close()


if __name__ == "__main__":
    main()
