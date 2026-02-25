"""Sweep joints through large angles to verify the sim renders motion.

Usage:
    python rl_task/scripts/joint_test.py --num_envs 4
"""

import argparse
import math

parser = argparse.ArgumentParser(description="Joint sweep test")
parser.add_argument("--num_envs", type=int, default=4)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No cameras needed for interactive viewport viewing

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import time
import torch
import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import alice_ball_transfer  # noqa: F401
from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg


def main():
    env_cfg = BallTransferEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # use_fabric=True so the viewport actually renders motion
    # (use_fabric=False is needed for cameras but breaks viewport updates)
    env_cfg.sim.use_fabric = True
    env_cfg.overhead_camera = None  # Skip camera (incompatible with fabric)
    env = gym.make("Alice-Ball-Transfer-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    raw_env = env.unwrapped

    n_envs = args_cli.num_envs
    n_actions = 7  # 5 arm joints + 2 finger joints

    print("\n=== JOINT SWEEP TEST ===")
    print("Watch the viewport — you should see large arm movements.\n")

    obs = env.get_observations()

    # Sweep each joint one at a time with large actions
    joint_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "left_finger", "right_finger"]

    for joint_idx in range(5):  # Only arm joints
        print(f"--- Sweeping {joint_names[joint_idx]} (joint {joint_idx}) ---")

        # Move positive
        for step in range(60):
            actions = torch.zeros(n_envs, n_actions, device="cuda:0")
            actions[:, joint_idx] = 1.0  # Full positive
            obs, _, _, _ = env.step(actions)
            time.sleep(1 / 60)

        # Move negative
        for step in range(120):
            actions = torch.zeros(n_envs, n_actions, device="cuda:0")
            actions[:, joint_idx] = -1.0  # Full negative
            obs, _, _, _ = env.step(actions)
            time.sleep(1 / 60)

        # Return to center
        for step in range(60):
            actions = torch.zeros(n_envs, n_actions, device="cuda:0")
            actions[:, joint_idx] = 1.0
            obs, _, _, _ = env.step(actions)
            time.sleep(1 / 60)

        jp = raw_env.robot.data.joint_pos[0].cpu().numpy()
        print(f"  Joint positions: {jp.round(3)}")

    # Now do a combined wave motion
    print("\n--- Combined wave motion (all joints) ---")
    for step in range(300):
        actions = torch.zeros(n_envs, n_actions, device="cuda:0")
        t = step / 60.0
        actions[:, 0] = math.sin(t * 2.0)        # base
        actions[:, 1] = math.sin(t * 1.5)         # shoulder
        actions[:, 2] = math.cos(t * 2.0)         # elbow
        actions[:, 3] = math.sin(t * 3.0)         # wrist pitch
        actions[:, 4] = math.cos(t * 2.5)         # wrist roll
        obs, _, _, _ = env.step(actions)
        time.sleep(1 / 60)

        if step % 60 == 0:
            jp = raw_env.robot.data.joint_pos[0].cpu().numpy()
            print(f"  Step {step}: joints={jp.round(3)}")

    print("\nDone! Closing in 5 seconds...")
    time.sleep(5)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
