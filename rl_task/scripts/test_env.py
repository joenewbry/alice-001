"""Smoke test for the Alice 001 ball transfer RL environment.

Verifies:
1. Robot, table, ball spawn correctly
2. Observation and reward shapes are correct
3. Random actions don't crash the simulation
4. Reset works properly

Usage:
    python ~/Alice-001/rl_task/scripts/test_env.py --num_envs 4
"""

import argparse

parser = argparse.ArgumentParser(description="Smoke test ball transfer env")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=100)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch

import alice_ball_transfer  # noqa: F401

from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg


def main():
    env_cfg = BallTransferEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Alice-Ball-Transfer-Direct-v0", cfg=env_cfg)

    num_envs = args_cli.num_envs
    num_obs = 28
    num_actions = 7

    print(f"\n{'='*60}")
    print(f"Smoke Test: Alice 001 Ball Transfer Environment")
    print(f"  Num envs: {num_envs}")
    print(f"{'='*60}")

    # Check observation space
    obs, info = env.reset()
    obs_tensor = obs["policy"]
    print(f"\n[CHECK] Observation shape: {obs_tensor.shape}")
    assert obs_tensor.shape == (num_envs, num_obs), (
        f"Expected ({num_envs}, {num_obs}), got {obs_tensor.shape}"
    )
    print("  PASS: Observation shape correct")

    # Check action space
    print(f"\n[CHECK] Action space: {env.action_space}")
    assert env.action_space.shape[-1] == num_actions
    print("  PASS: Action space correct")

    # Run random actions
    print(f"\n[CHECK] Running {args_cli.num_steps} random steps...")
    total_reward = torch.zeros(num_envs, device=obs_tensor.device)
    num_resets = 0

    for step in range(args_cli.num_steps):
        actions = torch.randn(num_envs, num_actions, device=obs_tensor.device) * 0.5
        obs, reward, terminated, truncated, info = env.step(actions)

        obs_tensor = obs["policy"]
        assert obs_tensor.shape == (num_envs, num_obs), f"Step {step}: bad obs shape"
        assert reward.shape == (num_envs,), f"Step {step}: bad reward shape"

        total_reward += reward
        num_resets += (terminated | truncated).sum().item()

        if step % 25 == 0:
            raw_env = env.unwrapped
            ball_pos = raw_env._get_ball_pos_local()[0].cpu().numpy()
            ee_pos = raw_env._get_ee_pos_local()[0].cpu().numpy()
            print(
                f"  Step {step:3d}: reward={reward.mean():.4f}, "
                f"EE={ee_pos.round(3)}, Ball={ball_pos.round(3)}"
            )

    print(f"\n  PASS: {args_cli.num_steps} steps completed without crash")
    print(f"  Total resets: {num_resets}")
    print(f"  Mean cumulative reward: {total_reward.mean():.4f}")

    # Test explicit reset
    print(f"\n[CHECK] Explicit reset...")
    obs, info = env.reset()
    assert obs["policy"].shape == (num_envs, num_obs)
    print("  PASS: Reset works")

    # Summary
    print(f"\n{'='*60}")
    print("ALL SMOKE TESTS PASSED")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
