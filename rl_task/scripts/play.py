"""Play back a trained policy for Alice 001 ball transfer.

Usage:
    python ~/Alice-001/rl_task/scripts/play.py \
        --checkpoint ~/Alice-001/logs/alice_ball_transfer/model_5000.pt \
        --num_envs 4
"""

import argparse
import os

parser = argparse.ArgumentParser(description="Play Alice 001 ball transfer policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pt")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=1000)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No --enable_cameras needed: interactive viewport uses Fabric rendering

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import time
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import alice_ball_transfer  # noqa: F401

from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg
from alice_ball_transfer.agents.rsl_rl_ppo_cfg import AliceBallTransferPPORunnerCfg


def main():
    env_cfg = BallTransferEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # use_fabric=True so the viewport renders motion in real-time
    # (use_fabric=False is for camera sensors / headless recording only)
    env_cfg.sim.use_fabric = True
    env_cfg.overhead_camera = None  # Skip camera sensor (incompatible with fabric)
    env = gym.make("Alice-Ball-Transfer-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    runner_cfg = AliceBallTransferPPORunnerCfg()
    log_dir = os.path.dirname(args_cli.checkpoint)

    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.load(args_cli.checkpoint)

    policy = runner.get_inference_policy(device="cuda:0")

    # Run — pace to real-time so you can watch in the viewport
    # Sim: dt=1/120, decimation=2 → each env.step = 1/60s sim time
    step_dt = 1.0 / 60.0
    obs = env.get_observations()
    raw_env = env.unwrapped
    print(f"\nRunning {args_cli.num_steps} steps at ~real-time. Watch the viewport!")
    print("Press Ctrl+C in terminal to stop early.\n")

    try:
        for step in range(args_cli.num_steps):
            t0 = time.time()
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Sleep to match real-time
            elapsed = time.time() - t0
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            if step % 100 == 0:
                ball_pos = raw_env._get_ball_pos_local()[0].cpu().numpy()
                ee_pos = raw_env._get_ee_pos_local()[0].cpu().numpy()
                dist = ((ee_pos - ball_pos) ** 2).sum() ** 0.5
                print(
                    f"Step {step}: EE={ee_pos.round(3)}, Ball={ball_pos.round(3)}, dist={dist:.4f}"
                )
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\nDone. Closing in 3 seconds...")
    time.sleep(3)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
