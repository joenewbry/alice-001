"""Training entry point for Alice 001 ball transfer task.

Usage:
    python ~/Alice-001/rl_task/scripts/train.py --num_envs 2048
    python ~/Alice-001/rl_task/scripts/train.py --num_envs 2048 --max_iterations 500

Curriculum (staged training):
    python ~/Alice-001/rl_task/scripts/train.py --num_envs 2048 --reach_only --max_iterations 500
    python ~/Alice-001/rl_task/scripts/train.py --num_envs 2048 --reach_grasp --max_iterations 1500
    python ~/Alice-001/rl_task/scripts/train.py --num_envs 2048 --max_iterations 5000
"""

import argparse
import os

# ── CLI args (before AppLauncher) ──
parser = argparse.ArgumentParser(description="Train Alice 001 ball transfer")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--reach_only", action="store_true", help="Only enable reach reward")
parser.add_argument("--reach_grasp", action="store_true", help="Enable reach + grasp only")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume")
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # Training is always headless

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

# ── Imports after AppLauncher ──
import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register the task
import alice_ball_transfer  # noqa: F401

from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg
from alice_ball_transfer.agents.rsl_rl_ppo_cfg import AliceBallTransferPPORunnerCfg


def main():
    # Build env config with overrides
    env_cfg = BallTransferEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Curriculum: disable later-phase rewards and reduce penalties
    if args_cli.reach_only:
        env_cfg.grasp_reward_scale = 0.0
        env_cfg.lift_reward_scale = 0.0
        env_cfg.transport_reward_scale = 0.0
        env_cfg.drop_reward_scale = 0.0
        # Moderate penalties to prevent wild flailing
        env_cfg.action_penalty_scale = 0.005
        env_cfg.velocity_penalty_scale = 0.0005
        print("[CURRICULUM] Reach-only mode (penalties reduced)")
    elif args_cli.reach_grasp:
        env_cfg.lift_reward_scale = 0.0
        env_cfg.transport_reward_scale = 0.0
        env_cfg.drop_reward_scale = 0.0
        print("[CURRICULUM] Reach + grasp mode")

    # Disable cameras for headless training (prevents rendering stalls on GCP VMs)
    env_cfg.overhead_camera = None
    env_cfg.side_camera = None

    # Create environment
    env = gym.make("Alice-Ball-Transfer-Direct-v0", cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Configure runner
    runner_cfg = AliceBallTransferPPORunnerCfg()
    runner_cfg.max_iterations = args_cli.max_iterations
    runner_cfg.seed = args_cli.seed

    # Log directory
    log_dir = os.path.join(
        os.path.expanduser("~/Alice-001/logs"),
        runner_cfg.experiment_name,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create RSL-RL runner
    from rsl_rl.runners import OnPolicyRunner
    import glob

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Resume from checkpoint if specified
    if args_cli.resume:
        resume_dir = os.path.expanduser(args_cli.resume)
        # Find latest checkpoint
        checkpoints = sorted(glob.glob(os.path.join(resume_dir, "model_*.pt")),
                             key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"[RESUME] Loading checkpoint: {latest}")
            runner.load(latest)
            print(f"[RESUME] Resuming from iteration {runner.current_learning_iteration}")
        else:
            print(f"[RESUME] WARNING: No checkpoints found in {resume_dir}")

    # Train
    print(f"\n{'='*60}")
    print(f"Training: Alice 001 Ball Transfer")
    print(f"  Envs: {args_cli.num_envs}")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"  Log dir: {log_dir}")
    if args_cli.reach_only:
        print(f"  Mode: reach-only")
    elif args_cli.reach_grasp:
        print(f"  Mode: reach + grasp")
    else:
        print(f"  Mode: full pipeline")
    print(f"{'='*60}\n")

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
