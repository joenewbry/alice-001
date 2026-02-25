"""Training entry point for vision-based ball transfer (Stages 2 & 3).

Stage 2: Vision RL with asymmetric actor-critic
    python ~/Alice-001/rl_task/scripts/train_vision.py --num_envs 256

Stage 3: Enable domain randomization (continue from Stage 2 checkpoint)
    python ~/Alice-001/rl_task/scripts/train_vision.py --num_envs 256 \
        --domain_rand --resume ~/Alice-001/logs/alice_ball_transfer_vision/

Curriculum (staged training):
    # Phase 1: Reach only (learn to use camera)
    python ~/Alice-001/rl_task/scripts/train_vision.py --num_envs 256 \
        --reach_only --max_iterations 1000

    # Phase 2: Reach + grasp
    python ~/Alice-001/rl_task/scripts/train_vision.py --num_envs 256 \
        --reach_grasp --max_iterations 3000 --resume <checkpoint>

    # Phase 3: Full pipeline
    python ~/Alice-001/rl_task/scripts/train_vision.py --num_envs 256 \
        --max_iterations 10000 --resume <checkpoint>

    # Phase 4: Add domain randomization
    python ~/Alice-001/rl_task/scripts/train_vision.py --num_envs 256 \
        --domain_rand --max_iterations 15000 --resume <checkpoint>
"""

import argparse
import os

# ── CLI args (before AppLauncher) ──
parser = argparse.ArgumentParser(description="Train Alice 001 vision ball transfer")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--reach_only", action="store_true", help="Only enable reach reward")
parser.add_argument("--reach_grasp", action="store_true", help="Enable reach + grasp only")
parser.add_argument("--domain_rand", action="store_true", help="Enable domain randomization (Stage 3)")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume")
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True  # Required for camera sensor rendering

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

# ── Imports after AppLauncher ──
import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register tasks
import alice_ball_transfer  # noqa: F401

from alice_ball_transfer.ball_transfer_vision_env_cfg import BallTransferVisionEnvCfg
from alice_ball_transfer.models.asymmetric_ac import AliceVisionPPORunnerCfg


def main():
    # Build env config
    env_cfg = BallTransferVisionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Domain randomization (Stage 3)
    if args_cli.domain_rand:
        env_cfg.enable_domain_rand = True
        print("[DR] Domain randomization ENABLED")

    # Curriculum: disable later-phase rewards
    if args_cli.reach_only:
        env_cfg.grasp_reward_scale = 0.0
        env_cfg.lift_reward_scale = 0.0
        env_cfg.transport_reward_scale = 0.0
        env_cfg.drop_reward_scale = 0.0
        env_cfg.action_penalty_scale = 0.005
        env_cfg.velocity_penalty_scale = 0.0005
        print("[CURRICULUM] Vision reach-only mode")
    elif args_cli.reach_grasp:
        env_cfg.lift_reward_scale = 0.0
        env_cfg.transport_reward_scale = 0.0
        env_cfg.drop_reward_scale = 0.0
        print("[CURRICULUM] Vision reach + grasp mode")

    # Create environment
    env = gym.make("Alice-Ball-Transfer-Vision-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Configure runner
    runner_cfg = AliceVisionPPORunnerCfg()
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
    mode = "reach-only" if args_cli.reach_only else "reach+grasp" if args_cli.reach_grasp else "full"
    print(f"\n{'='*60}")
    print(f"Training: Alice 001 Vision Ball Transfer")
    print(f"  Envs: {args_cli.num_envs}")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"  Mode: {mode}")
    print(f"  Domain rand: {args_cli.domain_rand}")
    print(f"  Actor obs: {env_cfg.num_observations} (512 visual + 14 proprio)")
    print(f"  Critic obs: {env_cfg.num_states} (privileged state)")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
