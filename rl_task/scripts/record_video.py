"""Record video of trained policy for visual inspection.

Renders the simulation and saves frames as a video file using Isaac Lab's
built-in offscreen rendering. Works headlessly on cloud GPUs.

Usage:
    python ~/Alice-001/rl_task/scripts/record_video.py \
        --checkpoint ~/Alice-001/logs/alice_ball_transfer/model_2999.pt \
        --num_envs 4 --num_steps 500 \
        --output ~/Alice-001/logs/videos/stage1_model2999.mp4

    # Vision policy
    python ~/Alice-001/rl_task/scripts/record_video.py \
        --checkpoint ~/Alice-001/logs/alice_ball_transfer_vision/model_5000.pt \
        --vision --num_envs 4 --num_steps 500 \
        --output ~/Alice-001/logs/videos/stage2_vision.mp4
"""

import argparse
import os

parser = argparse.ArgumentParser(description="Record policy video")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--vision", action="store_true", help="Use vision environment")
parser.add_argument("--fps", type=int, default=30)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True  # Required for offscreen rendering

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch
import numpy as np

import alice_ball_transfer  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    # Create environment
    if args_cli.vision:
        from alice_ball_transfer.ball_transfer_vision_env_cfg import BallTransferVisionEnvCfg
        from alice_ball_transfer.models.asymmetric_ac import AliceVisionPPORunnerCfg
        env_cfg = BallTransferVisionEnvCfg()
        runner_cfg = AliceVisionPPORunnerCfg()
        env_id = "Alice-Ball-Transfer-Vision-v0"
    else:
        from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg
        from alice_ball_transfer.agents.rsl_rl_ppo_cfg import AliceBallTransferPPORunnerCfg
        env_cfg = BallTransferEnvCfg()
        runner_cfg = AliceBallTransferPPORunnerCfg()
        env_id = "Alice-Ball-Transfer-Direct-v0"

    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(args_cli.checkpoint)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    # Set up output path
    if args_cli.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        args_cli.output = os.path.join(
            os.path.expanduser("~/Alice-001/logs/videos"),
            f"{ckpt_name}.mp4"
        )
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)

    # Try to capture frames using Isaac Sim's viewport
    frames = []
    print(f"\nRecording {args_cli.num_steps} steps from {args_cli.checkpoint}...")

    obs, _ = env.get_observations()
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        # Capture frame from simulation render
        if step % 2 == 0:  # Every other frame to reduce size
            try:
                # Try Isaac Sim viewport capture
                from isaacsim.core.utils.viewports import get_viewport_data
                frame = get_viewport_data()
                if frame is not None:
                    frames.append(frame)
            except (ImportError, Exception):
                pass

        if step % 100 == 0:
            raw_env = env.unwrapped
            ball_pos = raw_env._get_ball_pos_local()[0].cpu().numpy()
            ee_pos = raw_env._get_ee_pos_local()[0].cpu().numpy()
            print(f"  Step {step}: EE={ee_pos.round(3)}, Ball={ball_pos.round(3)}")

    # Save video if we captured frames
    if frames:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args_cli.output, fourcc, args_cli.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            size_mb = os.path.getsize(args_cli.output) / (1024 * 1024)
            print(f"\nVideo saved: {args_cli.output} ({size_mb:.1f} MB, {len(frames)} frames)")
        except ImportError:
            print("OpenCV not available for video writing")
    else:
        # Fallback: save TensorBoard-compatible metrics log
        print("\nNo viewport frames captured (headless mode)")
        print("Saving episode metrics instead...")

        # Run a few full episodes and log results
        metrics_path = args_cli.output.replace('.mp4', '_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Checkpoint: {args_cli.checkpoint}\n")
            f.write(f"Steps: {args_cli.num_steps}\n\n")

            obs, _ = env.get_observations()
            episode_rewards = []
            current_rewards = torch.zeros(args_cli.num_envs, device='cuda:0')

            for step in range(args_cli.num_steps):
                with torch.no_grad():
                    actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
                current_rewards += rewards

                # Check for episode ends
                for i in range(args_cli.num_envs):
                    if dones[i]:
                        episode_rewards.append(current_rewards[i].item())
                        current_rewards[i] = 0.0

                if step % 100 == 0:
                    raw_env = env.unwrapped
                    extras = raw_env.extras.get("log", {})
                    f.write(f"Step {step}:\n")
                    for k, v in extras.items():
                        f.write(f"  {k}: {v:.4f}\n")

            f.write(f"\nCompleted episodes: {len(episode_rewards)}\n")
            if episode_rewards:
                f.write(f"Mean episode reward: {np.mean(episode_rewards):.4f}\n")
                f.write(f"Min: {min(episode_rewards):.4f}, Max: {max(episode_rewards):.4f}\n")

        print(f"Metrics saved: {metrics_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
