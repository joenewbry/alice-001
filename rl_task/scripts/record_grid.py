"""Record multi-env grid video — the classic RL visualization.

Runs N environments simultaneously with overhead cameras, composites all views
into a grid layout, and saves as a single video. Great for verifying that the
policy generalizes across parallel environments.

Usage:
    python record_grid.py --checkpoint .../model_5749.pt --num_envs 16 --num_steps 600
    python record_grid.py --checkpoint .../model_5749.pt --num_envs 9 --grid_cols 3
    python record_grid.py --checkpoint .../model_10000.pt --vision --num_envs 16
"""

import argparse
import math
import os

parser = argparse.ArgumentParser(description="Record multi-env grid evaluation video")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--vision", action="store_true")
parser.add_argument("--grid_cols", type=int, default=None, help="Grid columns (default: sqrt(num_envs))")
parser.add_argument("--tile_size", type=int, default=480, help="Pixels per tile (overhead camera resolution)")
parser.add_argument("--fps", type=int, default=30)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch
import numpy as np

import alice_ball_transfer  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def composite_grid(frames_all_envs: np.ndarray, cols: int) -> np.ndarray:
    """Composite N frames into a grid image.

    Args:
        frames_all_envs: (N, H, W, 3) uint8 array
        cols: number of columns in the grid

    Returns:
        (rows*H, cols*W, 3) uint8 array
    """
    n, h, w, c = frames_all_envs.shape
    rows = math.ceil(n / cols)

    # Pad with black if n doesn't fill the grid
    if n < rows * cols:
        pad = np.zeros((rows * cols - n, h, w, c), dtype=np.uint8)
        frames_all_envs = np.concatenate([frames_all_envs, pad], axis=0)

    # Reshape into grid: (rows, cols, H, W, 3) -> (rows*H, cols*W, 3)
    grid = frames_all_envs.reshape(rows, cols, h, w, c)
    grid = grid.transpose(0, 2, 1, 3, 4)  # (rows, H, cols, W, 3)
    grid = grid.reshape(rows * h, cols * w, c)
    return grid


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

    # use_fabric=False is already set in the base config

    # Set overhead camera resolution
    env_cfg.overhead_camera.height = args_cli.tile_size
    env_cfg.overhead_camera.width = args_cli.tile_size

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(args_cli.checkpoint)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    # Grid layout
    cols = args_cli.grid_cols or int(math.ceil(math.sqrt(args_cli.num_envs)))
    rows = math.ceil(args_cli.num_envs / cols)
    grid_w = cols * args_cli.tile_size
    grid_h = rows * args_cli.tile_size

    # Output directory
    ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    if args_cli.output_dir is None:
        args_cli.output_dir = os.path.expanduser(f"~/Alice-001/logs/videos/{ckpt_name}")
    os.makedirs(args_cli.output_dir, exist_ok=True)

    raw_env = env.unwrapped
    if not hasattr(raw_env, 'overhead_camera'):
        print("ERROR: No overhead camera found in environment. Cannot create grid video.")
        env.close()
        simulation_app.close()
        return

    # ── Collect grid frames ──
    print(f"\nRecording {args_cli.num_steps} steps, {args_cli.num_envs} envs ({rows}x{cols} grid)")
    print(f"Grid resolution: {grid_w}x{grid_h}")
    print(f"Output: {args_cli.output_dir}")

    grid_frames = []
    obs = env.get_observations()

    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)

        # Capture overhead camera from ALL envs (every 2nd step for ~30fps)
        if step % 2 == 0:
            rgb = raw_env.overhead_camera.data.output["rgb"]  # (N, H, W, 4) uint8
            frames = rgb[:, :, :, :3].cpu().numpy()  # (N, H, W, 3)
            grid_frame = composite_grid(frames, cols)
            grid_frames.append(grid_frame)

        if step % 100 == 0:
            extras = raw_env.extras.get("log", {})
            print(f"  Step {step}: grasp={extras.get('pct_grasping', 0):.3f}, "
                  f"lift={extras.get('pct_lifted', 0):.3f}, reward={rewards.mean().item():.2f}")

    # ── Save grid video ──
    output_name = f"grid_{args_cli.num_envs}env.mp4"
    video_path = os.path.join(args_cli.output_dir, output_name)
    print(f"\nSaving grid video ({len(grid_frames)} frames, {grid_w}x{grid_h})...")

    try:
        import imageio
        writer = imageio.get_writer(video_path, fps=args_cli.fps)
        for frame in grid_frames:
            writer.append_data(frame)
        writer.close()
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"Grid video saved: {video_path} ({size_mb:.1f} MB)")
    except ImportError:
        print("imageio not available, trying ffmpeg fallback...")
        frames_dir = os.path.join(args_cli.output_dir, "grid_frames")
        os.makedirs(frames_dir, exist_ok=True)
        try:
            from PIL import Image
            for i, frame in enumerate(grid_frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            os.system(f"ffmpeg -y -framerate {args_cli.fps} -i {frames_dir}/frame_%04d.png "
                     f"-c:v libx264 -pix_fmt yuv420p {video_path} 2>/dev/null")
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"Grid video assembled: {video_path} ({size_mb:.1f} MB)")
        except ImportError:
            npz_path = video_path.replace(".mp4", ".npz")
            np.savez_compressed(npz_path, frames=np.stack(grid_frames))
            print(f"Frames saved as numpy: {npz_path}")

    print(f"\nDone! {len(grid_frames)} grid frames recorded.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
