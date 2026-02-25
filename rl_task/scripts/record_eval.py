"""Record evaluation of trained policy — captures overhead + wrist camera views + metrics.

Works headlessly on cloud GPUs. Saves:
- Overhead camera video (third-person bird's-eye view) as MP4
- Wrist camera video (what the policy sees, vision env only) as MP4
- Episode metrics as CSV

Usage:
    python record_eval.py --checkpoint .../model_10497.pt --vision --num_steps 600
    python record_eval.py --checkpoint .../model_5749.pt --num_steps 600
    python record_eval.py --checkpoint .../model_5749.pt --num_steps 600 --no_overhead
"""

import argparse
import os

parser = argparse.ArgumentParser(description="Record policy evaluation")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--vision", action="store_true")
parser.add_argument("--no_overhead", action="store_true", help="Skip overhead camera capture")
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
import csv

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

import alice_ball_transfer  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def save_video(frames, path, fps):
    """Save frames as MP4 video with imageio, falling back to ffmpeg/numpy."""
    if not frames:
        return
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Video saved: {path} ({size_mb:.1f} MB, {len(frames)} frames)")
    except ImportError:
        print("  imageio not available, trying ffmpeg fallback...")
        frames_dir = path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        try:
            from PIL import Image
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%04d.png "
                     f"-c:v libx264 -pix_fmt yuv420p {path} 2>/dev/null")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  Video assembled: {path} ({size_mb:.1f} MB)")
        except ImportError:
            npz_path = path.replace(".mp4", ".npz")
            np.savez_compressed(npz_path, frames=np.stack(frames))
            print(f"  Frames saved as numpy: {npz_path}")


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

    # use_fabric=False is now set in the base config — cameras need it

    # Disable overhead camera if requested
    if args_cli.no_overhead:
        env_cfg.overhead_camera = None

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(args_cli.checkpoint)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    # Output directory
    ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    if args_cli.output_dir is None:
        args_cli.output_dir = os.path.expanduser(f"~/Alice-001/logs/videos/{ckpt_name}")
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Check what cameras are available
    raw_env = env.unwrapped
    has_overhead = hasattr(raw_env, 'overhead_camera')
    has_wrist = hasattr(raw_env, 'wrist_camera')

    # ── Collect frames and metrics ──
    print(f"\nRecording {args_cli.num_steps} steps from {args_cli.checkpoint}...")
    print(f"Output: {args_cli.output_dir}")
    print(f"Cameras: overhead={'yes' if has_overhead else 'no'}, wrist={'yes' if has_wrist else 'no'}")

    overhead_frames = []
    wrist_frames = []
    metrics_rows = []
    obs = env.get_observations()

    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)

        # Capture overhead camera frames (every 2nd step for ~30fps from 60Hz sim)
        if has_overhead and step % 2 == 0:
            rgb = raw_env.overhead_camera.data.output["rgb"]  # (N, H, W, 4) uint8
            frame = rgb[0, :, :, :3].cpu().numpy()
            overhead_frames.append(frame)

        # Capture wrist camera frames (vision env only)
        if has_wrist and step % 2 == 0:
            rgb = raw_env.wrist_camera.data.output["rgb"]  # (N, H, W, 4) uint8
            frame = rgb[0, :, :, :3].cpu().numpy()
            wrist_frames.append(frame)

        # Collect metrics every step
        ee_pos = raw_env._get_ee_pos_local()[0].cpu().numpy()
        ball_pos = raw_env._get_ball_pos_local()[0].cpu().numpy()
        joint_pos = raw_env.robot.data.joint_pos[0].cpu().numpy()
        ee_to_ball = np.linalg.norm(ball_pos - ee_pos)
        extras = raw_env.extras.get("log", {})

        row = {
            "step": step,
            "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],
            "ball_x": ball_pos[0], "ball_y": ball_pos[1], "ball_z": ball_pos[2],
            "ee_to_ball_dist": ee_to_ball,
            "reward": rewards[0].item(),
            "pct_grasping": extras.get("pct_grasping", 0),
            "pct_lifted": extras.get("pct_lifted", 0),
            "reach_reward": extras.get("reward/reach", 0),
            "grasp_reward": extras.get("reward/grasp", 0),
            "lift_reward": extras.get("reward/lift", 0),
        }
        for i, jname in enumerate(["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "left_finger", "right_finger"]):
            row[f"joint_{jname}"] = joint_pos[i]
        metrics_rows.append(row)

        if step % 100 == 0:
            print(f"  Step {step}: ee_to_ball={ee_to_ball:.4f}, grasp={extras.get('pct_grasping', 0):.3f}, "
                  f"lift={extras.get('pct_lifted', 0):.3f}, reward={rewards[0].item():.2f}")

    # ── Save overhead camera video ──
    if overhead_frames:
        print(f"\nSaving overhead camera video ({len(overhead_frames)} frames)...")
        save_video(overhead_frames, os.path.join(args_cli.output_dir, "overhead.mp4"), args_cli.fps)

    # ── Save wrist camera video ──
    if wrist_frames:
        print(f"Saving wrist camera video ({len(wrist_frames)} frames)...")
        save_video(wrist_frames, os.path.join(args_cli.output_dir, "wrist_camera.mp4"), args_cli.fps)

    # ── Save metrics CSV ──
    csv_path = os.path.join(args_cli.output_dir, "eval_metrics.csv")
    if metrics_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"Metrics CSV: {csv_path} ({len(metrics_rows)} rows)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {ckpt_name}")
    print(f"{'='*60}")
    dists = [r["ee_to_ball_dist"] for r in metrics_rows]
    rewards_list = [r["reward"] for r in metrics_rows]
    grasps = [r["pct_grasping"] for r in metrics_rows]
    lifts = [r["pct_lifted"] for r in metrics_rows]
    print(f"  Mean EE-to-ball distance: {np.mean(dists):.4f} m")
    print(f"  Min EE-to-ball distance:  {np.min(dists):.4f} m")
    print(f"  Mean reward per step:     {np.mean(rewards_list):.4f}")
    print(f"  Mean grasping rate:       {np.mean(grasps):.4f}")
    print(f"  Mean lift rate:           {np.mean(lifts):.4f}")
    print(f"  Overhead frames:          {len(overhead_frames)}")
    print(f"  Wrist camera frames:      {len(wrist_frames)}")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
