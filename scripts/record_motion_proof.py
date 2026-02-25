"""Record motion proof videos with telemetry overlays for Alice 001."""

import argparse
import os

parser = argparse.ArgumentParser(description="Record motion proof")
parser.add_argument(
    "--mode",
    choices=["sweep", "policy_state", "policy_vision", "ball_drop", "collapse"],
    default="sweep",
)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=240)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output_dir", type=str, default="logs/videos/motion_proof")
parser.add_argument("--use_fabric", action="store_true", help="Enable sim.use_fabric")
parser.add_argument("--disable_overhead", action="store_true", help="Disable overhead camera")
parser.add_argument("--camera_size", type=int, default=640, help="Overhead camera square resolution")
parser.add_argument(
    "--camera_zoom",
    choices=["tight", "wide", "very_wide"],
    default="tight",
    help="Camera framing preset for overhead camera",
)
parser.add_argument("--action_gain", type=float, default=1.0, help="Scale sweep action amplitude")
parser.add_argument("--env_action_scale", type=float, default=-1.0, help="Override env action_scale if > 0")
parser.add_argument("--collapse_no_apply", action="store_true", help="Disable RL env _apply_action for collapse probes")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as iio
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import alice_ball_transfer  # noqa: F401


def draw_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle((6, 6, frame.shape[1] - 6, 42), fill=(0, 0, 0, 140))
        draw.text((12, 14), text, fill=(255, 255, 255))
        return np.asarray(img)
    except Exception:
        return frame


def save_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = iio.get_writer(path, fps=fps)
    for f in frames:
        w.append_data(f)
    w.close()


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args_cli.mode == "policy_vision":
        from alice_ball_transfer.ball_transfer_vision_env_cfg import BallTransferVisionEnvCfg
        from alice_ball_transfer.models.asymmetric_ac import AliceVisionPPORunnerCfg
        env_cfg = BallTransferVisionEnvCfg()
        env_id = "Alice-Ball-Transfer-Vision-v0"
        runner_cfg = AliceVisionPPORunnerCfg()
    else:
        from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg
        from alice_ball_transfer.ball_transfer_env_cfg import BALL_CFG, ALICE_001_CFG
        from alice_ball_transfer.agents.rsl_rl_ppo_cfg import AliceBallTransferPPORunnerCfg
        env_cfg = BallTransferEnvCfg()
        env_id = "Alice-Ball-Transfer-Direct-v0"
        runner_cfg = AliceBallTransferPPORunnerCfg()

        if args_cli.mode == "ball_drop":
            env_cfg.sim.gravity = (0.0, 0.0, -9.81)
            BALL_CFG.spawn.rigid_props.kinematic_enabled = False
            # Spawn ball above workspace so falling motion is obvious.
            BALL_CFG.init_state.pos = (-0.08, 0.0, 0.52)

        if args_cli.mode == "collapse":
            env_cfg.sim.gravity = (0.0, 0.0, -9.81)
            BALL_CFG.spawn.rigid_props.kinematic_enabled = False
            BALL_CFG.init_state.pos = (-0.08, 0.0, 0.52)
            # Reduce drive hold so arm is free to sag/fall under gravity.
            ALICE_001_CFG.actuators["arm"].stiffness = 0.0
            ALICE_001_CFG.actuators["arm"].damping = 0.0
            ALICE_001_CFG.actuators["gripper"].stiffness = 0.0
            ALICE_001_CFG.actuators["gripper"].damping = 0.0

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.use_fabric = args_cli.use_fabric

    if args_cli.env_action_scale > 0:
        env_cfg.action_scale = args_cli.env_action_scale

    # Camera framing presets. Wider framing helps verify whole-arm motion.
    if hasattr(env_cfg, "overhead_camera") and env_cfg.overhead_camera is not None:
        if args_cli.disable_overhead:
            env_cfg.overhead_camera = None
        else:
            env_cfg.overhead_camera.height = args_cli.camera_size
            env_cfg.overhead_camera.width = args_cli.camera_size
            if args_cli.camera_zoom == "tight":
                env_cfg.overhead_camera.offset.pos = (-0.07, 0.0, 0.43)
            elif args_cli.camera_zoom == "wide":
                env_cfg.overhead_camera.offset.pos = (-0.10, 0.0, 0.60)
            else:
                env_cfg.overhead_camera.offset.pos = (-0.14, 0.0, 0.80)

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    raw_env = env.unwrapped

    if args_cli.mode == "collapse" and args_cli.collapse_no_apply:
        raw_env._apply_action = lambda: None

    policy = None
    if args_cli.mode.startswith("policy"):
        if not args_cli.checkpoint:
            raise RuntimeError("--checkpoint is required for policy modes")
        from rsl_rl.runners import OnPolicyRunner
        ckpt = os.path.expanduser(args_cli.checkpoint)
        runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=os.path.dirname(ckpt), device=device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    n_envs = args_cli.num_envs
    n_actions = 7

    overhead_frames = []
    wrist_frames = []

    ball_z_series = []
    for step in range(args_cli.num_steps):
        if args_cli.mode == "sweep":
            actions = torch.zeros(n_envs, n_actions, device=device)
            j = (step // 40) % 5
            sign = 1.0 if ((step // 20) % 2 == 0) else -1.0
            actions[:, j] = float(np.clip(sign * args_cli.action_gain, -1.0, 1.0))
        elif args_cli.mode in ("ball_drop", "collapse"):
            actions = torch.zeros(n_envs, n_actions, device=device)
        else:
            with torch.no_grad():
                actions = policy(obs)

        obs, rewards, _, _ = env.step(actions)

        jp = raw_env.robot.data.joint_pos[0].detach().cpu().numpy()
        ee = raw_env._get_ee_pos_local()[0].detach().cpu().numpy()
        ball = raw_env._get_ball_pos_local()[0].detach().cpu().numpy()
        dist = float(np.linalg.norm(ee - ball))
        ball_z_series.append(float(ball[2]))

        overlay = (
            f"step={step} b={jp[0]:+.2f} s={jp[1]:+.2f} e={jp[2]:+.2f} "
            f"wp={jp[3]:+.2f} wr={jp[4]:+.2f} dist={dist:.3f}"
        )

        if hasattr(raw_env, "overhead_camera"):
            rgb = raw_env.overhead_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
            overhead_frames.append(draw_overlay(rgb, overlay))

        if hasattr(raw_env, "wrist_camera"):
            rgb = raw_env.wrist_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
            wrist_frames.append(draw_overlay(rgb, overlay))

    out_dir = os.path.expanduser(args_cli.output_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if overhead_frames:
        save_video(overhead_frames, os.path.join(out_dir, f"{args_cli.mode}_overhead.mp4"), args_cli.fps)
    if wrist_frames:
        save_video(wrist_frames, os.path.join(out_dir, f"{args_cli.mode}_wrist.mp4"), args_cli.fps)

    # Save lightweight telemetry summary
    summary = {
        "mode": args_cli.mode,
        "steps": args_cli.num_steps,
        "overhead_frames": len(overhead_frames),
        "wrist_frames": len(wrist_frames),
        "ball_z_min": float(np.min(ball_z_series)) if ball_z_series else None,
        "ball_z_max": float(np.max(ball_z_series)) if ball_z_series else None,
        "ball_drop_amount": (float(ball_z_series[0] - ball_z_series[-1]) if len(ball_z_series) > 1 else None),
    }
    import json
    with open(os.path.join(out_dir, f"{args_cli.mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved motion proof to", out_dir)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
