"""Deterministic articulation motion validation for Alice 001.

Outputs CSV + summary and enforces minimum motion ranges.
"""

import argparse
import csv
import os

parser = argparse.ArgumentParser(description="Validate articulation motion")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_csv", type=str, default="logs/recovery/articulation_motion.csv")
parser.add_argument("--min_joint_range", type=float, default=0.20)
parser.add_argument("--min_ee_disp", type=float, default=0.015)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import alice_ball_transfer  # noqa: F401
from alice_ball_transfer.ball_transfer_env_cfg import BallTransferEnvCfg


def main():
    env_cfg = BallTransferEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.overhead_camera = None
    env_cfg.sim.use_fabric = False

    env = gym.make("Alice-Ball-Transfer-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    raw_env = env.unwrapped

    n_envs = args_cli.num_envs
    n_actions = 7
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    obs = env.get_observations()
    rows = []

    joint_keys = [
        "joint_base",
        "joint_shoulder",
        "joint_elbow",
        "joint_wrist_pitch",
        "joint_wrist_roll",
        "joint_left_finger",
        "joint_right_finger",
    ]

    def capture(step_idx: int, phase: str):
        jp = raw_env.robot.data.joint_pos[0].detach().cpu().numpy()
        ee = raw_env._get_ee_pos_local()[0].detach().cpu().numpy()
        row = {
            "step": step_idx,
            "phase": phase,
            "ee_x": float(ee[0]),
            "ee_y": float(ee[1]),
            "ee_z": float(ee[2]),
        }
        for i, k in enumerate(joint_keys):
            row[k] = float(jp[i])
        rows.append(row)

    step_idx = 0

    for _ in range(30):
        actions = torch.zeros(n_envs, n_actions, device=device)
        obs, _, _, _ = env.step(actions)
        capture(step_idx, "warmup")
        step_idx += 1

    for joint_idx in range(5):
        for _ in range(45):
            actions = torch.zeros(n_envs, n_actions, device=device)
            actions[:, joint_idx] = 1.0
            obs, _, _, _ = env.step(actions)
            capture(step_idx, f"joint{joint_idx}_pos")
            step_idx += 1
        for _ in range(90):
            actions = torch.zeros(n_envs, n_actions, device=device)
            actions[:, joint_idx] = -1.0
            obs, _, _, _ = env.step(actions)
            capture(step_idx, f"joint{joint_idx}_neg")
            step_idx += 1
        for _ in range(45):
            actions = torch.zeros(n_envs, n_actions, device=device)
            actions[:, joint_idx] = 1.0
            obs, _, _, _ = env.step(actions)
            capture(step_idx, f"joint{joint_idx}_back")
            step_idx += 1

    out_csv = os.path.expanduser(args_cli.output_csv)
    if not os.path.isabs(out_csv):
        out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ranges = {}
    for k in joint_keys:
        vals = np.array([r[k] for r in rows], dtype=np.float32)
        ranges[k] = float(vals.max() - vals.min())

    ee_xyz = np.array([[r["ee_x"], r["ee_y"], r["ee_z"]] for r in rows], dtype=np.float32)
    ee_disp = float(np.linalg.norm(ee_xyz.max(axis=0) - ee_xyz.min(axis=0)))

    print("=== Articulation Motion Summary ===")
    print(f"CSV: {out_csv}")
    for k, v in ranges.items():
        print(f"{k}: range={v:.5f} rad")
    print(f"ee_disp_norm: {ee_disp:.5f} m")

    arm_joint_ranges = [ranges[k] for k in joint_keys[:5]]
    if min(arm_joint_ranges) < args_cli.min_joint_range:
        raise RuntimeError(
            f"Motion gate failed: min arm joint range {min(arm_joint_ranges):.5f} < {args_cli.min_joint_range:.5f}"
        )
    if ee_disp < args_cli.min_ee_disp:
        raise RuntimeError(
            f"EE displacement gate failed: {ee_disp:.5f} < {args_cli.min_ee_disp:.5f}"
        )

    print("PASS: articulation motion gates satisfied")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
