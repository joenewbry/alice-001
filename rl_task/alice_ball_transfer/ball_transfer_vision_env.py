"""Vision-based RL environment for Alice 001 ball transfer.

Extends the state-only environment with:
- Wrist camera rendering (224x224 RGB)
- ResNet-18 feature extraction (frozen, 512-dim)
- 4-frame stacking for temporal context
- Asymmetric observations: actor sees vision+proprio, critic sees privileged state
- Optional domain randomization (Stage 3)

The vision backbone runs on GPU inside the env. Features are stored in the
rollout buffer (526 floats per step vs 602K raw pixels), making PPO feasible.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera

from .ball_transfer_env import BallTransferEnv
from .ball_transfer_vision_env_cfg import BallTransferVisionEnvCfg
from .models.vision_backbone import VisionBackbone


class BallTransferVisionEnv(BallTransferEnv):
    """Ball transfer with wrist camera observations and asymmetric actor-critic."""

    cfg: BallTransferVisionEnvCfg

    def __init__(self, cfg: BallTransferVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Vision backbone (frozen ResNet-18)
        self._backbone = VisionBackbone(
            num_frames=cfg.num_frame_stack,
            feature_dim=cfg.vision_feature_dim,
        ).to(self.device)

        # Frame buffer: (num_envs, num_frames, 3, H, W)
        self._frame_buffer = torch.zeros(
            self.num_envs,
            cfg.num_frame_stack,
            3,
            cfg.camera_height,
            cfg.camera_width,
            device=self.device,
        )

        # Domain randomization state
        if cfg.enable_domain_rand:
            self._action_delay_buf = torch.zeros(
                self.num_envs, cfg.num_actions, device=self.device
            )
            self._action_delay_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._joint_offsets = torch.zeros(
                self.num_envs, self.robot.num_joints, device=self.device
            )
            self._camera_noise_std = torch.zeros(self.num_envs, device=self.device)
            self._camera_brightness = torch.ones(self.num_envs, device=self.device)

    # ── Scene setup ──────────────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()

        # Add wrist camera sensor
        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

    # ── Pre-physics (with optional action delay) ─────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.cfg.enable_domain_rand and hasattr(self, "_action_delay_steps"):
            # Apply randomized action delay
            delayed = self._apply_action_delay(actions)
            super()._pre_physics_step(delayed)

            # Add joint position offsets (simulates servo calibration error)
            self._robot_dof_targets += self._joint_offsets
            self._robot_dof_targets[:] = torch.clamp(
                self._robot_dof_targets, self._joint_lower, self._joint_upper
            )
        else:
            super()._pre_physics_step(actions)

    def _apply_action_delay(self, actions: torch.Tensor) -> torch.Tensor:
        """Buffer actions and return delayed version."""
        # For envs with delay=0, pass through immediately
        no_delay = self._action_delay_steps == 0
        delayed = self._action_delay_buf.clone()
        delayed[no_delay] = actions[no_delay]

        # Shift buffer
        self._action_delay_buf = actions.clone()
        return delayed.clamp(-1.0, 1.0)

    # ── Observations (vision + asymmetric) ───────────────────────────

    def _get_observations(self) -> dict:
        # ── Camera frame capture ──
        rgb = self.wrist_camera.data.output["rgb"]  # (N, H, W, 4) RGBA uint8

        # Convert to float [0,1], drop alpha, rearrange to (N, 3, H, W)
        rgb_float = rgb[:, :, :, :3].float() / 255.0
        rgb_float = rgb_float.permute(0, 3, 1, 2)

        # Apply domain randomization to camera image
        if self.cfg.enable_domain_rand:
            rgb_float = self._randomize_camera_image(rgb_float)

        # Push new frame into buffer, shift old ones
        self._frame_buffer = torch.roll(self._frame_buffer, shifts=-1, dims=1)
        self._frame_buffer[:, -1] = rgb_float

        # Stack frames: (N, num_frames*3, H, W)
        stacked = self._frame_buffer.reshape(
            self.num_envs, -1, self.cfg.camera_height, self.cfg.camera_width
        )

        # Extract visual features (frozen backbone, no grad)
        visual_features = self._backbone(stacked)  # (N, 512)

        # ── Proprioception (subset of full state) ──
        joint_pos = self.robot.data.joint_pos
        joint_pos_norm = 2.0 * (joint_pos - self._joint_lower) / self._joint_range - 1.0

        # Real joint velocities from PhysX simulation
        joint_vel = self.robot.data.joint_vel

        # ── Actor observation: visual features + proprioception ──
        actor_obs = torch.cat([
            visual_features,     # 512
            joint_pos_norm,      # 7
            joint_vel,           # 7
        ], dim=-1)  # Total: 526

        # ── Critic observation: full privileged state (same as Stage 1) ──
        ee_pos = self._get_ee_pos_local()
        ball_pos = self._get_ball_pos_local()
        ee_to_ball = ball_pos - ee_pos
        ball_to_target = self._target_pos_local - ball_pos

        gripper_opening = (
            torch.abs(joint_pos[:, self._left_finger_idx])
            + torch.abs(joint_pos[:, self._right_finger_idx])
        ).unsqueeze(-1) / 2.0

        ee_ball_dist = torch.norm(ee_to_ball, dim=-1, keepdim=True)
        has_contact = (ee_ball_dist < 0.02).float()

        critic_obs = torch.cat([
            joint_pos_norm,      # 7
            joint_vel,           # 7
            ee_pos,              # 3
            ball_pos,            # 3
            ee_to_ball,          # 3
            ball_to_target,      # 3
            gripper_opening,     # 1
            has_contact,         # 1
        ], dim=-1)  # Total: 28

        return {"policy": actor_obs, "critic": critic_obs}

    # ── Domain randomization helpers ─────────────────────────────────

    def _randomize_camera_image(self, images: torch.Tensor) -> torch.Tensor:
        """Apply visual domain randomization to camera images.

        Args:
            images: (N, 3, H, W) float tensor in [0, 1]

        Returns:
            Randomized images, same shape
        """
        # Gaussian noise (per-env sigma)
        noise_std = self._camera_noise_std.reshape(-1, 1, 1, 1)
        if noise_std.sum() > 0:
            noise = torch.randn_like(images) * noise_std
            images = images + noise

        # Brightness/contrast (per-env multiplier)
        brightness = self._camera_brightness.reshape(-1, 1, 1, 1)
        images = images * brightness

        return images.clamp(0.0, 1.0)

    def _randomize_on_reset(self, env_ids: torch.Tensor):
        """Apply domain randomization for newly reset environments."""
        if not self.cfg.enable_domain_rand:
            return

        n = len(env_ids)
        cfg = self.cfg

        # Action delay (integer sim steps)
        low, high = cfg.dr_action_delay_steps
        self._action_delay_steps[env_ids] = torch.randint(low, high + 1, (n,), device=self.device)

        # Joint position offsets
        self._joint_offsets[env_ids] = (
            torch.rand(n, self.robot.num_joints, device=self.device) * 2 - 1
        ) * cfg.dr_joint_offset

        # Camera noise
        lo, hi = cfg.dr_camera_noise_std
        self._camera_noise_std[env_ids] = torch.rand(n, device=self.device) * (hi - lo) + lo

        # Camera brightness
        lo, hi = cfg.dr_camera_brightness
        self._camera_brightness[env_ids] = torch.rand(n, device=self.device) * (hi - lo) + lo

    # ── Reset ────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Reset frame buffer for these envs
        self._frame_buffer[env_ids] = 0.0

        # Ball position randomization (geometric DR)
        super()._reset_idx(env_ids)

        if self.cfg.enable_domain_rand:
            # Override ball position with wider noise
            ball_state = self.ball.data.default_root_state[env_ids].clone()
            ball_state[:, :3] += self.scene.env_origins[env_ids]
            n = len(env_ids)
            ball_state[:, 0] += (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.dr_ball_start_noise_xy
            ball_state[:, 1] += (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.dr_ball_start_noise_xy
            ball_state[:, 7:] = 0.0
            self.ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
            self.ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)

            # Apply per-env randomization
            self._randomize_on_reset(env_ids)
