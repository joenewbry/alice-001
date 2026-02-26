"""Direct RL environment for Alice 001 ball transfer task.

The robot must pick up a ball from a source position and place it at a target position.
Follows the Isaac Lab DirectRLEnv pattern (Franka Cabinet reference).
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera

from .ball_transfer_env_cfg import ALICE_001_CFG, BALL_CFG, TABLE_CFG, BallTransferEnvCfg


class BallTransferEnv(DirectRLEnv):
    """Alice 001 picks up a ball and places it at a target location."""

    cfg: BallTransferEnvCfg

    def __init__(self, cfg: BallTransferEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Precompute dt (matches Franka pattern: sim_dt * decimation)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Cache joint limits for normalization and clamping (1D tensors)
        self._joint_lower = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self._joint_upper = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self._joint_range = self._joint_upper - self._joint_lower

        # Find body index for end-effector (gripper_base)
        self._ee_body_idx = self.robot.find_bodies("gripper_base")[0][0]

        # Find finger joint indices
        self._left_finger_idx = self.robot.find_joints("left_finger_joint")[0][0]
        self._right_finger_idx = self.robot.find_joints("right_finger_joint")[0][0]

        # Speed scales for delta target computation (1D, matches Franka pattern)
        self._robot_dof_speed_scales = torch.ones_like(self._joint_lower)
        # Gripper joints move slower
        self._robot_dof_speed_scales[self._left_finger_idx] = 0.1
        self._robot_dof_speed_scales[self._right_finger_idx] = 0.1

        # Persistent joint position targets (updated incrementally each step)
        self._robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Target and source positions in LOCAL frame (per-env, same value)
        self._target_pos_local = torch.tensor(
            self.cfg.target_pos, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)

        self._source_pos_local = torch.tensor(
            self.cfg.source_pos, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)

        # Domain randomization state
        if self.cfg.enable_domain_rand:
            self._action_delay_buf = torch.zeros(
                self.num_envs, self.cfg.num_actions, device=self.device
            )
            self._action_delay_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._joint_offsets = torch.zeros(
                self.num_envs, self.robot.num_joints, device=self.device
            )

        # Logging buffers for per-phase reward tracking
        self._reward_components = {
            "reach": torch.zeros(self.num_envs, device=self.device),
            "grasp": torch.zeros(self.num_envs, device=self.device),
            "lift": torch.zeros(self.num_envs, device=self.device),
            "transport": torch.zeros(self.num_envs, device=self.device),
            "drop": torch.zeros(self.num_envs, device=self.device),
            "penalty": torch.zeros(self.num_envs, device=self.device),
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_ee_pos_local(self) -> torch.Tensor:
        """End-effector position in env-local frame."""
        return self.robot.data.body_pos_w[:, self._ee_body_idx, :] - self.scene.env_origins

    def _get_ball_pos_local(self) -> torch.Tensor:
        """Ball position in env-local frame."""
        return self.ball.data.root_pos_w - self.scene.env_origins

    # ── Scene setup ──────────────────────────────────────────────────

    def _setup_scene(self):
        # Spawn robot
        self.robot = Articulation(ALICE_001_CFG)
        self.scene.articulations["robot"] = self.robot

        # Spawn table
        self.table = RigidObject(TABLE_CFG)
        self.scene.rigid_objects["table"] = self.table

        # Spawn ball
        self.ball = RigidObject(BALL_CFG)
        self.scene.rigid_objects["ball"] = self.ball

        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Clone environments (critical for multi-env + env_origins)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        # Overhead camera (visualization only, not a policy input)
        if hasattr(self.cfg, 'overhead_camera') and self.cfg.overhead_camera is not None:
            self.overhead_camera = Camera(self.cfg.overhead_camera)
            self.scene.sensors["overhead_camera"] = self.overhead_camera
        if hasattr(self.cfg, "side_camera") and self.cfg.side_camera is not None:
            self.side_camera = Camera(self.cfg.side_camera)
            self.scene.sensors["side_camera"] = self.side_camera

    # ── Pre-physics (apply actions) ──────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Apply action delay if domain randomization is active
        effective_actions = self._actions
        if self.cfg.enable_domain_rand and hasattr(self, "_action_delay_steps"):
            no_delay = self._action_delay_steps == 0
            delayed = self._action_delay_buf.clone()
            delayed[no_delay] = self._actions[no_delay]
            self._action_delay_buf = self._actions.clone()
            effective_actions = delayed

        # Delta target formulation (Franka Cabinet pattern):
        # targets += speed_scales * dt * actions * action_scale
        # This keeps PD errors small and prevents arm instability
        targets = self._robot_dof_targets + (
            self._robot_dof_speed_scales * self.dt * effective_actions * self.cfg.action_scale
        )

        # Apply joint offset noise if domain randomization is active
        if self.cfg.enable_domain_rand and hasattr(self, "_joint_offsets"):
            targets = targets + self._joint_offsets

        self._robot_dof_targets[:] = torch.clamp(targets, self._joint_lower, self._joint_upper)

    def _apply_action(self):
        # Kinematic mode: directly set joint positions (bypasses PhysX drives)
        # PhysX drives don't work reliably for this lightweight articulation
        joint_vel = torch.zeros_like(self._robot_dof_targets)
        self.robot.write_joint_state_to_sim(self._robot_dof_targets, joint_vel)

    # ── Observations ─────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # Joint positions (normalized to [-1, 1])
        joint_pos = self.robot.data.joint_pos
        joint_pos_norm = 2.0 * (joint_pos - self._joint_lower) / self._joint_range - 1.0

        # Implied joint velocities from last action (kinematic mode has zero sim velocities)
        joint_vel = self._actions * self.cfg.action_scale * 0.1 if hasattr(self, '_actions') else torch.zeros_like(joint_pos)

        # Positions in env-local frame
        ee_pos = self._get_ee_pos_local()
        ball_pos = self._get_ball_pos_local()

        # Relative vectors (local frame, so these are correct)
        ee_to_ball = ball_pos - ee_pos
        ball_to_target = self._target_pos_local - ball_pos

        # Gripper opening: mean absolute finger joint position
        gripper_opening = (
            torch.abs(joint_pos[:, self._left_finger_idx])
            + torch.abs(joint_pos[:, self._right_finger_idx])
        ).unsqueeze(-1) / 2.0

        # Proximity-based grasp detection (EE close to ball + gripper closing)
        ee_ball_dist = torch.norm(ball_pos - ee_pos, dim=-1, keepdim=True)
        has_contact = (ee_ball_dist < 0.02).float()  # Within 2cm

        obs = torch.cat([
            joint_pos_norm,      # 7
            joint_vel,           # 7
            ee_pos,              # 3
            ball_pos,            # 3
            ee_to_ball,          # 3
            ball_to_target,      # 3
            gripper_opening,     # 1
            has_contact,         # 1
        ], dim=-1)  # Total: 28

        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        # All positions in env-local frame
        ee_pos = self._get_ee_pos_local()
        ball_pos = self._get_ball_pos_local()
        joint_pos = self.robot.data.joint_pos

        # Distances (local frame)
        ee_to_ball_dist = torch.norm(ball_pos - ee_pos, dim=-1)
        ball_to_target_dist = torch.norm(ball_pos - self._target_pos_local, dim=-1)

        # Proximity-based grasp detection
        has_contact = (ee_to_ball_dist < 0.02).float()  # Within 2cm

        # Gripper closing check (fingers moving toward closed position)
        left_pos = joint_pos[:, self._left_finger_idx]
        right_pos = joint_pos[:, self._right_finger_idx]
        gripper_closed = ((torch.abs(left_pos) + torch.abs(right_pos)) < 0.2).float()

        # Ball height relative to source (local Z)
        ball_lifted = (
            ball_pos[:, 2] - self._source_pos_local[:, 2]
        ) > self.cfg.lift_height

        # Ball near target laterally and at table height (local frame)
        ball_at_target = ball_to_target_dist < self.cfg.target_radius
        ball_on_table = ball_pos[:, 2] < (self._source_pos_local[:, 2] + 0.01)
        gripper_open = ((torch.abs(left_pos) + torch.abs(right_pos)) > 0.3).float()

        # ── Phase rewards ──

        # 1. Reach: two-scale exp for gradient at all distances
        reach_reward = 0.5 * torch.exp(-10.0 * ee_to_ball_dist) + 0.5 * torch.exp(-100.0 * ee_to_ball_dist)

        # 2. Grasp: contact + gripper closing + near ball
        near_ball = (ee_to_ball_dist < 0.025).float()
        grasp_reward = has_contact * gripper_closed * near_ball

        # 3. Lift: ball above source height while grasping
        lift_reward = ball_lifted.float() * has_contact

        # 4. Transport: move ball toward target while lifted and grasping
        transport_reward = (
            (1.0 - torch.tanh(ball_to_target_dist / 0.05))
            * ball_lifted.float()
            * has_contact
        )

        # 5. Drop: ball at target, on table, gripper open
        drop_reward = ball_at_target.float() * ball_on_table.float() * gripper_open

        # 6. Penalties (velocity penalty uses actions since kinematic joints have zero sim vel)
        action_penalty = torch.sum(self._actions ** 2, dim=-1)
        velocity_penalty = action_penalty  # In kinematic mode, action magnitude ~= velocity

        # ── Total ──
        total = (
            self.cfg.reach_reward_scale * reach_reward
            + self.cfg.grasp_reward_scale * grasp_reward
            + self.cfg.lift_reward_scale * lift_reward
            + self.cfg.transport_reward_scale * transport_reward
            + self.cfg.drop_reward_scale * drop_reward
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.velocity_penalty_scale * velocity_penalty
        )

        # Store for logging
        self._reward_components["reach"] = reach_reward
        self._reward_components["grasp"] = grasp_reward
        self._reward_components["lift"] = lift_reward
        self._reward_components["transport"] = transport_reward
        self._reward_components["drop"] = drop_reward
        self._reward_components["penalty"] = -(
            self.cfg.action_penalty_scale * action_penalty
            + self.cfg.velocity_penalty_scale * velocity_penalty
        )

        # Log to extras for TensorBoard
        self.extras["log"] = {
            f"reward/{k}": v.mean().item() for k, v in self._reward_components.items()
        }
        self.extras["log"]["pct_grasping"] = has_contact.mean().item()
        self.extras["log"]["pct_lifted"] = ball_lifted.float().mean().item()
        self.extras["log"]["pct_at_target"] = ball_at_target.float().mean().item()
        self.extras["log"]["mean_ee_to_ball_dist"] = ee_to_ball_dist.mean().item()
        self.extras["log"]["mean_ball_to_target_dist"] = ball_to_target_dist.mean().item()

        return total

    # ── Dones ────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time-out
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros_like(time_out)

        # Terminate if ball falls off the table (local Z < -0.05)
        ball_pos = self._get_ball_pos_local()
        ball_fell = ball_pos[:, 2] < -0.05
        terminated = terminated | ball_fell

        return terminated, time_out

    # ── Reset ────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # Reset robot joints (following Franka Cabinet pattern)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset persistent DOF targets to init state
        self._robot_dof_targets[env_ids] = joint_pos

        # Reset ball to source position with noise
        ball_state = self.ball.data.default_root_state[env_ids].clone()
        ball_state[:, :3] += self.scene.env_origins[env_ids]

        if self.cfg.enable_domain_rand:
            # Wider noise for domain randomization
            noise_range = self.cfg.dr_ball_start_noise_xy
            ball_state[:, 0] += (torch.rand(num_reset, device=self.device) * 2 - 1) * noise_range
            ball_state[:, 1] += (torch.rand(num_reset, device=self.device) * 2 - 1) * noise_range

            # Randomize per-env action delays and joint offsets
            low, high = self.cfg.dr_action_delay_steps
            self._action_delay_steps[env_ids] = torch.randint(low, high + 1, (num_reset,), device=self.device)
            self._joint_offsets[env_ids] = (
                torch.rand(num_reset, self.robot.num_joints, device=self.device) * 2 - 1
            ) * self.cfg.dr_joint_offset
        else:
            # Small position noise (2mm)
            ball_state[:, 0] += torch.randn(num_reset, device=self.device) * 0.002
            ball_state[:, 1] += torch.randn(num_reset, device=self.device) * 0.002

        # Zero velocity
        ball_state[:, 7:] = 0.0
        self.ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)
