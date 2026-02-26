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

from .ball_transfer_env_cfg import ALICE_001_CFG, BALL_CFG, PEDESTAL_CFG, TABLE_CFG, BallTransferEnvCfg


class BallTransferEnv(DirectRLEnv):
    """Alice 001 picks up a ball and places it at a target location."""

    cfg: BallTransferEnvCfg

    def __init__(self, cfg: BallTransferEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Precompute dt (matches Franka pattern: sim_dt * decimation)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Desired initial joint positions (explicit — init_state may not be applied)
        self._init_joint_pos = torch.tensor(
            [0.0, -0.3, -1.8, -0.5, 0.0, 0.0, 0.0],  # base, shoulder, elbow, wrist_p, wrist_r, L finger, R finger
            device=self.device, dtype=torch.float32
        )

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
        # Gripper joints — faster than before so exploration can close them
        self._robot_dof_speed_scales[self._left_finger_idx] = 0.5
        self._robot_dof_speed_scales[self._right_finger_idx] = 0.5

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

        # Kinematic grasp state: tracks whether ball is "attached" to EE
        self._ball_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Diagnostic: print initial EE and ball positions
        ee0 = self._get_ee_pos_local()[0].cpu().numpy()
        ball0 = self._get_ball_pos_local()[0].cpu().numpy()
        jp0 = self.robot.data.joint_pos[0].cpu().numpy()
        dist0 = ((ee0 - ball0) ** 2).sum() ** 0.5
        print(f"[INIT] EE=({ee0[0]:.4f},{ee0[1]:.4f},{ee0[2]:.4f}) Ball=({ball0[0]:.4f},{ball0[1]:.4f},{ball0[2]:.4f}) dist={dist0:.4f}")
        print(f"[INIT] Joints: {jp0.round(3)}")

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

        # Spawn pedestal (raised platform for ball)
        self.pedestal = RigidObject(PEDESTAL_CFG)
        self.scene.rigid_objects["pedestal"] = self.pedestal

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
        # Actuator-driven: set position targets for PD controllers (like Franka examples)
        self.robot.set_joint_position_target(self._robot_dof_targets)

        # ── Kinematic grasp attachment ──
        # Standard approach for small-object manipulation in Isaac Lab:
        # full contact physics is unreliable at 8mm ball scale.
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ball_pos_w = self.ball.data.root_pos_w
        ee_to_ball_dist = torch.norm(ball_pos_w - ee_pos_w, dim=-1)

        joint_pos = self.robot.data.joint_pos
        left_pos = joint_pos[:, self._left_finger_idx]
        right_pos = joint_pos[:, self._right_finger_idx]
        gripper_opening = (torch.abs(left_pos) + torch.abs(right_pos)) / 2.0

        # Grasp condition: EE within grasp_distance of ball AND gripper not wide open
        # Natural resting opening is ~0.146 rad, so threshold at 0.2 catches resting state
        grasp_dist = getattr(self.cfg, 'grasp_distance', 0.02)
        can_grasp = (ee_to_ball_dist < grasp_dist) & (gripper_opening < 0.2)
        # Release condition: gripper actively opened wide (>0.3 rad)
        releasing = gripper_opening > 0.3

        # Update grasp state
        self._ball_grasped = (self._ball_grasped | can_grasp) & ~releasing

        # Snap grasped balls to EE position (with small offset below gripper)
        if self._ball_grasped.any():
            grasped_ids = self._ball_grasped.nonzero(as_tuple=False).squeeze(-1)
            new_ball_pos = ee_pos_w[grasped_ids].clone()
            new_ball_pos[:, 2] -= 0.015  # Ball hangs slightly below gripper center

            # Build full pose (keep existing orientation)
            ball_quat = self.ball.data.root_quat_w[grasped_ids]
            ball_pose = torch.cat([new_ball_pos, ball_quat], dim=-1)
            ball_vel = torch.zeros(len(grasped_ids), 6, device=self.device)

            self.ball.write_root_pose_to_sim(ball_pose, grasped_ids)
            self.ball.write_root_velocity_to_sim(ball_vel, grasped_ids)

    # ── Observations ─────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # Joint positions (normalized to [-1, 1])
        joint_pos = self.robot.data.joint_pos
        joint_pos_norm = 2.0 * (joint_pos - self._joint_lower) / self._joint_range - 1.0

        # Real joint velocities from PhysX simulation
        joint_vel = self.robot.data.joint_vel

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

        # Grasp state from kinematic attachment (1.0 if ball is grasped)
        has_contact = self._ball_grasped.float().unsqueeze(-1)

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

        # Use real grasp state from kinematic attachment
        is_grasped = self._ball_grasped.float()

        # Gripper state
        left_pos = joint_pos[:, self._left_finger_idx]
        right_pos = joint_pos[:, self._right_finger_idx]
        gripper_open = ((torch.abs(left_pos) + torch.abs(right_pos)) > 0.5).float()

        # Ball near target (local frame)
        ball_at_target = ball_to_target_dist < self.cfg.target_radius
        ball_lifted = (ball_pos[:, 2] - self._source_pos_local[:, 2]) > self.cfg.lift_height

        # ── Simplified rewards: direct ball-to-target distance ──
        # Ball is always grasped from start, so the core task is:
        # move the arm to bring ball closer to target.

        # 1. Reach: keep EE near ball (gradient for maintaining grasp)
        reach_reward = torch.exp(-20.0 * ee_to_ball_dist)

        # 2. Grasp: small ongoing reward for maintaining kinematic grasp
        grasp_reward = is_grasped * 0.1

        # 3. Transport: MAIN REWARD — bring ball closer to target (continuous, strong gradient)
        # Two-scale exponential: broad gradient from far + sharp near target
        transport_reward = (
            0.5 * torch.exp(-20.0 * ball_to_target_dist)
            + 0.5 * torch.exp(-100.0 * ball_to_target_dist)
        ) * is_grasped

        # 4. Lift: bonus for any upward ball movement (continuous)
        ball_height_above_source = ball_pos[:, 2] - self._source_pos_local[:, 2]
        height_progress = torch.clamp(ball_height_above_source, 0.0, 0.05) / 0.05
        lift_reward = height_progress * is_grasped

        # 5. Drop: ball at target, gripper open
        drop_reward = ball_at_target.float() * gripper_open

        # 6. Penalties
        action_penalty = torch.sum(self._actions ** 2, dim=-1)
        velocity_penalty = torch.sum(self.robot.data.joint_vel ** 2, dim=-1)

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
        self.extras["log"]["pct_grasping"] = is_grasped.mean().item()
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

        # Terminate if ball falls off the pedestal
        # Pedestal top at z=0.15, ball starts at z=0.159
        # z < 0.12 means ball has clearly fallen off pedestal
        ball_pos = self._get_ball_pos_local()
        ball_fell = ball_pos[:, 2] < 0.12
        # Check XY bounds (pedestal is 20cm x 20cm centered at -0.091, -0.03)
        ball_off_xy = (
            (torch.abs(ball_pos[:, 0] + 0.091) > 0.15)
            | (torch.abs(ball_pos[:, 1] + 0.03) > 0.15)
        )
        terminated = terminated | ball_fell | ball_off_xy

        return terminated, time_out

    # ── Reset ────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Clear grasp state for reset envs
        self._ball_grasped[env_ids] = False

        num_reset = len(env_ids)

        # Reset robot joints — use explicit values (init_state config may not be applied)
        joint_pos = self._init_joint_pos.unsqueeze(0).expand(num_reset, -1).clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

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
