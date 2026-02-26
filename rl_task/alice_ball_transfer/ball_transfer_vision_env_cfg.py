"""Vision environment configuration for Alice 001 ball transfer.

Extends the state-only config with:
- Wrist camera sensor (224x224 RGB, mounted on gripper_base)
- Asymmetric observation spaces (actor: vision+proprio, critic: privileged state)
- Domain randomization parameters (used in Stage 3)
"""

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .ball_transfer_env_cfg import BallTransferEnvCfg


@configclass
class BallTransferVisionEnvCfg(BallTransferEnvCfg):
    """Configuration for vision-based ball transfer with asymmetric actor-critic."""

    # ── Disable Fabric (usdrt.hierarchy not available in Isaac Sim 4.5) ──
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=False,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=16,
            min_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
        ),
    )

    # ── Reduce envs (camera rendering is GPU-intensive) ──────────────
    # scene.num_envs overridden at runtime; default lower for vision
    # Typically 32-64 for vision on L4 GPU

    # ── Camera configuration ────────────────────────────────────────
    camera_width: int = 224
    camera_height: int = 224
    num_frame_stack: int = 4

    # Wrist camera mounted on gripper_base, looking forward/down
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_base/wrist_cam",
        update_period=0,  # Update every sim step
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,  # Wide FOV to approximate fisheye (~120 deg)
            focus_distance=0.4,
            horizontal_aperture=3.6,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Mounted 5cm above gripper base, looking forward and down
            pos=(0.0, 0.0, 0.05),
            rot=(0.653, -0.271, 0.653, -0.271),  # ~45 deg down from horizontal
            convention="ros",
        ),
    )

    # ── Observation spaces (asymmetric) ─────────────────────────────
    # Actor: ResNet-18 features (512) + joint pos (7) + joint vel (7) = 526
    # Critic: privileged state (same 28 as Stage 1)
    vision_feature_dim: int = 512
    num_proprio: int = 14  # 7 joint pos + 7 joint vel

    num_observations: int = 526  # Actor observation dim
    num_states: int = 28  # Critic (privileged) observation dim
    num_actions: int = 7

    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(526,))
    state_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(28,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))

    # ── Domain randomization (Stage 3) ──────────────────────────────
    # Set enable_domain_rand=True to activate. Ranges ramp linearly
    # from conservative to full over dr_ramp_iterations.
    enable_domain_rand: bool = False
    dr_ramp_iterations: int = 2000  # Iterations to ramp from min to max range

    # Physics randomization ranges [min, max]
    dr_table_friction: tuple = (0.3, 1.2)
    dr_ball_friction: tuple = (0.3, 1.2)
    dr_ball_mass: tuple = (0.003, 0.010)  # 3g to 10g
    dr_servo_stiffness: tuple = (30.0, 50.0)
    dr_servo_damping: tuple = (3.0, 6.0)
    dr_action_delay_steps: tuple = (0, 3)  # Sim steps of delay
    dr_joint_offset: float = 0.02  # +/- radians

    # Visual randomization
    dr_light_intensity: tuple = (500.0, 4000.0)
    dr_ball_color_range: float = 1.0  # Full RGB random
    dr_camera_noise_std: tuple = (0.0, 0.02)
    dr_camera_brightness: tuple = (0.8, 1.2)  # Multiplier

    # Geometric randomization
    dr_ball_start_noise_xy: float = 0.03  # +/- 3cm
    dr_target_noise_xy: float = 0.03
    dr_camera_rot_noise: float = 0.035  # ~2 degrees in radians
    dr_camera_pos_noise: float = 0.003  # 3mm
