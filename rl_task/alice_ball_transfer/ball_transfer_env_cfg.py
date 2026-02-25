"""Environment configuration for Alice 001 ball transfer task."""

from pathlib import Path

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# ── Robot configuration ──────────────────────────────────────────────────

USD_PATH = str((Path(__file__).resolve().parents[2] / "usd" / "alice_001.usd"))

ALICE_001_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # On top of the table
        joint_pos={
            "base_joint": 0.0,
            "shoulder_joint": 0.0,
            "elbow_joint": -1.047,
            "wrist_pitch_joint": 0.0,
            "wrist_roll_joint": 0.0,
            "left_finger_joint": 0.26,
            "right_finger_joint": -0.26,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "base_joint",
                "shoulder_joint",
                "elbow_joint",
                "wrist_pitch_joint",
                "wrist_roll_joint",
            ],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger_joint", "right_finger_joint"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)

# ── Ball configuration ───────────────────────────────────────────────────

BALL_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.008,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.1, 0.1),
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.080, 0.03, 0.333),  # 3cm Y offset from stable EE
    ),
)

# ── Table configuration ──────────────────────────────────────────────────

TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.CuboidCfg(
        size=(0.50, 0.50, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.6, 0.5, 0.4),
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.025),
    ),
)


# ── Environment configuration ────────────────────────────────────────────

@configclass
class BallTransferEnvCfg(DirectRLEnvCfg):
    """Configuration for the Alice 001 ball transfer environment."""

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, 0.0),  # Gravity disabled: PhysX drives can't hold this lightweight arm
        use_fabric=False,  # Required for camera sensors (usdrt.hierarchy bug in Isaac Sim 4.5)
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=16,
            min_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
        ),
    )

    # ── Overhead camera for visualization (not a policy input) ────
    # Workspace is at z≈0.33m (arm operating height), centered around (-0.06, 0.0)
    # Camera at z=0.55 gives ~22cm above action, 60deg FOV covers ~25cm — enough to
    # see the 8mm ball, gripper, and the ~2cm of arm motion clearly.
    overhead_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/overhead_cam",
        update_period=0,
        height=480,
        width=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=0.22,
            horizontal_aperture=11.5,  # ~60 deg FOV — tight on workspace
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.05, 0.0, 0.55),    # 55cm up, centered over workspace
            rot=(0.0, 1.0, 0.0, 0.0),  # 180 deg around X — look straight down
            convention="ros",
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=1.0,
    )

    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(28,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
    num_observations = 28
    num_actions = 7
    num_states = 0

    episode_length_s = 8.0
    decimation = 2

    action_scale = 0.5

    # Reward scales
    reach_reward_scale = 5.0
    grasp_reward_scale = 10.0
    lift_reward_scale = 15.0
    transport_reward_scale = 10.0
    drop_reward_scale = 50.0
    action_penalty_scale = 0.005
    velocity_penalty_scale = 0.0005

    # Task positions (ball 3cm Y from stable EE, same Z/X)
    source_pos = (-0.080, 0.03, 0.333)
    target_pos = (-0.080, -0.03, 0.333)
    target_radius = 0.02
    lift_height = 0.03

    # ── Domain randomization (off by default for Stage 1) ───────────
    enable_domain_rand: bool = False
    dr_ramp_iterations: int = 2000

    # Physics
    dr_ball_mass: tuple = (0.003, 0.010)
    dr_action_delay_steps: tuple = (0, 3)
    dr_joint_offset: float = 0.02

    # Geometric
    dr_ball_start_noise_xy: float = 0.03
    dr_target_noise_xy: float = 0.03
