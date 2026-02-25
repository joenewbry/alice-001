"""ArticulationCfg for Alice 001 (Hiwonder xArm 1S) in Isaac Lab.

Usage in an Isaac Lab environment:
    from alice_001_cfg import ALICE_001_CFG
    robot = Articulation(cfg=ALICE_001_CFG)

Prerequisites:
    - Convert URDF to USD first using Isaac Lab's convert_urdf.py
    - Place the generated USD at the path referenced below
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

ALICE_001_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="{ISAAC_NUCLEUS_DIR}/Alice-001/alice_001.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "base_joint": 0.0,
            "shoulder_joint": 0.0,
            "elbow_joint": -0.5,       # Slightly bent
            "wrist_pitch_joint": 0.3,   # Slightly tilted forward
            "wrist_roll_joint": 0.0,
            "left_finger_joint": 0.15,  # Slightly open
            "right_finger_joint": -0.15,
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
            effort_limit=2.5,       # LX-15D stall torque ~2.5 Nm
            velocity_limit=5.24,    # LX-15D ~50 RPM
            stiffness=40.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_finger_joint",
                "right_finger_joint",
            ],
            effort_limit=1.0,
            velocity_limit=5.24,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)
"""
Notes on tuning:
- stiffness/damping: Start with arm stiffness=40, damping=4. Increase if arm sags
  under gravity. Decrease if motion is too stiff/oscillatory.
- effort_limit: 2.5 Nm matches LX-15D servos at stall. Gripper uses 1.0 Nm
  since finger servos don't need as much torque.
- For position control: stiffness=40, damping=4 gives ~10:1 ratio (critically damped)
- For effort/torque control: set stiffness=0, damping=0, joint_target_type="none"
  in the URDF converter
"""
