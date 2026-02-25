"""Alice 001 ball transfer RL task for Isaac Lab."""

import gymnasium as gym

from .ball_transfer_env_cfg import BallTransferEnvCfg
from .ball_transfer_vision_env_cfg import BallTransferVisionEnvCfg

# Stage 1: State-only RL
gym.register(
    id="Alice-Ball-Transfer-Direct-v0",
    entry_point="alice_ball_transfer.ball_transfer_env:BallTransferEnv",
    kwargs={
        "env_cfg_entry_point": BallTransferEnvCfg,
        "rsl_rl_cfg_entry_point": (
            "alice_ball_transfer.agents.rsl_rl_ppo_cfg:AliceBallTransferPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

# Stage 2/3: Vision RL with asymmetric actor-critic
gym.register(
    id="Alice-Ball-Transfer-Vision-v0",
    entry_point="alice_ball_transfer.ball_transfer_vision_env:BallTransferVisionEnv",
    kwargs={
        "env_cfg_entry_point": BallTransferVisionEnvCfg,
        "rsl_rl_cfg_entry_point": (
            "alice_ball_transfer.models.asymmetric_ac:AliceVisionPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)
