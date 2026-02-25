"""Asymmetric actor-critic PPO configuration for vision-based training.

Uses RSL-RL's native asymmetric support:
- Actor: visual features (512) + proprioception (14) = 526 dims
- Critic: privileged state (28 dims) — exact positions, contacts

The critic has access to ground-truth state during training but is discarded
at deployment. Only the actor (which sees camera + joints) runs on the robot.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AliceVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for vision-based ball transfer with asymmetric actor-critic."""

    num_steps_per_env = 24  # Fewer steps (camera rendering is expensive)
    max_iterations = 10000
    save_interval = 500
    experiment_name = "alice_ball_transfer_vision"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Actor: visual features (512) + proprioception (14) → smaller MLP
        actor_hidden_dims=[256, 128],
        # Critic: privileged state (28) → deeper MLP
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Slightly higher entropy for vision exploration
        num_learning_epochs=5,  # Fewer epochs (larger effective batch from vision)
        num_mini_batches=4,
        learning_rate=1e-4,  # Lower LR for stability with vision features
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
