"""RSL-RL PPO configuration for Alice 001 ball transfer task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AliceBallTransferPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24   # Match shorter episodes (4s = 240 steps)
    max_iterations = 5000
    save_interval = 250
    experiment_name = "alice_ball_transfer"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # High exploration — only 3 effective joints, need to find transport
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],   # Smaller network for simpler 3-joint task
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,   # Higher entropy for exploration in 3-joint space
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
