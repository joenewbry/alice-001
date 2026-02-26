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
        init_noise_std=1.0,  # High noise for 3-joint absolute exploration
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],     # Small network — task is simple (3D IK)
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Moderate entropy
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,  # Faster learning rate for simple task
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
