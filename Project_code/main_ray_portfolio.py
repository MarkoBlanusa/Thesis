import json
import os

from trade_env_ray_portfolio import TradingEnvironment

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.tune.search import basic_variant
from ray.tune.stopper import TrialPlateauStopper

import logging

# Set up logger
logger = logging.getLogger(__name__)


from gymnasium.envs.registration import register

# initialize torch and neural networks
torch, nn = try_import_torch()


if __name__ == "__main__":

    # Register the environment in gymnasium
    register(
        id="trade_env_ray_portfolio",
        entry_point="trade_env_ray_portfolio:TradingEnvironment",
    )

    # Define the environment creator function
    def env_creator(env_config):
        return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trade_env_ray_portfolio", env_creator)

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=10 * (1024**3))

    # Define paths to datasets
    train_data_path = os.path.abspath(
        "data/train_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17"
    )
    val_data_path = os.path.abspath(
        "data/val_portfolio_data_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17"
    )
    test_data_path = os.path.abspath(
        "data/test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17"
    )

    train_raw_path = os.path.abspath("data/train_raw_ohlcv_100_1d_NewVal2.npy")
    val_raw_path = os.path.abspath("data/val_raw_ohlcv_100_1d_NewVal2.npy")
    test_raw_path = os.path.abspath("data/test_raw_ohlcv_100_1d_NewVal2.npy")

    # Verify that the files exist
    assert os.path.exists(
        train_data_path
    ), f"Training dataset not found at {train_data_path}"
    assert os.path.exists(
        val_data_path
    ), f"Validation dataset not found at {val_data_path}"

    # Print paths for debugging
    print(f"Training data path: {train_data_path}")
    print(f"Validation data path: {val_data_path}")

    entropy_coeff_schedule = [
        [0, 0.01],  # start very exploratory
        [1e6, 0.005],  # after ~1 M env steps
        [3e6, 0.002],  # match LR/KL decay
    ]

    # with open(f"env_state_{274}.json") as f:
    #     states_ = json.load(f)

    # Configuration using PPOConfig
    config = PPOConfig()
    config.environment(
        env="trade_env_ray_portfolio",
        env_config={
            "data_path": train_data_path,
            "raw_data_path": train_raw_path,
            "states": None,
            "mode": "train",
            "leverage": 5,
            "input_length": 100,
            "market_fee": 0.0005,
            "limit_fee": 0.0002,
            "slippage_mean": 0.000001,
            "slippage_std": 0.00005,
            "initial_balance": 1000,
            "total_episodes": 1,
            "episode_length": 180,
            "max_risk": 0.02,
            "min_risk": 0.001,
            "min_profit": 0,
            "limit_bounds": False,
            "margin_mode": "cross",
            "predict_leverage": False,
            "ppo_mode": True,
            "full_invest": False,
        },
    )
    config.framework("torch")
    config.resources(num_gpus=1, num_cpus_per_worker=1)
    config.rollouts(
        num_rollout_workers=14,
        rollout_fragment_length=120,  # 1 day of data
        batch_mode="complete_episodes",
    )
    config.training(
        gamma=0.97,
        lr=1e-4,
        lr_schedule=[[0, 1e-4], [5e6, 5e-5], [10e6, 2e-5], [15e6, 1e-5]],
        train_batch_size=1680,
        sgd_minibatch_size=280,
        num_sgd_iter=10,
        shuffle_sequences=False,
        grad_clip=0.5,
        lambda_=0.9,
        entropy_coeff=0.01,
        entropy_coeff_schedule=entropy_coeff_schedule,
        clip_param=0.1,
        vf_clip_param=0.5,
        vf_loss_coeff=0.3,
        kl_coeff=0.5,
        kl_target=0.015,
        use_kl_loss=True,
    )
    # Access the model configuration directly via the `.model` attribute
    config.model["use_lstm"] = True
    config.model["lstm_cell_size"] = 128
    config.model["fcnet_hiddens"] = [128, 128]
    config.model["fcnet_activation"] = "relu"
    config.model["post_fcnet_activation"] = "linear"
    config.model["lstm_use_prev_action_reward"] = True
    config.model["max_seq_len"] = 100
    config.model["_disable_action_flattening"] = True

    # --- new evaluation block ---
    config.evaluation(
        evaluation_interval=10,  # validate every 10 iteration
        evaluation_duration=5,
        evaluation_config={
            "env_config": {
                "data_path": val_data_path,  # validation split
                "raw_data_path": val_raw_path,
                "mode": "val",
                "limit_bounds": False,
            },
            "explore": False,  # greedy policy during eval
            "num_gpus": 0,  # or 1 if your model is large
        },
        evaluation_sample_timeout_s=180,  # fail fast if env hangs
        # evaluation_force_reset_envs_before_iteration=True,
    )

    # checkpoint_path = r"C:\Users\marko\ray_results\Full_episode_LowLambda_stage3\PPO_trade_env_ray_portfolio_faf88_00000_0_2025-06-03_22-11-48\checkpoint_000273"

    results = tune.run(
        "PPO",
        config=config,
        metric="episode_reward_mean",
        mode="max",
        # stop=stopper,
        # num_samples=10,  # Number of different sets of hyperparameters to try
        search_alg=basic_variant.BasicVariantGenerator(),  # Simple random search
        # scheduler=scheduler,
        verbose=1,
        checkpoint_freq=3,  # Save a checkpoint every 10 training iterations
        checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end of training
        # local_dir=r"C:\Users\marko\ray_results\Full_episode_LowLambda_stage3",
        # restore=checkpoint_path,
    )

    # Access the best trial's results and checkpoints
    best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print(
        "Best trial final reward: {}".format(
            best_trial.last_result["episode_reward_mean"]
        )
    )
