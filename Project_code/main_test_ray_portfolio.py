import gymnasium
import time

from test_trade_env_ray_portfolio import TradingEnvironment

import json

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch

import torch


from gymnasium.envs.registration import register

# initialize torch and neural networks
torch, nn = try_import_torch()

if __name__ == "__main__":

    input_length = 100  # Define the length of the input window

    # Register the environment in gymnasium
    register(
        id="trade_env_ray_portfolio",
        entry_point="test_trade_env_ray_portfolio:TradingEnvironment",
    )

    # Define the environment creator function
    def env_creator(env_config):
        return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trade_env_ray_portfolio", env_creator)

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=1 * (1024**3))

    entropy_coeff_schedule = [
        [0, 0.01],  # start very exploratory
        [1e6, 0.005],  # after ~1 M env steps
        [3e6, 0.001],  # match LR/KL decay
        [5e6, 0.0005],  # late fine-tune
    ]

    # Configuration using PPOConfig
    config = PPOConfig()
    config.environment(
        env="trade_env_ray_portfolio",
        env_config={
            # "data": train_X_ds,
            "input_length": 100,
            "market_fee": 0.0005,
            "limit_fee": 0.0002,
            "slippage_mean": 0.000001,
            "slippage_std": 0.00005,
            "initial_balance": 1000,
            "total_episodes": 1,
            "max_episodes": False,
            "full_episode": True,
            "episode_length": 120,
            "max_risk": 0.02,
            "min_risk": 0.001,
            "min_profit": 0,
            "seed": 42,
            "limit_bounds": False,
        },
    )
    config.framework("torch")
    config.resources(num_gpus=1, num_cpus_per_worker=1)
    config.rollouts(
        num_rollout_workers=1,
        rollout_fragment_length=120,  # 1 day of data
        batch_mode="complete_episodes",
    )
    config.training(
        gamma=0.97,
        lr=1e-4,
        lr_schedule=[[0, 1e-4], [3e6, 2e-5]],
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
        kl_coeff=0.5,
        kl_target=0.015,
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

    test_env = gymnasium.make(
        "trade_env_ray_portfolio", mode="test", input_length=100, seed=42
    )
    observation_space = test_env.observation_space
    action_space = test_env.action_space

    # Define the paths
    base_dir = r"C:\Users\marko\ray_results\Full_episode_LowLambda_stage1_conditioned\PPO_trade_env_ray_portfolio_c5092_00000_0_2025-06-08_09-39-55\checkpoint_000018"

    trainer = config.build()

    # Restore the checkpoint
    trainer.restore(base_dir)

    # Verify the checkpoint has been loaded correctly
    print(f"Restored from checkpoint: {base_dir}")

    state, _ = test_env.reset()
    with open(
        f"C:/Users/marko/Marko_documents/Etudes/Master_2ème/2ème_semestre/Thesis/Results_final/PPO/Stage_1/Full_episode_Lambda_01_stage1/saved_states/env_state_{57}.json"
    ) as f:
        saved = json.load(f)
    test_env.set_state(saved)
    # test_env.check_alignment(asset_idx=5, ohlcv_field=3, max_points=1000)
    terminated = False
    cumulative_reward = 0

    # Initial LSTM state (empty state)
    lstm_state = trainer.get_policy().get_initial_state()
    # Print the initial LSTM state
    print(f"INITIAL LSTM STATE: {lstm_state}")
    max_steps = 50  # Set the maximum number of steps to print
    step_count = 0

    # Start the timer
    start_time = time.time()
    while not terminated:

        if state.shape != (input_length, state.shape[1]):
            raise ValueError(f"Unexpected state shape: {state.shape}")

        # Convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        action, lstm_state, _ = trainer.compute_single_action(
            state_tensor,
            state=lstm_state,
            explore=False,
        )

        state, reward, terminated, truncated, info = test_env.step(action)
        cumulative_reward += reward

        step_count += 1

    print(f"Total reward on test data: {cumulative_reward}")

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time in seconds
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    # test_env.plot_alignment()

    # After the testing simulation loop in your main script
    test_env.render()
