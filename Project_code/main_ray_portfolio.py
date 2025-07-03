import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
from typing import Dict
import gc
import psutil
import cProfile
import json
import pstats
import os

from binance import BinanceClient
from database import Hdf5client
import data_collector
import utils
from trade_env_ray_portfolio import TradingEnvironment

import joblib  # to persist scalers to disk
import re

from tqdm import tqdm
import gymnasium as gym

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from ray.rllib.utils.framework import try_import_torch
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search import basic_variant
from ray.rllib.algorithms.ppo import PPO
from ray.data import from_pandas, Dataset, from_numpy
from ray.data.preprocessors import Concatenator
from ray.tune import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.rllib.models.torch.misc import normc_initializer, SlimFC

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog

from torch.utils.data import Dataset, DataLoader
import torch
import logging

# Set up logger
logger = logging.getLogger(__name__)


from gymnasium.envs.registration import register

# initialize torch and neural networks
torch, nn = try_import_torch()


class SAC_LSTM_Model(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Extract relevant model config parameters
        self.lstm_cell_size = model_config.get("lstm_cell_size", 64)
        self.fcnet_hiddens = model_config.get("fcnet_hiddens", [32, 32])
        self.fcnet_activation = model_config.get("fcnet_activation", "relu")
        self.max_seq_len = 100  # Hardcoded to match input_length

        # Flatten action space to determine num_outputs
        self.action_dim = (
            int(np.prod(action_space.shape))
            if hasattr(action_space, "shape")
            else action_space.n
        )
        if num_outputs is None or (
            self.action_dim != num_outputs and hasattr(action_space, "shape")
        ):
            logger.warning(
                f"num_outputs ({num_outputs}) does not match action_space shape ({action_space.shape}). Using {self.action_dim}."
            )
            num_outputs = self.action_dim

        # Shared encoder (LSTM-based)
        self.lstm = nn.LSTM(
            input_size=obs_space.shape[0],
            hidden_size=self.lstm_cell_size,
            num_layers=1,
            batch_first=True,
        )

        # Policy network (after LSTM)
        self.policy_layers = nn.ModuleList()
        prev_layer_size = self.lstm_cell_size
        for hidden_size in self.fcnet_hiddens:
            self.policy_layers.append(
                SlimFC(
                    prev_layer_size,
                    hidden_size,
                    activation_fn=nn.ReLU,
                    initializer=normc_initializer(0.01),
                )
            )
            prev_layer_size = hidden_size
        self.policy_out = SlimFC(
            prev_layer_size,
            num_outputs,
            activation_fn=None,
            initializer=normc_initializer(0.01),
        )

        # Q-value network (after LSTM)
        self.q_layers = nn.ModuleList()
        prev_layer_size = self.lstm_cell_size
        for hidden_size in self.fcnet_hiddens:
            self.q_layers.append(
                SlimFC(
                    prev_layer_size,
                    hidden_size,
                    activation_fn=nn.ReLU,
                    initializer=normc_initializer(0.01),
                )
            )
            prev_layer_size = hidden_size
        self.q_out = SlimFC(
            prev_layer_size, 1, activation_fn=None, initializer=normc_initializer(0.01)
        )

        # Initialize LSTM state
        self._hidden_state = None
        self._cell_state = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        batch_size = obs.size(0)

        # Reshape for LSTM (batch_size, seq_len, num_features)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
        if obs.shape[1] != self.max_seq_len:
            current_seq_len = obs.shape[1]
            if current_seq_len < self.max_seq_len:
                padding = torch.zeros(
                    batch_size,
                    self.max_seq_len - current_seq_len,
                    obs.shape[2],
                    device=obs.device,
                )
                obs = torch.cat([obs, padding], dim=1)
            else:
                obs = obs[:, : self.max_seq_len, :]

        # Initialize LSTM state if not provided
        if not state:
            self._hidden_state = torch.zeros(
                1, batch_size, self.lstm_cell_size, device=obs.device
            )
            self._cell_state = torch.zeros(
                1, batch_size, self.lstm_cell_size, device=obs.device
            )
        else:
            self._hidden_state, self._cell_state = state

        # Pass through LSTM
        lstm_out, (self._hidden_state, self._cell_state) = self.lstm(
            obs, (self._hidden_state, self._cell_state)
        )

        # Take the last timestep’s output
        lstm_out = lstm_out[:, -1, :]

        # Policy branch
        policy_hidden = lstm_out
        for layer in self.policy_layers:
            policy_hidden = layer(policy_hidden)
        policy_out = self.policy_out(policy_hidden)

        # Q-value branch (for value_function)
        q_hidden = lstm_out
        for layer in self.q_layers:
            q_hidden = layer(q_hidden)
        self._q_value_out = self.q_out(q_hidden)

        # Return policy output and updated LSTM states
        new_state = [self._hidden_state, self._cell_state]
        return policy_out, new_state

    def value_function(self):
        return torch.reshape(self._q_value_out, [-1])

    def get_initial_state(self):
        return [
            torch.zeros(1, self.lstm_cell_size),
            torch.zeros(1, self.lstm_cell_size),
        ]


# Register the custom model with RLlib
ModelCatalog.register_custom_model("sac_lstm_model", SAC_LSTM_Model)


# Data retrieving
def get_timeframe_data(symbol, from_time, to_time, timeframe):
    h5_db = Hdf5client("binance")
    data = h5_db.get_data(symbol, from_time, to_time)
    if timeframe != "1m":
        data = utils.resample_timeframe(data, timeframe)
    return data


def prepare_additional_data(file_path, asset_prefix, timeframe):
    """
    Prepares additional data in the same format as the EURUSD example and resamples it
    to match the provided timeframe.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.
    asset_prefix : str
        The prefix to prepend to column names, e.g. 'eurusd' or 'ustbond'.
    timeframe : str
        The target timeframe to which the data should be resampled (e.g., '4h', '1h', etc.).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by timestamp at the specified timeframe and columns renamed
        with the asset_prefix.
    """
    # Read the CSV
    df = pd.read_csv(file_path)

    # Convert the timestamp from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["Local time"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Keep only the required columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Rename columns to include the asset prefix
    df.columns = [f"{asset_prefix}_{col.lower()}" for col in df.columns]

    # The original data is in 1m timeframe by default, so resample if needed
    if timeframe != "1m":
        df = utils.resample_timeframe(df, timeframe)

    return df


# Features engineering
def calculate_indicators(df):
    # List of assets including BTC as a special case
    # For BTC, columns are named: close, high, low
    # For others, columns are named: <asset>_close, <asset>_high, <asset>_low, etc.
    assets = [
        "btcusdt",
        "ethusdt",
        "bnbusdt",
        "xrpusdt",
        "solusdt",
        "adausdt",
        "dogeusdt",
        "trxusdt",
        "avaxusdt",
        "shibusdt",
        "dotusdt",
    ]

    for asset in assets:
        if asset == "btcusdt":
            # BTC columns have no prefix
            close_col = "close"
            high_col = "high"
            low_col = "low"
        else:
            # Other assets have prefixed columns
            close_col = f"{asset}_close"
            high_col = f"{asset}_high"
            low_col = f"{asset}_low"

        # --- Simple Moving Averages ---
        df[f"SMA20_{asset}"] = df[close_col].rolling(window=20).mean()
        df[f"SMA50_{asset}"] = df[close_col].rolling(window=50).mean()
        df[f"SMA100_{asset}"] = df[close_col].rolling(window=100).mean()

        # df[f"SMA_week_{asset}"] = df[close_col].rolling(window=168).mean()
        # df[f"SMA_month_{asset}"] = df[close_col].rolling(window=672).mean()
        # df[f"SMA_year_{asset}"] = df[close_col].rolling(window=8064).mean()

        # # --- EMAs ---
        # df[f"EMA20_{asset}"] = df[close_col].ewm(span=20, adjust=False).mean()
        # df[f"EMA50_{asset}"] = df[close_col].ewm(span=50, adjust=False).mean()
        # df[f"EMA100_{asset}"] = df[close_col].ewm(span=100, adjust=False).mean()

        # # --- Bollinger Bands (using SMA20) ---
        # df[f"BB_up_20_{asset}"] = (
        #     df[f"SMA20_{asset}"] + 2 * df[close_col].rolling(window=20).std()
        # )
        # df[f"BB_low_20_{asset}"] = (
        #     df[f"SMA20_{asset}"] - 2 * df[close_col].rolling(window=20).std()
        # )

        # # --- ATR (Average True Range) ---
        # df[f"high-low_{asset}"] = df[high_col] - df[low_col]
        # df[f"high-close_{asset}"] = (df[high_col] - df[close_col].shift()).abs()
        # df[f"low-close_{asset}"] = (df[low_col] - df[close_col].shift()).abs()
        # df[f"TR_{asset}"] = df[
        #     [f"high-low_{asset}", f"high-close_{asset}", f"low-close_{asset}"]
        # ].max(axis=1)
        # df[f"ATR14_{asset}"] = df[f"TR_{asset}"].rolling(window=14).mean()

        # --- RSI (14) ---
        delta = df[close_col].diff()
        gain_14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs_14 = gain_14 / loss_14
        df[f"RSI14_{asset}"] = 100 - (100 / (1 + rs_14))

        # --- MACD (12, 26) and Signal (9) ---
        exp1 = df[close_col].ewm(span=12, adjust=False).mean()
        exp2 = df[close_col].ewm(span=26, adjust=False).mean()
        df[f"MACD_{asset}"] = exp1 - exp2
        df[f"Signal_{asset}"] = df[f"MACD_{asset}"].ewm(span=9, adjust=False).mean()

        # # --- ADX (14) ---
        # df[f"plus_dm_{asset}"] = np.where(
        #     (df[high_col] - df[high_col].shift(1))
        #     > (df[low_col].shift(1) - df[low_col]),
        #     df[high_col] - df[high_col].shift(1),
        #     0,
        # )
        # df[f"minus_dm_{asset}"] = np.where(
        #     (df[low_col].shift(1) - df[low_col])
        #     > (df[high_col] - df[high_col].shift(1)),
        #     df[low_col].shift(1) - df[low_col],
        #     0,
        # )

        # df[f"TR14_{asset}"] = df[f"TR_{asset}"].rolling(window=14).sum()
        # df[f"plus_di_14_{asset}"] = 100 * (
        #     df[f"plus_dm_{asset}"].rolling(window=14).sum() / df[f"TR14_{asset}"]
        # )
        # df[f"minus_di_14_{asset}"] = 100 * (
        #     df[f"minus_dm_{asset}"].rolling(window=14).sum() / df[f"TR14_{asset}"]
        # )

        # df[f"DX14_{asset}"] = 100 * (
        #     (df[f"plus_di_14_{asset}"] - df[f"minus_di_14_{asset}"]).abs()
        #     / (df[f"plus_di_14_{asset}"] + df[f"minus_di_14_{asset}"])
        # )
        # df[f"ADX14_{asset}"] = df[f"DX14_{asset}"].rolling(window=14).mean()

        # # Drop intermediate columns for this asset
        # df.drop(
        #     [
        #         f"high-low_{asset}",
        #         f"high-close_{asset}",
        #         f"low-close_{asset}",
        #         f"TR_{asset}",
        #         f"plus_dm_{asset}",
        #         f"minus_dm_{asset}",
        #         f"TR14_{asset}",
        #         f"DX14_{asset}",
        #         f"SMA20_{asset}",
        #         f"SMA50_{asset}",
        #         f"SMA100_{asset}",
        #     ],
        #     axis=1,
        #     inplace=True,
        # )

    # Drop rows that contain NaN due to rolling calculations
    df = df.dropna()

    return df

    # Normalize the dataframes


# For the realized volatility computation we need to aggregate the 1-min returns.
def compute_realized_vol(df_1min, target_freq="D"):
    """
    Compute realized volatility at a target frequency from 1-min data.
    The function computes 1-min returns, groups them by the target frequency,
    sums the squared returns for each group, then takes the square root.

    Parameters:
      - df_1min: DataFrame of 1-min prices (indexed by datetime)
      - target_freq: frequency string accepted by pd.Grouper (e.g., 'D', '4H', '1H')

    Returns:
      A DataFrame of realized volatilities at the target frequency.
    """
    df_returns = df_1min.pct_change()
    # Group by the target frequency using pd.Grouper and compute sum of squares and sqrt.
    rv = df_returns.pow(2).groupby(df_returns.index.date).sum().apply(np.sqrt)
    # rv = df_returns.pow(2).groupby(pd.Grouper(freq=target_freq)).sum().apply(np.sqrt)
    return rv


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_EMA(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_MACD(series, span_short=12, span_long=26):
    ema_short = compute_EMA(series, span_short)
    ema_long = compute_EMA(series, span_long)
    return ema_short - ema_long


def build_deepcov_features(
    df_prices,
    df_rv,
    realized_cov_dict,
    features_engineered=False,
    df_macro=None,
    df_lunar=None,
    freq="1d",
    target_type="cholesky",
    constraints=None,
):
    """
    Construct a DataFrame of features for deep covariance prediction.
    For each crypto, compute:
      - Resampled price.
      - Log returns (from the resampled price).
      - Realized volatility (computed from 1-min data at the target frequency).
      - HAR features: rolling averages computed over a lookback window.
      - Technical indicators: RSI, EMA12, EMA26, MACD (computed on the resampled price).

    When freq == '1d', macro features are appended.

    Parameters:
      - df_prices: resampled price DataFrame at target frequency.
      - df_rv: realized volatility DataFrame at target frequency.
      - df_macro: macro DataFrame (only used when freq=='1d').
      - freq: chosen frequency as string ('1d', '4h', or '1h').

    Returns:
      A DataFrame of deep model features.
    """
    # Define rolling window sizes depending on frequency.
    # For daily: use 7 and 28 days.
    # For 4h: 7 days = 7*24/4 = 42 periods; 28 days = 28*24/4 = 168 periods.
    # For 1h: 7 days = 7*24 = 168 periods; 28 days = 28*24 = 672 periods.
    if freq.lower() == "1d":
        window_1 = 7
        window_2 = 28
    elif freq.lower() == "4h":
        window_1 = int(7 * 24 / 4)  # 42
        window_2 = int(28 * 24 / 4)  # 168
    elif freq.lower() == "1h":
        window_1 = 7 * 24  # 168
        window_2 = 28 * 24  # 672
    elif freq.lower() == "15m":
        window_1 = 7 * 24 * 4
        window_2 = 28 * 24 * 4
    else:
        raise ValueError("Frequency must be one of '1d', '4h', or '1h'.")

    # Identify the 50 crypto assets from column names
    crypto_assets = sorted(
        set(col.split("_")[0] for col in df_prices.columns if "_close" in col)
    )
    assert len(crypto_assets) == 10, "Expected exactly 10 assets."

    # Initialize lists to store OHLCV and feature DataFrames
    ohlcv_list = []
    feature_list = []

    for crypto in crypto_assets:
        # Extract OHLCV columns for the asset
        ohlcv_cols = [
            # f"{crypto}_open",
            # f"{crypto}_high",
            # f"{crypto}_low",
            f"{crypto}_close",
            f"{crypto}_volume",
        ]
        df_ohlcv = df_prices[ohlcv_cols].copy()

        # Use only the close price for feature calculations
        close_price = df_ohlcv[f"{crypto}_close"]
        df_features = pd.DataFrame(index=df_ohlcv.index)

        df_features[f"{crypto}_log_return"] = np.log(close_price).diff()

        if features_engineered:
            # Compute features from close price
            df_features[f"{crypto}_realized_vol"] = df_rv[crypto]
            df_features[f"{crypto}_RV_roll_1"] = (
                df_features[f"{crypto}_realized_vol"].rolling(window=window_1).mean()
            )
            df_features[f"{crypto}_RV_roll_2"] = (
                df_features[f"{crypto}_realized_vol"].rolling(window=window_2).mean()
            )
            df_features[f"{crypto}_RSI_14"] = compute_RSI(close_price, period=14)
            df_features[f"{crypto}_EMA_12"] = compute_EMA(close_price, span=12)
            df_features[f"{crypto}_EMA_26"] = compute_EMA(close_price, span=26)
            df_features[f"{crypto}_EMA_100"] = compute_EMA(close_price, span=100)
            df_features[f"{crypto}_MACD"] = (
                df_features[f"{crypto}_EMA_12"] - df_features[f"{crypto}_EMA_26"]
            )

        feature_cols = [
            col
            for col in df_features.columns
            if col not in [f"{crypto}_EMA_12", f"{crypto}_EMA_26"]
        ]
        df_features = df_features[feature_cols]
        # Append OHLCV and features to their respective lists
        ohlcv_list.append(df_ohlcv)
        feature_list.append(df_features)

    # Concatenate all OHLCV DataFrames first, then all feature DataFrames
    df_ohlcv_all = pd.concat(ohlcv_list, axis=1)
    df_features_all = pd.concat(feature_list, axis=1)

    print("df combined shape before features engineering : ", df_ohlcv_all.shape)

    # Combine OHLCV and features: OHLCV columns first, followed by features
    df_combined = pd.concat([df_ohlcv_all, df_features_all], axis=1)

    # If macro data is provided and frequency is daily, append macro features
    if df_macro is not None and freq.lower() == "1d":
        print("df macro columns : ", df_macro.columns)
        print("df combined shape before macros : ", df_combined.shape)
        print("df macro passed the test !")
        df_combined = df_combined.join(df_macro, how="left")
        print("df combined shape after macros : ", df_combined.shape)
        print("df combined after macros : ", df_combined)
    if df_lunar is not None and freq.lower() == "1d":
        print("df lunar passed the test !")
        df_combined = df_combined.join(df_lunar, how="left")
        for crypto in lunar_list:
            df_combined = df_combined.drop(f"{crypto}_contributors_created", axis=1)
            df_combined = df_combined.drop(f"{crypto}_posts_created", axis=1)
            df_combined = df_combined.drop(f"{crypto}_spam", axis=1)
        print("df combined shape after lunar : ", df_combined.shape)
        print("df combined after lunar : ", df_combined)

    # # Drop rows with NaN values
    # df_combined = df_combined.dropna()

    # n_nan = df_combined.isna().sum().sum()
    # if n_nan:
    #     by_col = df_combined.isna().sum()
    #     worst = by_col[by_col > 0].sort_values(ascending=False).head(15)
    #     raise ValueError(
    #         f"[build_deepcov_features] {n_nan} NaNs survived merge.\n"
    #         f"Top offenders:\n{worst}"
    #     )

    return df_combined


# def normalize_dataframes(
#     data,
#     ohlc_columns=["open", "high", "low", "close"],
#     volume_column="volume",
#     indicator_columns=[
#         "EMA20",
#         "EMA50",
#         "EMA100",
#         "BB_up_20",
#         "BB_low_20",
#         # "BB_up_50",
#         # "BB_low_50",
#         "ATR14",
#         # "ATR50",
#         "RSI14",
#         # "RSI30",
#         "MACD",
#         "Signal",
#         "plus_di_14",
#         "minus_di_14",
#         "ADX14",
#         # "plus_di_30",
#         # "minus_di_30",
#         # "ADX30",
#     ],
#     epsilon=0.0001,  # Small constant to avoid zero in normalized data
# ):
#     """
#     Normalize the features of financial dataframes.

#     :param data: A dictionary of pandas dataframes keyed by timeframe.
#     :param ohlc_columns: List of columns to be normalized across all dataframes together.
#     :param volume_column: The volume column to be normalized independently for each dataframe.
#     :param indicator_columns: List of other indicator columns to normalize independently for each dataframe.
#     :param epsilon: Small constant to set the lower bound of the normalized range.
#     :return: The dictionary of normalized dataframes and the OHLC scaler used.
#     """
#     # Initialize the scalers
#     ohlc_scaler = MinMaxScaler(
#         feature_range=(epsilon, 1)
#     )  # Set feature range with epsilon
#     volume_scaler = MinMaxScaler(feature_range=(epsilon, 1))

#     # Create a new dictionary to store the normalized dataframes
#     normalized_data = {}

#     # Normalize OHLC data across all timeframes together
#     combined_ohlc = pd.concat([df[ohlc_columns] for df in data.values()], axis=0)
#     scaled_ohlc = ohlc_scaler.fit_transform(combined_ohlc).astype(np.float32)

#     # Distribute the normalized OHLC values back to the original dataframes
#     start_idx = 0
#     for tf, df in data.items():
#         end_idx = start_idx + len(df)
#         # Create a copy of the original dataframe to avoid modifying it
#         normalized_df = df.copy()
#         normalized_df[ohlc_columns] = scaled_ohlc[start_idx:end_idx]
#         # Store the normalized dataframe in the new dictionary
#         normalized_data[tf] = normalized_df
#         start_idx = end_idx

#     # Normalize volume independently for each timeframe
#     for tf, df in normalized_data.items():
#         volume_scaler = MinMaxScaler(
#             feature_range=(epsilon, 1)
#         )  # Reinitialize scaler for each dataframe
#         df[volume_column] = volume_scaler.fit_transform(df[[volume_column]])

#     # Normalize other indicators independently for each indicator within each timeframe
#     for tf, df in normalized_data.items():
#         for col in indicator_columns:
#             if col in df.columns:
#                 scaler = MinMaxScaler(feature_range=(epsilon, 1))
#                 df[[col]] = scaler.fit_transform(df[[col]])

#     return normalized_data, ohlc_scaler


# Add identifiers for the timeframes in order to help the LSTM to make the difference
# def add_timeframe_identifier(data_dict):
#    timeframe_ids = {
#        "15m": 0,
#        "30m": 1,
#        "1h": 2,
#        "1d": 3,
#    }
#    for timeframe, data in data_dict.items():
#        # Assuming `data` is a DataFrame
#        data["timeframe_id"] = timeframe_ids[timeframe]
#    return data_dict


def _asset_from_col(col):
    for a in ASSETS:
        if col.startswith(a):
            return a
    return None


def fit_scalers(df_full: pd.DataFrame, train_end_date: pd.Timestamp, eps=1e-4):
    """
    Fit one scaler (or rule) per column based ONLY on df_train.
    Returns dict {column: scaler_obj or ("rule",param)}.
    """

    df_train = df_full.loc[:train_end_date]
    scalers = {}
    for col in df_train.columns:

        # # 1) OHLC grouped scaling ---------------------------------------
        asset = _asset_from_col(col)
        # if asset and any(col.endswith(sfx) for sfx in ("open", "high", "low", "close")):
        #     lo = df_train[f"{asset}_low"].min()
        #     hi = df_train[f"{asset}_high"].max()
        #     scalers[col] = ("minmax_pair", lo, hi, eps)
        #     continue

        # 2) Volume per asset ------------------------------------------
        if col.endswith("volume"):
            print("VOLUME NORMALIZED !")
            ss = StandardScaler().fit(np.log(df_train[[col]] + 1))
            scalers[col] = ("std_ln", ss, CLIP_Z)  # ← clip later
            continue

        # 3) Macro traded indices – z-score of ln-returns
        if col in PRICE_MACRO:
            print("PRICE MACRO NORMALIZED !")
            lnret = np.log(df_train[col]).diff().dropna().values.reshape(-1, 1)
            ss = StandardScaler().fit(lnret)
            scalers[col] = ("price_lnret", ss, CLIP_Z)
            continue

        # 4) Macro level indicators
        if col in LEVEL_MACRO:
            print("LEVEL MACRO NORMALIZED !")
            ss = StandardScaler().fit(df_train[[col]].astype("float32"))
            scalers[col] = ("std", ss, CLIP_Z)
            continue

        # 5) RSI (bounded 0-100) ---------------------------------------
        if col.endswith("RSI_14"):
            print("RSI NORMALIZED !")
            scalers[col] = ("bounded", 100.0)
            continue

        # 6) -------- MACD ---------------------------------------------------
        if asset and col.endswith("MACD"):
            print("MACD NORMALIZED !")
            # scale relative to price, then z-score
            scaler = StandardScaler().fit(
                (df_train[col] / df_train[f"{asset}_close"]).values.reshape(-1, 1)
            )
            scalers[col] = ("macd_rel", scaler, CLIP_Z)  # <── NEW rule
            continue

        # 7) -------- long EMAs (100 / 200) ---------------------------------
        if re.search(r"EMA_(100|200)$", col):
            print("EMA NORMALIZED !")
            scaler = StandardScaler().fit(
                np.log(df_train[[col]] + 1.0)  # log → stabilise variance
            )
            scalers[col] = ("ln_std", scaler, CLIP_Z)  # <── NEW rule
            continue

        # 8) Log-returns & realised vol → z-score ----------------------
        if col.endswith(("log_return", "realized_vol", "RV_roll_1", "RV_roll_2")):
            ss = StandardScaler().fit(df_train[[col]].astype("float32"))
            scalers[col] = ("std", ss, CLIP_Z)
            continue

        # 9) Default: identity ----------------------------------------
        scalers[col] = ("identity",)

        # 3-b)  ──  LunarCrush metrics  ─────────────────────────────────────
        # ----- bounded 0-100 ------------------------------------------------
        if col.endswith(("sentiment", "galaxy_score", "market_dominance")):
            scalers[col] = ("bounded", 100.0)
            continue

        # ---- AltRank (un-bounded rank metric) ----------
        if col.endswith("alt_rank"):
            # log-scale to tame long tail, then z-score
            ss = StandardScaler().fit(np.log(df_train[[col]]))
            scalers[col] = ("std_ln_inv_rank", ss, CLIP_Z)
            continue

        # ----- Market-cap – log-standardise ----------------------------------
        if col.endswith("market_cap"):
            ss = StandardScaler().fit(np.log(df_train[[col]] + 1))
            scalers[col] = ("std_ln", ss, CLIP_Z)
            continue

        # ----- Activity / count metrics – log-standardise --------------------
        if col.endswith(
            (
                "contributors_active",
                "contributors_created",
                "posts_active",
                "posts_created",
                "interactions",
                "spam",
            )
        ):
            ss = StandardScaler().fit(np.log(df_train[[col]] + 1))
            scalers[col] = ("std_ln", ss, CLIP_Z)
            continue

    return scalers


def transform(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    Apply previously fitted scalers (dict) to a dataframe with matching columns.
    Works with both tuple-based rules and sklearn scaler objects.
    """
    out = df.copy()
    for col, rule in scalers.items():

        asset = _asset_from_col(col)
        # -------- custom tuple rules ---------------------------------
        if isinstance(rule, tuple):
            tag = rule[0]

            if tag == "minmax_pair":
                _, lo, hi, eps = rule
                scaled = (out[col] - lo) / (hi - lo + eps)
                out[col] = scaled * 2.0 - 1.0

            if tag == "bounded":
                _, denom = rule
                out[col] = (out[col] / denom) * 2.0 - 1.0

            # --- MACD relative ---------------------------------------------
            if tag == "macd_rel":
                _, scaler, clip = rule
                rel = out[col] / out[f"{asset}_close"]
                out[col] = scaler.transform(rel.values.reshape(-1, 1)).flatten()
                out[col] = np.clip(out[col], -clip, clip) / clip
                continue

            # --- log-then-standardise --------------------------------------
            if tag == "ln_std":
                _, scaler, clip = rule
                out[col] = scaler.transform(np.log(out[[col]] + 1.0)).flatten()
                out[col] = np.clip(out[col], -clip, clip) / clip
                continue

            if tag == "invert_bounded":
                _, denom = rule
                out[col] = (1.0 - (out[col] / denom)) * 2.0 - 1.0

            if tag == "std_ln":
                _, ss, clip = rule
                out[col] = ss.transform(np.log(out[[col]] + 1)).flatten()
                out[col] = np.clip(out[col], -clip, clip) / clip

            if tag == "std":
                _, ss, clip = rule
                out[col] = ss.transform(out[[col]].values).flatten()
                out[col] = np.clip(out[col], -clip, clip) / clip

            if tag == "price_lnret":
                _, ss, clip = rule
                lnret = np.log(out[col]).diff()
                out[col] = ss.transform(lnret.values.reshape(-1, 1)).flatten()
                out[col] = np.clip(out[col], -clip, clip) / clip

            # -- std_ln_inv_rank  (AltRank) -------------------
            if tag == "std_ln_inv_rank":
                _, ss, clip = rule
                scaled = ss.transform(np.log(out[[col]] + 1.0)).flatten()
                scaled = -scaled  # invert so larger=better
                out[col] = np.clip(scaled, -clip, clip) / clip
                continue

            # "identity" → leave column unchanged
            continue

        # -------- sklearn scalers (MinMax, Standard) -----------------
        else:
            out[col] = rule.transform(out[[col]].values.astype("float32"))

    return out.astype("float32")


def resample_to_frequency(df, freq):
    # Resample the dataframe to the specified frequency using forward-fill to handle NaNs
    return df.resample(freq).ffill()


# Resample crypto prices to the chosen frequency
def resample_crypto_prices(df_1min, freq="1d"):
    """
    Resample 1-minute crypto price data to a given frequency, preserving OHLCV structure.
    """
    freq_map = {"1d": "D", "4h": "4H", "1h": "1H", "15m": "15M"}
    if freq.lower() not in freq_map:
        raise ValueError("Frequency must be one of '1d', '4h', '1h', or '15m'.")

    # Group by asset and resample each OHLCV column appropriately
    assets = sorted(
        set(col.split("_")[0] for col in df_1min.columns if "_close" in col)
    )
    resampled_dfs = []
    for asset in assets:
        asset_cols = [col for col in df_1min.columns if col.startswith(asset)]
        df_asset = df_1min[asset_cols]
        resampled = df_asset.resample(freq_map[freq.lower()]).agg(
            {
                f"{asset}_open": "first",
                f"{asset}_high": "max",
                f"{asset}_low": "min",
                f"{asset}_close": "last",
                f"{asset}_volume": "sum",
            }
        )
        resampled_dfs.append(resampled)
    return pd.concat(resampled_dfs, axis=1)


# Create sequences and split them for the LSTM
def create_and_split_sequences(
    data_dict, input_length, validation_pct, test_pct, base_freq="D"
):
    # Resample all timeframes to the base frequency of 15 minutes
    resampled_data = {
        tf: resample_to_frequency(df, base_freq) for tf, df in data_dict.items()
    }

    # Align lengths by truncating to the shortest length after resampling
    min_length = min(len(df) for df in resampled_data.values())
    aligned_data = {
        tf: df.iloc[:min_length].reset_index(drop=True)
        for tf, df in resampled_data.items()
    }

    # Concatenate data from all timeframes
    concatenated_data = pd.concat(aligned_data.values(), axis=1)

    # Create sequences
    num_sequences = len(concatenated_data) - input_length + 1
    X = np.zeros(
        (num_sequences, input_length, concatenated_data.shape[1]), dtype=np.float32
    )
    # Array for last observations
    last_observations = np.zeros(
        (num_sequences, 4), dtype=np.float32  # only storing OHLC
    )

    for i in range(num_sequences):
        X[i] = concatenated_data.iloc[i : (i + input_length)].values
        # Capture the last observation (close, high, low)
        last_observations[i] = concatenated_data.iloc[i + input_length - 1][
            [0, 1, 2, 3]
        ].values

    # Split the data
    n = X.shape[0]
    test_index = int(n * (1 - test_pct))
    validation_index = int(n * (1 - test_pct - validation_pct))

    train_X = X[:validation_index]
    val_X = X[validation_index:test_index]
    test_X = X[test_index:]

    # Split the last observations data
    train_last_obs = last_observations[:validation_index]
    val_last_obs = last_observations[validation_index:test_index]
    test_last_obs = last_observations[test_index:]

    return (
        train_X.astype(np.float32),
        val_X.astype(np.float32),
        test_X.astype(np.float32),
        train_last_obs.astype(np.float32),
        val_last_obs.astype(np.float32),
        test_last_obs.astype(np.float32),
    )


# Same sequence function with action-state information


def create_and_split_sequences_static(
    data_dict,
    input_length,
    validation_pct,
    test_pct,
    train_end_date,
    val_end_date,
    base_freq="D",
    num_action_state_features=17,  # Number of action-state features to include
    use_date=True,
):
    # Resample all timeframes to the base frequency
    resampled_data = {}
    for tf, df in data_dict.items():
        resampled_df = df.resample(base_freq).ffill()
        if not isinstance(resampled_df.index, pd.DatetimeIndex):
            raise ValueError(f"Resampled data for {tf} does not have a DatetimeIndex.")
        resampled_data[tf] = resampled_df

    # Align lengths by truncating to the shortest length
    min_length = min(len(df) for df in resampled_data.values())
    aligned_data = {}
    for tf, df in resampled_data.items():
        aligned_df = df.iloc[:min_length]
        if not isinstance(aligned_df.index, pd.DatetimeIndex):
            raise ValueError(f"Aligned data for {tf} does not have a DatetimeIndex.")
        aligned_data[tf] = aligned_df

    # Concatenate data from all timeframes
    concatenated_data = pd.concat(aligned_data.values(), axis=1)
    # Ensure the index is a DatetimeIndex
    if not isinstance(concatenated_data.index, pd.DatetimeIndex):
        try:
            concatenated_data.index = pd.to_datetime(concatenated_data.index)
            print("Converted concatenated_data.index to DatetimeIndex.")
        except Exception as e:
            print("Failed to convert index to DatetimeIndex:", e)
            raise

    # Add placeholders for action-state features, preserving the index
    action_state_df = pd.DataFrame(
        np.zeros((len(concatenated_data), num_action_state_features)),
        columns=[f"action_state_{i}" for i in range(num_action_state_features)],
        index=concatenated_data.index,
    )
    concatenated_data = pd.concat([concatenated_data, action_state_df], axis=1)

    # Create sequences
    num_sequences = len(concatenated_data) - input_length + 1
    X = np.zeros(
        (num_sequences, input_length, concatenated_data.shape[1]), dtype=np.float32
    )

    print(f"Creating {num_sequences} sequences...")

    for i in tqdm(range(num_sequences), desc="Sequencing Data", unit="seq"):
        X[i] = concatenated_data.iloc[i : (i + input_length)].values

    if use_date:

        # Get the ending dates for each sequence
        sequence_end_dates = concatenated_data.index[
            input_length - 1 : input_length - 1 + num_sequences
        ]

        print("Type of sequence_end_dates:", type(sequence_end_dates))

        # Verify it's a DatetimeIndex
        if not isinstance(sequence_end_dates, pd.DatetimeIndex):
            raise ValueError("sequence_end_dates is not a DatetimeIndex.")

        # Convert sequence_end_dates to integer nanosecond timestamps
        sequence_end_dates_int = (
            sequence_end_dates.asi8
        )  # Returns a NumPy array of int64

        # Convert split dates to datetime
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        train_end_int = train_end.value  # Integer nanosecond timestamp
        val_end_int = val_end.value  # Integer nanosecond timestamp

        # Find split indices using integer comparisons
        train_end_idx = np.searchsorted(
            sequence_end_dates_int, train_end_int, side="right"
        )
        val_end_idx = np.searchsorted(sequence_end_dates_int, val_end_int, side="right")

        # Ensure indices are within bounds
        train_end_idx = min(train_end_idx, num_sequences)
        val_end_idx = min(val_end_idx, num_sequences)

        # Split the sequences
        train_X = X[:train_end_idx]
        val_X = X[train_end_idx:val_end_idx]
        test_X = X[val_end_idx:]

    else:

        # Split the data
        n = X.shape[0]
        test_index = int(n * (1 - test_pct))
        validation_index = int(n * (1 - test_pct - validation_pct))

        train_X = X[:validation_index]
        val_X = X[validation_index:test_index]
        test_X = X[test_index:]

    # Clean up memory by deleting the DataFrame and triggering garbage collection
    del concatenated_data, aligned_data, resampled_data  # Deleting large variables
    gc.collect()  # Force garbage collection to free up memory

    return (
        train_X.astype(np.float32),
        val_X.astype(np.float32),
        test_X.astype(np.float32),
    )


# Creates the environment for the ray library
def env_creator(env_config):
    return TradingEnvironment(
        # data=env_config["data"],
        input_length=env_config.get("input_length", 100),
        market_fee=env_config.get("market_fee", 0.0005),
        limit_fee=env_config.get("limit_fee", 0.0002),
        liquidation_fee=env_config.get("liquidation_fee", 0.0125),
        slippage_mean=env_config.get("slippage_mean", 0.000001),
        slippage_std=env_config.get("slippage_std", 0.00005),
        initial_balance=env_config.get("initial_balance", 1000),
        total_episodes=env_config.get("total_episodes", 1),
        episode_length=env_config.get("episode_length", 180),
        max_risk=env_config.get("max_risk", 0.02),
        min_risk=env_config.get("min_risk", 0.001),
        min_profit=env_config.get("min_profit", 0),
        limit_bounds=env_config.get("limit_bounds", False),
        margin_mode=env_config.get("margin_mode", "cross"),
        predict_leverage=env_config.get("predict_leverage", False),
        ppo_mode=env_config.get("ppo_mode", True),
        full_invest=env_config.get("full_invest", True),
    )


if __name__ == "__main__":

    # # Get the total system memory
    # total_memory = psutil.virtual_memory().total

    # # Calculate 50% of total system memory
    # memory_to_allocate = total_memory * 0.5

    # from_time = "2019-11-01"
    # to_time = "2024-09-01"
    # symbol = "BTCUSDT"

    # # Define timeframes
    # timeframes = ["1d"]
    # tf = timeframes[0]

    # # Convert times
    # from_time = int(
    #     datetime.datetime.strptime(from_time, "%Y-%m-%d").timestamp() * 1000
    # )
    # to_time = int(datetime.datetime.strptime(to_time, "%Y-%m-%d").timestamp() * 1000)

    # data = get_timeframe_data(symbol, from_time, to_time, tf)
    # ethusdt_df = get_timeframe_data("ETHUSDT", from_time, to_time, tf)
    # bnbusdt_df = get_timeframe_data("BNBUSDT", from_time, to_time, tf)
    # xrpusdt_df = get_timeframe_data("XRPUSDT", from_time, to_time, tf)
    # solusdt_df = get_timeframe_data("SOLUSDT", from_time, to_time, tf)
    # adausdt_df = get_timeframe_data("ADAUSDT", from_time, to_time, tf)
    # dogeusdt_df = get_timeframe_data("DOGEUSDT", from_time, to_time, tf)
    # trxusdt_df = get_timeframe_data("TRXUSDT", from_time, to_time, tf)
    # avaxusdt_df = get_timeframe_data("AVAXUSDT", from_time, to_time, tf)
    # shibusdt_df = get_timeframe_data("1000SHIBUSDT", from_time, to_time, tf)
    # dotusdt_df = get_timeframe_data("DOTUSDT", from_time, to_time, tf)

    # # Rename columns to include the asset prefix
    # ethusdt_df.columns = [f"ethusdt_{col.lower()}" for col in ethusdt_df.columns]
    # bnbusdt_df.columns = [f"bnbusdt_{col.lower()}" for col in bnbusdt_df.columns]
    # xrpusdt_df.columns = [f"xrpusdt_{col.lower()}" for col in xrpusdt_df.columns]
    # solusdt_df.columns = [f"solusdt_{col.lower()}" for col in solusdt_df.columns]
    # adausdt_df.columns = [f"adausdt_{col.lower()}" for col in adausdt_df.columns]
    # dogeusdt_df.columns = [f"dogeusdt_{col.lower()}" for col in dogeusdt_df.columns]
    # trxusdt_df.columns = [f"trxusdt_{col.lower()}" for col in trxusdt_df.columns]
    # avaxusdt_df.columns = [f"avaxusdt_{col.lower()}" for col in avaxusdt_df.columns]
    # shibusdt_df.columns = [f"shibusdt_{col.lower()}" for col in shibusdt_df.columns]
    # dotusdt_df.columns = [f"dotusdt_{col.lower()}" for col in dotusdt_df.columns]

    # print(ethusdt_df)
    # print(bnbusdt_df)
    # print(xrpusdt_df)
    # print(solusdt_df)
    # print(adausdt_df)
    # print(dogeusdt_df)
    # print(trxusdt_df)
    # print(avaxusdt_df)
    # print(shibusdt_df)
    # print(dotusdt_df)

    # ethusdt_close_df = ethusdt_df[["ethusdt_close"]]
    # bnbusdt_close_df = bnbusdt_df[["bnbusdt_close"]]
    # xrpusdt_close_df = xrpusdt_df[["xrpusdt_close"]]

    # # Additional data preparation and resampling to match main_data timeframe
    # eurusd_df = prepare_additional_data(
    #     "data/EURUSD/eurusd_cleaned.csv", "eurusd", timeframe=tf
    # )
    # eurusd_close_df = eurusd_df[["eurusd_close"]]
    # gbpusd_df = prepare_additional_data(
    #     "data/GBPUSD/gbpusd_cleaned.csv", "gbpusd", timeframe=tf
    # )
    # gbpusd_close_df = gbpusd_df[["gbpusd_close"]]
    # xauusd_df = prepare_additional_data(
    #     "data/Gold/xauusd_cleaned.csv", "xauusd", timeframe=tf
    # )
    # xauusd_close_df = xauusd_df[["xauusd_close"]]
    # xleusd_df = prepare_additional_data(
    #     "data/XLE_US_USD/xleusd_cleaned.csv", "xleusd", timeframe=tf
    # )
    # xleusd_close_df = xleusd_df[["xleusd_close"]]
    # xlpusd_df = prepare_additional_data(
    #     "data/XLP_US_USD/xlpusd_cleaned.csv", "xlpusd", timeframe=tf
    # )
    # xlpusd_close_df = xlpusd_df[["xlpusd_close"]]
    # ustbond_df = prepare_additional_data(
    #     "data/US_T-Bonds/ustbond_cleaned.csv", "ustbond", timeframe=tf
    # )
    # ustbond_close_df = ustbond_df[["ustbond_close"]]
    # sp500_df = prepare_additional_data(
    #     "data/SP500/sp500_cleaned.csv", "sp500", timeframe=tf
    # )
    # sp500_close_df = sp500_df[["sp500_close"]]
    # uk100_df = prepare_additional_data(
    #     "data/UK100/uk100_cleaned.csv", "uk100", timeframe=tf
    # )
    # uk100_close_df = uk100_df[["uk100_close"]]
    # aus200_df = prepare_additional_data(
    #     "data/AUS200/aus200_cleaned.csv", "aus200", timeframe=tf
    # )
    # aus200_close_df = aus200_df[["aus200_close"]]
    # chi50_df = prepare_additional_data(
    #     "data/CHI50/chi50_cleaned.csv", "chi50", timeframe=tf
    # )
    # chi50_close_df = chi50_df[["chi50_close"]]
    # dollar_idx_df = prepare_additional_data(
    #     "data/DOLLAR_IDX/dollar_idx_cleaned.csv", "dollar_idx", timeframe=tf
    # )
    # dollar_idx_close_df = dollar_idx_df[["dollar_idx_close"]]
    # eurbond_df = prepare_additional_data(
    #     "data/EUR_Bonds/eurbond_cleaned.csv", "eurbond", timeframe=tf
    # )
    # eurbond_close_df = eurbond_df[["eurbond_close"]]
    # jpn225_df = prepare_additional_data(
    #     "data/JPN225/jpn225_cleaned.csv", "jpn225", timeframe=tf
    # )
    # jpn225_close_df = jpn225_df[["jpn225_close"]]
    # ukbonds_df = prepare_additional_data(
    #     "data/UK_Bonds/ukbonds_cleaned.csv", "ukbonds", timeframe=tf
    # )
    # ukbonds_close_df = ukbonds_df[["ukbonds_close"]]
    # ussc2000_df = prepare_additional_data(
    #     "data/USSC2000/ussc2000_cleaned.csv", "ussc2000", timeframe=tf
    # )
    # ussc2000_close_df = ussc2000_df[["ussc2000_close"]]

    # print(eurusd_df)
    # print(gbpusd_df)
    # print(xauusd_df)
    # print(xleusd_df)
    # print(xlpusd_df)
    # print(ustbond_df)
    # print(sp500_df)
    # print(uk100_df)
    # print(aus200_df)
    # print(chi50_df)
    # print(dollar_idx_df)
    # print(eurbond_df)
    # print(jpn225_df)
    # print(ukbonds_df)
    # print(ussc2000_df)

    # # Merge all into a single DataFrame
    # final_data = (
    #     data.join(ethusdt_df, how="left")
    #     .join(bnbusdt_df, how="left")
    #     .join(xrpusdt_df, how="left")
    #     .join(solusdt_df, how="left")
    #     .join(adausdt_df, how="left")
    #     .join(dogeusdt_df, how="left")
    #     .join(trxusdt_df, how="left")
    #     .join(avaxusdt_df, how="left")
    #     .join(shibusdt_df, how="left")
    #     .join(dotusdt_df, how="left")
    #     .join(eurusd_close_df, how="left")
    #     .join(ustbond_close_df, how="left")
    #     .join(xauusd_close_df, how="left")
    #     .join(xleusd_close_df, how="left")
    #     .join(xlpusd_close_df, how="left")
    #     .join(sp500_close_df, how="left")
    #     .join(gbpusd_close_df, how="left")
    #     .join(uk100_close_df, how="left")
    #     .join(aus200_close_df, how="left")
    #     .join(chi50_close_df, how="left")
    #     .join(dollar_idx_close_df, how="left")
    #     .join(eurbond_close_df, how="left")
    #     .join(jpn225_close_df, how="left")
    #     .join(ukbonds_close_df, how="left")
    #     .join(ussc2000_close_df, how="left")
    # )

    # final_data = final_data.dropna()

    # dataframes = {}

    # for tf in timeframes:
    #     dataframes[tf] = get_timeframe_data(symbol, from_time, to_time, tf)

    # dataframes[tf] = final_data

    # # Syncronize the timeframes after computing the features
    # for tf in timeframes:
    #     dataframes[tf] = calculate_indicators(dataframes[tf]).dropna()

    # latest_start_date = None
    # earliest_end_date = None

    # for df in dataframes.values():
    #     start_date = df.index.min()
    #     end_date = df.index.max()
    #     if latest_start_date is None or start_date > latest_start_date:
    #         latest_start_date = start_date
    #     if earliest_end_date is None or end_date < earliest_end_date:
    #         earliest_end_date = end_date

    # # Ensure all DataFrames start and end on these dates
    # for tf in dataframes:
    #     dataframes[tf] = dataframes[tf][
    #         (dataframes[tf].index >= latest_start_date)
    #         & (dataframes[tf].index <= earliest_end_date)
    #     ]

    # pd.reset_option("display.max_rows")
    # print(dataframes)

    # # Normalize the dataframes and add identifiers in timeframes for the LSTM
    # # normalized_dataframes, ohlc_scaler = normalize_dataframes(dataframes)
    # # normalized_dataframes = add_timeframe_identifier(normalized_dataframes)

    # # Sequence and split the normalized data for the LSTM
    # input_length = 10  # Define the length of the input window
    # validation_pct = 0  # 0% validation set
    # test_pct = 0.1  # 10% test set

    # train_X, val_X, test_X = create_and_split_sequences_static(
    #     dataframes, input_length, validation_pct, test_pct
    # )

    # print("NUM OBSERVATIONS : ", len(train_X))

    # # train_torch_ds, val_torch_ds, test_torch_ds = convert_to_torch_datasets(
    # #     train_X, val_X, test_X, batch_size=batch_size
    # # )

    ########################################
    # Import already made data
    ########################################

    # File paths
    crypto_file = "Diversified_Portfolio_Data_Complete_DRL_3_neo.csv"
    macro_file = "Diversified_Portfolio_Data_Complete_Macro_3.csv"
    lunar_file = "lunarcrush_daily_wide_imputed.csv"

    # Read macro data (daily)
    df_prices_macro = pd.read_csv(macro_file, parse_dates=["Date"], index_col="Date")
    df_prices_macro = df_prices_macro[:"2025-05-20"]
    print("DF MACROS : ", df_prices_macro)

    # Read crypto data (1-minute frequency)
    df_prices_crypto = pd.read_csv(crypto_file, parse_dates=["Date"], index_col="Date")
    df_prices_crypto = df_prices_crypto[:"2025-05-20"]
    print("DF CRYPTOS : ", df_prices_crypto)

    # Read lunarcrush data (daily)
    df_lunarcrush = pd.read_csv(lunar_file, parse_dates=["Date"], index_col="Date")
    df_lunarcrush = df_lunarcrush[:"2025-05-20"]
    print("DF LUNARCRUSH : ", df_lunarcrush)

    # In your XGBoost setup (after resampling / percent‐change):
    df_prices_macro.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_lunarcrush.replace([np.inf, -np.inf], np.nan, inplace=True)

    assert not df_prices_macro.isnull().values.any(), "NaN found in df_prices_macro!"
    assert not df_lunarcrush.isnull().values.any(), "NaN found in df_lunarcrush!"

    # Drop NaNs
    df_prices_crypto = df_prices_crypto.dropna()
    df_prices_macro = df_prices_macro.dropna()
    df_lunarcrush = df_lunarcrush.dropna()

    # Choose the desired frequency as a string: '1d', '4h', or '1h'
    chosen_freq = "1d"  # change as needed

    # For macro data, since it’s daily, we only use it when chosen_freq is '1d'.
    if chosen_freq.lower() == "1d":
        # Create a complete daily index from crypto data and reindex macro data accordingly.
        full_daily_index = pd.date_range(
            start=df_prices_crypto.index.min().date(),
            end=df_prices_crypto.index.max().date(),
            freq="D",
        )
        df_prices_macro = df_prices_macro.reindex(full_daily_index).ffill()
        df_prices_macro.index.name = "Date"
        df_lunarcrush = df_lunarcrush.reindex(full_daily_index).ffill()
        df_lunarcrush.index.name = "Date"
    else:
        # For intraday frequencies, macro data will not be merged.
        df_prices_macro = None
        df_lunarcrush = None

    # Compute realized volatility at the chosen frequency.
    # (This is computed from the original 1-min data.)
    freq_map = {"1d": "D", "4h": "4H", "1h": "1H", "15m": "15M"}

    # Resample crypto prices to the chosen frequency.
    df_prices_resampled = resample_crypto_prices(df_prices_crypto, freq=chosen_freq)
    df_prices_resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prices_resampled.dropna(inplace=True)

    df_realized_vol = pd.concat(
        [
            compute_realized_vol(
                df_prices_crypto[[col]], target_freq=freq_map[chosen_freq.lower()]
            ).rename(columns={col: col.split("_")[0]})
            for col in df_prices_crypto.columns
            if "_close" in col
        ],
        axis=1,
    )
    # df_realized_vol.columns = [col.split("_")[0] for col in df_realized_vol.columns]

    # Compute returns from the resampled prices.
    df_returns_crypto = df_prices_resampled.pct_change().dropna()
    crypto_assets = df_returns_crypto.columns
    n_assets = len(crypto_assets)

    print("Crypto assets:", crypto_assets)
    if df_prices_macro is not None:
        print("Macro features available:", df_prices_macro.columns)
    else:
        print("No macro features (intraday frequency chosen).")
    if df_lunarcrush is not None:
        print("LunarCrush features available:", df_lunarcrush.columns)
    else:
        print("No LunarCrush features (intraday frequency chosen).")

    #############################################
    # 3. Compute Realized Covariance Targets from 1-min Data
    #############################################

    # Let chosen_freq be one of '1d', '4h', or '1h'
    freq_map = {"1d": "D", "4h": "4H", "1h": "1H", "15m": "15M"}
    target_freq = freq_map[chosen_freq.lower()]  # e.g. 'D' for daily

    # Compute minute‐level returns from the 1‐min price data
    df_returns_min = df_prices_crypto.pct_change()

    # Create a dictionary of “true” realized covariance matrices using minute data,
    # grouped by the target frequency.
    realized_cov_dict = {}
    for period, group in df_returns_min.groupby(pd.Grouper(freq=target_freq)):
        # Only compute if there are enough observations (at least 2)
        if len(group) > 1:
            print(len(group))
            cov_matrix = np.cov(group.values.T, ddof=1)
            realized_cov_dict[period] = cov_matrix

    constraints = {
        "long_only": False,
        "use_sentiment": False,
        "sentiment_window": 30,
        "tau_value": 0.5,
        "date_range_filter": False,
        "include_transaction_fees": True,
        "fees": 0.0005,
        "turnover_limit": None,
        "net_exposure": True,
        "net_exposure_value": 1,
        "net_exposure_constraint_type": "Equality constraint",
        "leverage_limit": True,
        "leverage_limit_value": 5,
        "leverage_limit_constraint_type": "Inequality constraint",
        "include_risk_free_asset": False,
        "risk_free_rate": 0.01,
        "min_weight_value": -5,
        "max_weight_value": 5,
    }

    # # Filter columns ending with '_close' or '_volume'
    # df_filtered = DF_PRICES_RESAMPLED.loc[:, DF_PRICES_RESAMPLED.columns.str.endswith(('_close', '_volume'))]

    print("DF_PRICES_RESAMPLED : ", df_prices_resampled)

    lunar_list = ["ada", "bnb", "btc", "doge", "eos", "eth", "ltc", "trx", "xlm", "xrp"]

    # Build the features using the chosen frequency.
    df_features = build_deepcov_features(
        df_prices_resampled,
        df_realized_vol,
        realized_cov_dict,
        features_engineered=False,
        df_macro=None,
        df_lunar=None,
        freq=chosen_freq,
        target_type="gmv",
        constraints=constraints,
    )
    df_features = df_features.sort_index()
    # print("FEATURES JUST AFTER BUILDING : ", df_features)
    # df_features = df_features.dropna()
    # # 1) strict index equality: prices ↔ features
    # assert df_prices_resampled.index.equals(
    #     df_features.index
    # ), "Resampled OHLCV & feature frame mis-aligned"

    # 2) sanity on value ranges (catch ±Inf early)
    _bad = np.isinf(df_features.values).sum()
    assert _bad == 0, f"{_bad} ±Inf values found in features"

    # Assuming 'df' is your DataFrame
    zero_positions = df_features == 0

    # Extract the positions where the condition is True
    zero_locations = zero_positions.stack()

    # Filter to get only the positions with zero values
    zero_locations = zero_locations[zero_locations]

    # Convert to a list of tuples (index, column)
    zero_indices = list(zero_locations.index)

    print("Positions with zero values:")
    for row_index, column_name in zero_indices:
        print(f"Row: {row_index}, Column: {column_name}")

    ### NEW Normalizing step

    # -----------------------------------------------------------
    ASSETS = [
        "adausdt",  # 0
        "bnbusdt",  # 1
        "btcusdt",  # 2
        "dogeusdt",  # 3
        "ethusdt",  # 4   (EOS removed, everything re-indexed)
        "ltcusdt",  # 5
        "trxusdt",  # 6
        "xlmusdt",  # 7
        "xrpusdt",  # 8
        "neousdt",
    ]
    #     # remaining 41 contracts – any order is fine
    #     s.lower()
    #     for s in [
    #         "BCHUSDT",
    #         "ETCUSDT",
    #         "LINKUSDT",
    #         "XMRUSDT",
    #         "DASHUSDT",
    #         "ZECUSDT",
    #         "XTZUSDT",
    #         "ATOMUSDT",
    #         "ONTUSDT",
    #         "IOTAUSDT",
    #     ]
    # ]
    PRICE_MACRO = [  # traded indices / ETFs
        "Global_Stocks",
        "Emerging_Markets",
        "Intl_Bonds",
        "Broad_Commodities",
        "Gold",
        "Global_RealEstate",
        "High_Yield_Bonds",
        "Inv_Grade_Bonds",
        "Crude_Oil",
        "Copper",
        "Semiconductors",
    ]

    LEVEL_MACRO = [  # economic-level & vol indices
        "Volatility_Index",
        "US_Dollar_Index",
        "10y_Treasury_Yield",
        "Bond_Volatility",
        "Inflation_TIPS",
        "Gold_Volatility",
        "Oil_Volatility",
        "Cash",
    ]
    CLIP_Z = 8.0  # <── central place to adjust the winsor-level
    # -----------------------------------------------------------

    # 1) split chronologically BEFORE fitting scalers
    df_train = df_features.loc[:"2023-04-30"]
    df_val = df_features.loc["2023-05-01":"2024-02-08"]
    df_test = df_features.loc["2024-02-09":]

    # print("DF TRAIN : ", df_train)

    # assert not np.isnan(df_features.values).any(), "NaNs before scaling"
    # assert not np.isinf(df_features.values).any(), "±Inf before scaling"

    # 2) fit column-wise scalers only on the train slice
    scalers = fit_scalers(df_features, "2023-04-30")

    # 3) transform all three sets with the SAME scalers
    df_features_scaled = transform(df_features, scalers)

    df_features_scaled = df_features_scaled.loc["2020-01-01":"2025-05-20"]
    # print("FEATURES JUST AFTER NORMALIZATION : ", df_features_scaled)

    # Identify the 10 crypto assets from column names
    crypto_assets = sorted(
        set(col.split("_")[0] for col in df_prices_resampled.columns if "_close" in col)
    )
    assert len(crypto_assets) == 10, "Expected exactly 10 assets."
    for crypto in crypto_assets:
        df_features_scaled = df_features_scaled.drop(f"{crypto}_close", axis=1)

    df_features_scaled = df_features_scaled.dropna()
    df_features_scaled.describe()
    assert not np.isnan(df_features_scaled.values).any(), "NaNs after scaling"
    assert not np.isinf(df_features_scaled.values).any(), "±Inf after scaling"
    # df_val_scaled = transform_with_scalers(df_val, feature_scalers)
    # df_test_scaled = transform_with_scalers(df_test, feature_scalers)

    # gmv_target_dict = build_gmv_targets(realized_cov_dict, crypto_assets, constraints)

    pd.set_option("display.max_rows", 10000)
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.width", 100)

    print("Feature DataFrame shape: ", df_features_scaled.shape)
    print("Feature DataFrame columns: ", df_features_scaled.columns.tolist())
    print("Feature Dataframe: ", df_features_scaled)
    # Maximum values per column
    print("Maximum values:")
    print(df_features_scaled.max())

    # Minimum values per column
    print("\nMinimum values:")
    print(df_features_scaled.min())
    # print("Feature Dataframe columns: ", df_features_scaled.columns.tolist())

    # print("Raw OHLCV DataFrame shape before sync: ", df_prices_resampled.shape)
    # print("Raw OHLCV Dataframe before sync: ", df_prices_resampled)

    n_dropped = len(df_prices_resampled) - len(df_features_scaled)  # rows lost
    # assert len(df_prices_resampled) == len(
    #     df_features_scaled
    # ), "Raw OHLCV & Features length diverged – check earlier drop logic"
    df_prices_resampled = df_prices_resampled.iloc[n_dropped:]  # ↔ keep in sync

    df_raw_ohlcv_train = df_prices_resampled.loc[:"2023-04-30"]
    df_raw_ohlcv_val = df_prices_resampled.loc["2023-05-01":"2024-02-08"]
    df_raw_ohlcv_test = df_prices_resampled.loc["2024-02-09":]

    print("Raw OHLCV DataFrame shape: ", df_prices_resampled.shape)
    print("Raw OHLCV Dataframe: ", df_prices_resampled)
    print("Raw OHLCV Dataframe columns: ", df_prices_resampled.columns.tolist())

    dataframes = {}
    dataframes[chosen_freq] = df_features_scaled

    print(dataframes[chosen_freq])

    # Sequence and split the normalized data for the LSTM
    input_length = 100  # Define the length of the input window
    validation_pct = 0.17  # 0% validation set
    test_pct = 0.15  # 10% test set

    train_X, val_X, test_X = create_and_split_sequences_static(
        dataframes,
        input_length,
        validation_pct,
        test_pct,
        "2023-04-30",
        "2024-02-08",
        base_freq="D",
        num_action_state_features=17,
        use_date=True,
    )
    print()
    print(train_X)
    print()
    print("Train shape:", train_X.shape)
    print("Validation shape:", val_X.shape)
    print("Test shape:", test_X.shape)

    assert not np.isnan(train_X).any(), "NaNs after sequencing"
    assert not np.isinf(train_X).any(), "±Inf after sequencing"
    assert not np.isnan(test_X).any(), "NaNs after sequencing"
    assert not np.isinf(test_X).any(), "±Inf after sequencing"

    # Register the environment in gymnasium
    register(
        id="trade_env_ray_portfolio",
        entry_point="trade_env_ray_portfolio:TradingEnvironment",
    )

    # Save the dataset to a file
    np.save(
        "train_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1_neo.npy",
        train_X,
    )
    np.save(
        "val_portfolio_data_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1_neo.npy",
        val_X,
    )
    np.save(
        "test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1_neo.npy",
        test_X,
    )

    df_raw_ohlcv_train_np = df_raw_ohlcv_train.to_numpy(dtype=np.float32)
    df_raw_ohlcv_val_np = df_raw_ohlcv_val.to_numpy(dtype=np.float32)
    df_raw_ohlcv_test_np = df_raw_ohlcv_test.to_numpy(dtype=np.float32)

    np.save("train_raw_ohlcv_100_1d_NewVal2_neo.npy", df_raw_ohlcv_train_np)
    np.save("val_raw_ohlcv_100_1d_NewVal2_neo.npy", df_raw_ohlcv_val_np)
    np.save("test_raw_ohlcv_100_1d_NewVal2_neo.npy", df_raw_ohlcv_test_np)

    # Define the environment creator function
    def env_creator(env_config):
        return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trade_env_ray_portfolio", env_creator)

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=10 * (1024**3))

    print("Converting numpy dataset to ray object....")
    train_X_ds = from_numpy(train_X)
    val_X_ds = from_numpy(val_X)
    test_X_ds = from_numpy(test_X)

    df_raw_ohlcv_train_ds = from_numpy(df_raw_ohlcv_train_np)
    df_raw_ohlcv_val_ds = from_numpy(df_raw_ohlcv_val_np)
    df_raw_ohlcv_test_ds = from_numpy(df_raw_ohlcv_test_np)
    print("Convertion complete. ")

    del dataframes
    del train_X
    del val_X
    del test_X
    gc.collect()

    print("Saving the ray object datasets....")
    train_X_ds.write_parquet(
        "train_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1_neo"
    )
    val_X_ds.write_parquet(
        "val_portfolio_data_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1_neo"
    )
    test_X_ds.write_parquet(
        "test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1_neo"
    )

    df_raw_ohlcv_train_ds.write_parquet("train_raw_ohlcv_100_1d_NewVal2_neo")
    df_raw_ohlcv_val_ds.write_parquet("val_raw_ohlcv_100_1d_NewVal2_neo")
    df_raw_ohlcv_test_ds.write_parquet("test_raw_ohlcv_100_1d_NewVal2_neo")
    print("Ray datasets saved. ")

    del train_X_ds
    del val_X_ds
    del test_X_ds
    gc.collect()

    # # Define the search space
    # search_space = {
    #     "lr": tune.loguniform(1e-5, 1e-1),  # Learning rate
    #     # "train_batch_size": tune.choice([1024, 2048]),
    #     "sgd_minibatch_size": tune.choice([50, 100]),
    #     "num_sgd_iter": tune.choice([10, 20, 30]),
    #     "gamma": tune.quniform(0.95, 0.99, 0.01),  # Range for gamma
    #     "model": {
    #         "lstm_cell_size": tune.choice([32, 64, 128]),
    #         "fcnet_hiddens": tune.choice([[16], [32], [64]]),
    #     },
    # }
    # # Scheduler to prune less promising trials
    # scheduler = HyperBandScheduler(
    #     time_attr="training_iteration",
    #     max_t=10,  # maximum iterations per configuration
    #     reduction_factor=3,
    #     stop_last_trials=True,
    # )

    # np.set_printoptions(threshold=3000)
    # train_np = np.load("train_raw_ohlcv_100_1d.npy")
    # print(train_np)
    # print("SAVED NP SHAPE : ", train_np.shape)
    # print()
    # print("DATASET : ", df_raw_ohlcv_train)

    # # Define paths to datasets
    # train_data_path = os.path.abspath(
    #     "train_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage3"
    # )
    # val_data_path = os.path.abspath(
    #     "val_portfolio_data_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage3"
    # )
    # test_data_path = os.path.abspath(
    #     "test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage3"
    # )

    # train_raw_path = os.path.abspath("train_raw_ohlcv_100_1d_NewVal2.npy")
    # val_raw_path = os.path.abspath("val_raw_ohlcv_100_1d_NewVal2.npy")
    # test_raw_path = os.path.abspath("test_raw_ohlcv_100_1d_NewVal2.npy")

    # # Verify that the files exist
    # assert os.path.exists(
    #     train_data_path
    # ), f"Training dataset not found at {train_data_path}"
    # assert os.path.exists(
    #     val_data_path
    # ), f"Validation dataset not found at {val_data_path}"

    # # Print paths for debugging
    # print(f"Training data path: {train_data_path}")
    # print(f"Validation data path: {val_data_path}")

    # entropy_coeff_schedule = [
    #     [0, 0.01],  # start very exploratory
    #     [5.2e6, 0.005],  # after ~1 M env steps
    #     [6.2e6, 0.002],  # match LR/KL decay
    # ]

    # with open(f"env_state_{274}.json") as f:
    #     states_ = json.load(f)

    # # Configuration using PPOConfig
    # config = PPOConfig()
    # config.environment(
    #     env="trade_env_ray_portfolio",
    #     env_config={
    #         "data_path": train_data_path,
    #         "raw_data_path": train_raw_path,
    #         "states": states_,
    #         "mode": "train",
    #         "leverage": 5,
    #         "input_length": 100,
    #         "market_fee": 0.0005,
    #         "limit_fee": 0.0002,
    #         "slippage_mean": 0.000001,
    #         "slippage_std": 0.00005,
    #         "initial_balance": 1000,
    #         "total_episodes": 1,
    #         "episode_length": 180,
    #         "max_risk": 0.02,
    #         "min_risk": 0.001,
    #         "min_profit": 0,
    #         "limit_bounds": False,
    #         "margin_mode": "cross",
    #         "predict_leverage": False,
    #         "ppo_mode": True,
    #         "full_invest": True,
    #     },
    # )
    # config.framework("torch")
    # config.resources(num_gpus=1, num_cpus_per_worker=1)
    # config.rollouts(
    #     num_rollout_workers=14,
    #     rollout_fragment_length=120,  # 1 day of data
    #     batch_mode="complete_episodes",
    # )
    # config.training(
    #     gamma=0.97,
    #     lr=1e-4,
    #     lr_schedule=[[0, 1e-4], [6.2e6, 5e-5], [10e6, 2e-5], [15e6, 1e-5]],
    #     train_batch_size=1680,
    #     sgd_minibatch_size=280,
    #     num_sgd_iter=10,
    #     shuffle_sequences=False,
    #     grad_clip=0.5,
    #     lambda_=0.9,
    #     entropy_coeff=0.01,
    #     entropy_coeff_schedule=entropy_coeff_schedule,
    #     clip_param=0.1,
    #     vf_clip_param=0.5,
    #     vf_loss_coeff=0.3,
    #     kl_coeff=0.5,
    #     kl_target=0.015,
    #     use_kl_loss=True,
    # )
    # # Access the model configuration directly via the `.model` attribute
    # config.model["use_lstm"] = True
    # config.model["lstm_cell_size"] = 128
    # config.model["fcnet_hiddens"] = [128, 128]
    # config.model["fcnet_activation"] = "relu"
    # config.model["post_fcnet_activation"] = "linear"
    # config.model["lstm_use_prev_action_reward"] = True
    # config.model["max_seq_len"] = 100
    # config.model["_disable_action_flattening"] = True

    # # # Evaluation configuration for One Big Episode Method
    # # config.evaluation(
    # #     evaluation_interval=10,  # Evaluate every 50 iterations
    # #     evaluation_duration="auto",  # Use 10 episodes per evaluation
    # #     evaluation_config={
    # #         "env_config": {
    # #             "data_path": val_data_path,  # Path to validation data
    # #         },
    # #     },
    # # )

    # # --- new evaluation block ---
    # config.evaluation(
    #     evaluation_interval=10,  # validate every iteration
    #     evaluation_duration=5,  # run as long as training step runs
    #     evaluation_config={
    #         "env_config": {
    #             "data_path": val_data_path,  # validation split
    #             "raw_data_path": val_raw_path,
    #             "mode": "val",
    #             "limit_bounds": False,
    #         },
    #         "explore": False,  # greedy policy during eval
    #         "num_gpus": 0,  # or 1 if your model is large
    #     },
    #     evaluation_sample_timeout_s=180,  # fail fast if env hangs
    #     # evaluation_force_reset_envs_before_iteration=True,
    # )

    # # Custom stopper for One Big Episode Method
    # def custom_stopper(trial_id, result):
    #     if "evaluation/episode_reward_mean" in result:
    #         if not hasattr(custom_stopper, "best_reward"):
    #             custom_stopper.best_reward = result["evaluation/episode_reward_mean"]
    #             custom_stopper.counter = 0
    #         else:
    #             if (
    #                 result["evaluation/episode_reward_mean"]
    #                 > custom_stopper.best_reward + 0.005
    #             ):
    #                 custom_stopper.best_reward = result[
    #                     "evaluation/episode_reward_mean"
    #                 ]
    #                 custom_stopper.counter = 0
    #             else:
    #                 custom_stopper.counter += 1
    #             if custom_stopper.counter >= 10:
    #                 return True  # Stop training
    #     return False  # Continue training

    # stopper = tune.stopper.TrialPlateauStopper(
    #     metric="evaluation/episode_reward_mean",
    #     std=0.01,  # plateau definition
    #     num_results=10,  # look‑back window
    #     grace_period=20,
    # )  # allow warm‑up

    # # Apply the stopper to the config
    # config.stop = custom_stopper

    # # Add evaluation configuration
    # config.evaluation(
    #     evaluation_interval=100,
    #     evaluation_duration=10,
    #     evaluation_config={
    #         "env_config": {
    #             "data_path": val_data_path,
    #         },
    #     },
    # )

    # # Define the stopper
    # stopper = TrialPlateauStopper(
    #     metric="evaluation/episode_reward_mean",
    #     std=0.01,
    #     num_results=10,
    #     grace_period=20,
    # )

    # # Verify configuration
    # # print(config.to_dict())  # This will print the current configuration as a dictionary

    # checkpoint_path = r"C:\Users\marko\ray_results\Full_episode_LowLambda_stage3\PPO_trade_env_ray_portfolio_faf88_00000_0_2025-06-03_22-11-48\checkpoint_000273"

    # ### LIST TO DO :
    # ### 1) TRAIN WITH 21 FEATURES AND CORRECT FREE COLLATERAL
    # ### 2) TRAIN WITH 17 FEATURES AND CORRECT FREE COLLATERAL
    # ### 3) (ONLY IF 2) IS NOT GOOD ENOUGH) CONTINUE FROM 17 FEATURES AND BAD FREE COLLATERALS
    # ### 4) TRAIN WITH (17 OR 21 DEPENDS ON WHICH ONE HAS MORE POTENTIAL) FEATURES AND NOT NORMALIZED STATES AND ONLY NORMALIZED REWARD WITH GOOD FREE COLLATERALS
    # ### 5) TRAIN WITH 17 FEATURES WITH NORMALIZED STATES AND REWARD AND GOOD COLLATERALS, NORMALIZE REWARDS USING WELFORD
    # ### 6) TRAIN WITH 17 FEATURES WITH NORMALIZED STATES AND REWARD AND GOOD COLLATERALS AND LUNARCRUSH DATA, USE THE BEST NORMALIZATION FOR THE REWARD

    # results = tune.run(
    #     "PPO",
    #     config=config,
    #     metric="episode_reward_mean",
    #     mode="max",
    #     # stop=stopper,
    #     # num_samples=10,  # Number of different sets of hyperparameters to try
    #     search_alg=basic_variant.BasicVariantGenerator(),  # Simple random search
    #     # scheduler=scheduler,
    #     verbose=1,
    #     checkpoint_freq=3,  # Save a checkpoint every 10 training iterations
    #     checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end of training
    #     local_dir=r"C:\Users\marko\ray_results\Full_episode_LowLambda_stage3",
    #     restore=checkpoint_path,
    # )

    # # Access the best trial's results and checkpoints
    # best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print(
    #     "Best trial final reward: {}".format(
    #         best_trial.last_result["episode_reward_mean"]
    #     )
    # )

    # # Initialize SAC configuration
    # config = SACConfig()

    # # # Disable the new API stack to use the old API
    # # config.api_stack(
    # #     enable_rl_module_and_learner=False,
    # #     enable_env_runner_and_connector_v2=False,
    # # )

    # # Environment setup (identical to PPO)
    # config.environment(
    #     env="trade_env_ray_portfolio",
    #     env_config={
    #         "data_path": train_data_path,
    #         "input_length": 100,
    #         "market_fee": 0.0005,
    #         "limit_fee": 0.0002,
    #         "slippage_mean": 0.000001,
    #         "slippage_std": 0.00005,
    #         "initial_balance": 1000,
    #         "total_episodes": 1,
    #         "episode_length": 120,
    #         "max_risk": 0.02,
    #         "min_risk": 0.001,
    #         "min_profit": 0,
    #         "limit_bounds": False,
    #     },
    # )

    # # Framework (same as PPO)
    # config.framework("torch")

    # # Resources (same as PPO)
    # config.resources(num_gpus=1, num_cpus_per_worker=1)

    # # Rollouts
    # # SAC is off-policy and uses a replay buffer, but rollout workers still collect data
    # config.rollouts(
    #     num_rollout_workers=10,  # Same as PPO
    #     rollout_fragment_length=120,  # Same as PPO (1 day of data)
    #     enable_connectors=True,
    #     # Note: batch_mode is not directly applicable as SAC uses a replay buffer
    # )

    # # Training configuration
    # # SAC has different training parameters; mapped from PPO where possible
    # config.training(
    #     gamma=0.99,  # Same discount factor as PPO
    #     optimization_config={
    #         "actor_learning_rate": 3e-4,  # Same as PPO's lr, though SAC often uses 3e-4
    #         "critic_learning_rate": 3e-4,  # Matched to PPO's lr
    #         "entropy_learning_rate": 3e-4,  # Matched to PPO's lr
    #     },
    #     train_batch_size=280,  # Similar to PPO's sgd_minibatch_size (PPO uses 1200 total, 200 per minibatch)
    #     grad_clip=0.5,  # Same gradient clipping as PPO
    #     tau=0.005,  # Soft target update parameter (SAC-specific, default value)
    #     initial_alpha=1.0,  # Initial entropy coefficient (SAC-specific, default)
    #     target_entropy="auto",  # Automatically tune entropy (SAC-specific)
    #     twin_q=True,
    #     target_network_update_freq=1,
    #     n_step=1,
    #     store_buffer_in_checkpoints=True,
    #     replay_buffer_config={
    #         "type": "MultiAgentPrioritizedReplayBuffer",
    #         "capacity": 1000000,
    #         "prioritized_replay_alpha": 0.6,
    #         "prioritized_replay_beta": 0.4,
    #         "prioritized_replay_eps": 1e-6,
    #     },
    #     policy_model_config={
    #         "custom_model": "sac_lstm_model",
    #         "custom_model_config": {
    #             "lstm_cell_size": 64,
    #             "fcnet_hiddens": [32, 32],
    #             "fcnet_activation": "relu",
    #         },
    #     },
    #     q_model_config={
    #         "custom_model": "sac_lstm_model",
    #         "custom_model_config": {
    #             "lstm_cell_size": 64,
    #             "fcnet_hiddens": [32, 32],
    #             "fcnet_activation": "relu",
    #         },
    #     },
    # )

    # # Model configuration with custom LSTM model
    # config.model = {
    #     "custom_model": "sac_lstm_model",  # Use our custom model
    #     "custom_model_config": {
    #         "lstm_cell_size": 64,
    #         "fcnet_hiddens": [32, 32],
    #         "fcnet_activation": "relu",
    #         "lstm_use_prev_action_reward": True,
    #         "max_seq_len": 100,
    #     },
    # }

    # config.model["use_lstm"] = True
    # config.model["lstm_cell_size"] = 64
    # config.model["fcnet_hiddens"] = [32, 32]
    # config.model["fcnet_activation"] = "relu"
    # config.model["post_fcnet_activation"] = "linear"
    # config.model["lstm_use_prev_action_reward"] = True
    # config.model["max_seq_len"] = 100
    # config.model["_disable_action_flattening"] = True

    # # Model configuration using the new RLModule API
    # config.rl_module(
    #     model_config_dict={
    #         "fcnet_hiddens": [32, 32],  # Same as before
    #         "fcnet_activation": "relu",  # Same as before
    #         "post_fcnet_activation": "linear",
    #         "use_lstm": True,  # Enable LSTM
    #         "lstm_cell_size": 64,  # Same as before
    #         "max_seq_len": 100,  # Same as before
    #         "lstm_use_prev_action": True,  # Include previous action in LSTM state
    #         "lstm_use_prev_reward": True,  # Include previous reward in LSTM state
    #     },
    # )
    # config.model["_disable_action_flattening"] = True
    # # Remove the extra key that LSTMWrapper does not expect.
    # config.model.pop("policy_model_config", None)

    # Note: _disable_action_flattening is not explicitly needed for SAC; omitted

    # # IMPORTANT: disable the new RLModule API to ensure native recurrent support is used.
    # config.api_stack(
    #     enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    # )

    # # Evaluation configuration (same as PPO)
    # config.evaluation(
    #     evaluation_interval=100,  # Evaluate every 100 iterations
    #     evaluation_duration=10,  # Use 10 episodes per evaluation
    #     evaluation_config={
    #         "env_config": {
    #             "data_path": val_data_path,  # Path to validation data
    #         },
    #     },
    # )

    # # Custom stopper (identical to PPO)
    # def custom_stopper(trial_id, result):
    #     if "evaluation/episode_reward_mean" in result:
    #         if not hasattr(custom_stopper, "best_reward"):
    #             custom_stopper.best_reward = result["evaluation/episode_reward_mean"]
    #             custom_stopper.counter = 0
    #         else:
    #             if (
    #                 result["evaluation/episode_reward_mean"]
    #                 > custom_stopper.best_reward + 0.005
    #             ):
    #                 custom_stopper.best_reward = result[
    #                     "evaluation/episode_reward_mean"
    #                 ]
    #                 custom_stopper.counter = 0
    #             else:
    #                 custom_stopper.counter += 1
    #             if custom_stopper.counter >= 10:
    #                 return True  # Stop training
    #     return False  # Continue training

    # # Run the experiment with SAC
    # results = tune.run(
    #     "SAC",  # Use SAC algorithm instead of PPO
    #     config=config,
    #     metric="episode_reward_mean",
    #     mode="max",
    #     stop=custom_stopper,  # Same stopping criterion as PPO
    #     search_alg=BasicVariantGenerator(),  # Same simple random search
    #     verbose=1,
    #     checkpoint_freq=10,  # Save checkpoints every 10 iterations
    #     checkpoint_at_end=True,  # Ensure a checkpoint at the end
    #     # Adjust local_dir or restore as needed, similar to PPO
    #     # local_dir=r"C:\Users\marko\ray_results\SAC_...",
    #     # restore=checkpoint_path,  # If resuming from a checkpoint
    # )

    # # Access the best trial's results
    # best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print(
    #     "Best trial final reward: {}".format(
    #         best_trial.last_result["episode_reward_mean"]
    #     )
    # )
