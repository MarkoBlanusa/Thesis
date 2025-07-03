import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
from typing import Dict
import gc
import psutil
import cProfile
import pstats
import gymnasium
import time

from binance import BinanceClient
from database import Hdf5client
import data_collector
import utils
from test_trade_env_ray_portfolio import TradingEnvironment

from tqdm import tqdm
import json

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search import basic_variant
from ray.rllib.algorithms.ppo import PPO
from ray.data import from_pandas, Dataset, from_numpy
from ray.data.preprocessors import Concatenator
from ray.tune import CLIReporter

from torch.utils.data import Dataset, DataLoader
import torch


from gymnasium.envs.registration import register

# initialize torch and neural networks
torch, nn = try_import_torch()


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

        df[f"SMA_week_{asset}"] = df[close_col].rolling(window=168).mean()
        df[f"SMA_month_{asset}"] = df[close_col].rolling(window=672).mean()
        df[f"SMA_year_{asset}"] = df[close_col].rolling(window=8064).mean()

        # --- EMAs ---
        df[f"EMA20_{asset}"] = df[close_col].ewm(span=20, adjust=False).mean()
        df[f"EMA50_{asset}"] = df[close_col].ewm(span=50, adjust=False).mean()
        df[f"EMA100_{asset}"] = df[close_col].ewm(span=100, adjust=False).mean()

        # --- Bollinger Bands (using SMA20) ---
        df[f"BB_up_20_{asset}"] = (
            df[f"SMA20_{asset}"] + 2 * df[close_col].rolling(window=20).std()
        )
        df[f"BB_low_20_{asset}"] = (
            df[f"SMA20_{asset}"] - 2 * df[close_col].rolling(window=20).std()
        )

        # --- ATR (Average True Range) ---
        df[f"high-low_{asset}"] = df[high_col] - df[low_col]
        df[f"high-close_{asset}"] = (df[high_col] - df[close_col].shift()).abs()
        df[f"low-close_{asset}"] = (df[low_col] - df[close_col].shift()).abs()
        df[f"TR_{asset}"] = df[
            [f"high-low_{asset}", f"high-close_{asset}", f"low-close_{asset}"]
        ].max(axis=1)
        df[f"ATR14_{asset}"] = df[f"TR_{asset}"].rolling(window=14).mean()

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

        # --- ADX (14) ---
        df[f"plus_dm_{asset}"] = np.where(
            (df[high_col] - df[high_col].shift(1))
            > (df[low_col].shift(1) - df[low_col]),
            df[high_col] - df[high_col].shift(1),
            0,
        )
        df[f"minus_dm_{asset}"] = np.where(
            (df[low_col].shift(1) - df[low_col])
            > (df[high_col] - df[high_col].shift(1)),
            df[low_col].shift(1) - df[low_col],
            0,
        )

        df[f"TR14_{asset}"] = df[f"TR_{asset}"].rolling(window=14).sum()
        df[f"plus_di_14_{asset}"] = 100 * (
            df[f"plus_dm_{asset}"].rolling(window=14).sum() / df[f"TR14_{asset}"]
        )
        df[f"minus_di_14_{asset}"] = 100 * (
            df[f"minus_dm_{asset}"].rolling(window=14).sum() / df[f"TR14_{asset}"]
        )

        df[f"DX14_{asset}"] = 100 * (
            (df[f"plus_di_14_{asset}"] - df[f"minus_di_14_{asset}"]).abs()
            / (df[f"plus_di_14_{asset}"] + df[f"minus_di_14_{asset}"])
        )
        df[f"ADX14_{asset}"] = df[f"DX14_{asset}"].rolling(window=14).mean()

        # Drop intermediate columns for this asset
        df.drop(
            [
                f"high-low_{asset}",
                f"high-close_{asset}",
                f"low-close_{asset}",
                f"TR_{asset}",
                f"plus_dm_{asset}",
                f"minus_dm_{asset}",
                f"TR14_{asset}",
                f"DX14_{asset}",
                f"SMA20_{asset}",
                f"SMA50_{asset}",
                f"SMA100_{asset}",
            ],
            axis=1,
            inplace=True,
        )

    # Drop rows that contain NaN due to rolling calculations
    df = df.dropna()

    return df

    # Normalize the dataframes


def normalize_dataframes(
    data,
    ohlc_columns=["open", "high", "low", "close"],
    volume_column="volume",
    indicator_columns=[
        "EMA20",
        "EMA50",
        "EMA100",
        "BB_up_20",
        "BB_low_20",
        # "BB_up_50",
        # "BB_low_50",
        "ATR14",
        # "ATR50",
        "RSI14",
        # "RSI30",
        "MACD",
        "Signal",
        "plus_di_14",
        "minus_di_14",
        "ADX14",
        # "plus_di_30",
        # "minus_di_30",
        # "ADX30",
    ],
    epsilon=0.0001,  # Small constant to avoid zero in normalized data
):
    """
    Normalize the features of financial dataframes.

    :param data: A dictionary of pandas dataframes keyed by timeframe.
    :param ohlc_columns: List of columns to be normalized across all dataframes together.
    :param volume_column: The volume column to be normalized independently for each dataframe.
    :param indicator_columns: List of other indicator columns to normalize independently for each dataframe.
    :param epsilon: Small constant to set the lower bound of the normalized range.
    :return: The dictionary of normalized dataframes and the OHLC scaler used.
    """
    # Initialize the scalers
    ohlc_scaler = MinMaxScaler(
        feature_range=(epsilon, 1)
    )  # Set feature range with epsilon
    volume_scaler = MinMaxScaler(feature_range=(epsilon, 1))

    # Create a new dictionary to store the normalized dataframes
    normalized_data = {}

    # Normalize OHLC data across all timeframes together
    combined_ohlc = pd.concat([df[ohlc_columns] for df in data.values()], axis=0)
    scaled_ohlc = ohlc_scaler.fit_transform(combined_ohlc).astype(np.float32)

    # Distribute the normalized OHLC values back to the original dataframes
    start_idx = 0
    for tf, df in data.items():
        end_idx = start_idx + len(df)
        # Create a copy of the original dataframe to avoid modifying it
        normalized_df = df.copy()
        normalized_df[ohlc_columns] = scaled_ohlc[start_idx:end_idx]
        # Store the normalized dataframe in the new dictionary
        normalized_data[tf] = normalized_df
        start_idx = end_idx

    # Normalize volume independently for each timeframe
    for tf, df in normalized_data.items():
        volume_scaler = MinMaxScaler(
            feature_range=(epsilon, 1)
        )  # Reinitialize scaler for each dataframe
        df[volume_column] = volume_scaler.fit_transform(df[[volume_column]])

    # Normalize other indicators independently for each indicator within each timeframe
    for tf, df in normalized_data.items():
        for col in indicator_columns:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(epsilon, 1))
                df[[col]] = scaler.fit_transform(df[[col]])

    return normalized_data, ohlc_scaler


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


def resample_to_frequency(df, freq):
    # Resample the dataframe to the specified frequency using forward-fill to handle NaNs
    return df.resample(freq).ffill()


# Create sequences and split them for the LSTM
def create_and_split_sequences(
    data_dict, input_length, validation_pct, test_pct, base_freq="1H"
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
    base_freq="1H",
    num_action_state_features=22,  # Number of action-state features to include
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

    # Add placeholders for action-state features to each sequence
    concatenated_data = pd.concat(
        [
            concatenated_data,
            pd.DataFrame(
                np.zeros((len(concatenated_data), num_action_state_features)),
                columns=[f"action_state_{i}" for i in range(num_action_state_features)],
            ),
        ],
        axis=1,
    )

    # Create sequences
    num_sequences = len(concatenated_data) - input_length + 1
    X = np.zeros(
        (num_sequences, input_length, concatenated_data.shape[1]), dtype=np.float32
    )

    print(f"Creating {num_sequences} sequences...")

    for i in tqdm(range(num_sequences), desc="Sequencing Data", unit="seq"):
        X[i] = concatenated_data.iloc[i : (i + input_length)].values

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
        max_episodes=env_config.get("max_episodes", False),
        full_episode=env_config.get("full_episode", True),
        episode_length=env_config.get("episode_length", 120),
        max_risk=env_config.get("max_risk", 0.02),
        min_risk=env_config.get("min_risk", 0.001),
        min_profit=env_config.get("min_profit", 0),
        seed=env_config.get("seed", 42),
        limit_bounds=env_config.get("limit_bounds", False),
    )


if __name__ == "__main__":

    # # Get the total system memory
    # # total_memory = psutil.virtual_memory().total

    # # Calculate 50% of total system memory
    # # memory_to_allocate = total_memory * 0.5

    # from_time = "2019-11-01"
    # to_time = "2024-09-01"
    # symbol = "BTCUSDT"

    # # Define timeframes
    # timeframes = ["1h"]
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

    # # for tf in timeframes:
    # #     dataframes[tf] = get_timeframe_data(symbol, from_time, to_time, tf)

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
    input_length = 100  # Define the length of the input window
    # validation_pct = 0  # 0% validation set
    # test_pct = 0.1  # 10% test set

    # train_X, val_X, test_X = create_and_split_sequences_static(
    #     dataframes, input_length, validation_pct, test_pct
    # )

    # print("NUM OBSERVATIONS : ", len(train_X))

    # # train_torch_ds, val_torch_ds, test_torch_ds = convert_to_torch_datasets(
    # #     train_X, val_X, test_X, batch_size=batch_size
    # # )

    # Register the environment in gymnasium
    register(
        id="trade_env_ray_portfolio",
        entry_point="test_trade_env_ray_portfolio:TradingEnvironment",
    )

    # # Save the dataset to a file
    # np.save("train_portfolio_data_40_1h.npy", train_X)
    # np.save("val_portfolio_data_40_1h.npy", val_X)
    # np.save("test_portfolio_data_40_1h.npy", test_X)

    # Define the environment creator function
    def env_creator(env_config):
        return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trade_env_ray_portfolio", env_creator)

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=1 * (1024**3))

    # print("Converting numpy dataset to ray object....")
    # train_X_ds = from_numpy(train_X)
    # val_X_ds = from_numpy(val_X)
    # test_X_ds = from_numpy(test_X)
    # print("Convertion complete. ")

    # del dataframes
    # del train_X
    # del val_X
    # del test_X
    # gc.collect()

    # print("Saving the ray object datasets....")
    # train_X_ds.write_parquet("train_portfolio_dataset_40_1h")
    # val_X_ds.write_parquet("val_portfolio_dataset_40_1h")
    # test_X_ds.write_parquet("test_portfolio_dataset_40_1h")
    # print("Ray datasets saved. ")

    # del train_X_ds
    # del val_X_ds
    # del test_X_ds
    # gc.collect()

    # # Define the search space
    # search_space = {
    #     "lr": tune.loguniform(1e-4, 1e-1),  # Learning rate
    #     "train_batch_size": tune.choice([1024, 2048]),
    #     "sgd_minibatch_size": tune.choice([256, 512]),
    #     "num_sgd_iter": tune.choice([20, 30, 40]),
    #     "gamma": tune.quniform(0.95, 0.99, 0.01),  # Range for gamma
    #     "model": {
    #         "lstm_cell_size": tune.choice([8, 16, 32]),
    #         "fcnet_hiddens": tune.choice([[8], [16], [32]]),
    #     },
    # }
    # # Scheduler to prune less promising trials
    # scheduler = HyperBandScheduler(
    #     time_attr="training_iteration",
    #     max_t=10,  # maximum iterations per configuration
    #     reduction_factor=3,
    #     stop_last_trials=True,
    # )

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
    # Verify configuration
    # print(config.to_dict())  # This will print the current configuration as a dictionary

    # results = tune.run(
    #     "PPO",
    #     config=config,
    #     metric="episode_reward_mean",
    #     mode="max",
    #     stop={"training_iteration": 2500},
    #     # num_samples=1,  # Number of different sets of hyperparameters to try
    #     search_alg=basic_variant.BasicVariantGenerator(),  # Simple random search
    #     # scheduler=scheduler,
    #     verbose=1,
    #     checkpoint_freq=10,  # Save a checkpoint every 10 training iterations
    #     checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end of training
    #     # local_dir=r"C:\Users\marko\ray_results\1h_672_compounded_bonus",
    #     # restore=checkpoint_path,
    # )

    # # Access the best trial's results and checkpoints
    # best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print(
    #     "Best trial final reward: {}".format(
    #         best_trial.last_result["episode_reward_mean"]
    #     )
    # )

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
        f"C:/Users/marko/ray_results/Full_episode_LowLambda_stage1_conditioned/saved_states/env_state_{57}.json"
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

        # print("STATE TENSOR : ", state_tensor)
        # print("STATE : ", state)
        # print("ACTIONS TAKEN : ", action)

        # if step_count <= max_steps:
        #     print(
        #         f"Step: {test_env.current_step}, Action: {action}, LSTM State: {lstm_state}"
        #     )

        state, reward, terminated, truncated, info = test_env.step(action)
        cumulative_reward += reward

        # if step_count <= max_steps:
        #     print(
        #         f"Reward: {reward}, Cumulative Reward: {cumulative_reward}, Terminated: {terminated}"
        #     )

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


import os
import glob
import json
import time
from pathlib import Path
import argparse

import gymnasium as gym
import pandas as pd
import torch
from ray import tune

# ---------------------------------------------------------------------
# ---------------------- USER CONFIGURABLE PART -----------------------
# ---------------------------------------------------------------------
CHECKPOINTS_ROOT = Path(
    r"C:\Users\marko\ray_results\Full_episode_LowLambda_NoConstraint_stage4"
    r"\PPO_trade_env_ray_portfolio_2a377_00000_0_2025-06-03_13-30-34"
)
ENV_STATE_DIR = Path(
    r"C:\Users\marko\ray_results\Full_episode_LowLambda_NoConstraint_stage4\saved_states"
)  # folder with env_state_*.json
INPUT_LENGTH = 100
TEST_MODE = "all"  # "single"  or  "all"
CHECKPOINT_PATH = CHECKPOINTS_ROOT / "checkpoint_000000"  # used only if mode=="single"
OUTPUT_CSV = (
    "checkpoint_metrics_lev5_full_invest_GMV_EWMA_close_NoConstraint_Stage4.csv"
)


# ---------------------------------------------------------------------
# ------------------------- HELPER FUNCTIONS --------------------------
# ---------------------------------------------------------------------
def list_checkpoints(root: Path):
    """Return a list of (iteration_number:int, checkpoint_path:Path), sorted ascending."""
    cp_dirs = glob.glob(
        str(root / "checkpoint_*")
    )  # glob returns arbitrary order :contentReference[oaicite:3]{index=3}

    def _iter_num(p):
        return int(Path(p).name.split("_")[-1])

    return sorted(
        [(_iter_num(p), Path(p)) for p in cp_dirs], key=lambda t: t[0]
    )  # numeric sort :contentReference[oaicite:4]{index=4}


def run_episode(trainer, env, lstm_state):
    """Roll through one full episode; return cumulative reward and episode_metrics dict."""
    state, _ = env.reset()
    terminated = False
    cumulative_reward = 0.0
    step_count = 0
    while not terminated:
        if state.shape != (INPUT_LENGTH, state.shape[1]):
            raise ValueError(f"Unexpected state shape: {state.shape}")
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, lstm_state, _ = trainer.compute_single_action(
            state_tensor, state=lstm_state, explore=False
        )
        state, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1
    # Collect whatever metrics object your env exposes
    if hasattr(env, "checkpoint_metrics"):
        metrics = env.checkpoint_metrics.copy()
    else:
        metrics = {}
    metrics["cumulative_reward"] = cumulative_reward
    metrics["num_steps"] = step_count
    return metrics


# ---------------------------------------------------------------------
# ------------------------------ MAIN ---------------------------------
# ---------------------------------------------------------------------
def main(args):
    if args.mode not in ("single", "all"):
        raise ValueError("mode must be 'single' or 'all'")

    # Build a fresh trainer (same config as during training)
    trainer = config.build()  # your existing Trainer builder

    # Prepare env (we reuse the same one for all checkpoints)
    test_env = gym.make(
        "trade_env_ray_portfolio", mode="val", input_length=INPUT_LENGTH, seed=42
    )
    observation_space = test_env.observation_space  # noqa: F841
    action_space = test_env.action_space  # noqa: F841

    # Container for per-checkpoint rows
    rows = []

    # Decide which checkpoints to evaluate
    if args.mode == "single":
        eval_list = [(int(Path(args.ckpt).name.split("_")[-1]), Path(args.ckpt))]
    else:  # mode == "all"
        eval_list = list_checkpoints(CHECKPOINTS_ROOT)

    print(f"Found {len(eval_list)} checkpoint(s) to evaluate.")

    for iter_num, cp_path in eval_list:
        print(f"\n--- Evaluating checkpoint {cp_path.name} (iter {iter_num}) ---")

        # 1) Restore algorithm
        trainer.restore(
            str(cp_path)
        )  # restore API :contentReference[oaicite:5]{index=5}
        lstm_state = trainer.get_policy().get_initial_state()

        # 2) Load the matching env_state JSON
        env_state_idx = (iter_num + 1) * 3
        env_state_file = ENV_STATE_DIR / f"env_state_{env_state_idx}.json"
        if not env_state_file.exists():
            print(
                f"  !!! env_state file {env_state_file} not found; skipping checkpoint."
            )
            continue
        with open(env_state_file) as f:
            saved_state = json.load(f)
        test_env.set_state(saved_state)

        # 3) Run the episode & measure time
        t0 = time.time()
        metrics = run_episode(trainer, test_env, lstm_state)
        metrics["checkpoint"] = cp_path.name
        metrics["iter_num"] = iter_num
        metrics["elapsed_sec"] = time.time() - t0
        rows.append(metrics)

        print(
            f"  cumulative_reward={metrics['cumulative_reward']:.4f}  "
            f"elapsed={metrics['elapsed_sec']:.2f}s"
        )

        # # 4) Optional render / plotting (keep your original calls)
        # test_env.render()  # comment out if you don't want GUI per checkpoint
        # test_env.plot_alignment()

        if args.mode == "single":
            break

        # -----------------------------------------------------------------
        # Save results to CSV
        if args.mode == "all" and rows:
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nSaved per-checkpoint metrics to {OUTPUT_CSV}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["single", "all"], default=TEST_MODE)
#     parser.add_argument(
#         "--ckpt",
#         type=str,
#         default=str(CHECKPOINT_PATH),
#         help="Full path to a single checkpoint dir (only for mode=single)",
#     )
#     args = parser.parse_args()
#     main(args)

#     import pandas as pd
#     import matplotlib.pyplot as plt

#     # Path to the metrics file uploaded by the user
#     csv_path = (
#         "checkpoint_metrics_lev5_full_invest_GMV_EWMA_close_NoConstraint_Stage4.csv"
#     )

#     # Load
#     df = pd.read_csv(csv_path)

#     # Detect the iteration column name
#     if "iter_num" in df.columns:
#         x_col = "iter_num"
#     elif "iteration" in df.columns:
#         x_col = "iteration"
#     else:
#         raise ValueError("No iteration-number column found in the CSV.")

#     # Sort by iteration just in case
#     df = df.sort_values(x_col)

#     # Identify which metric columns exist
#     possible_metrics = {
#         "cumulative_reward": "Cumulative reward",
#         "Cumulative Return": "Cumulative return",
#         "Sharpe Ratio": "Sharpe ratio",
#         "Sortino Ratio": "Sortino ratio",
#         "Annualized Return": "Annualised return",
#         "annualized_returns": "Annualised return",  # tolerate plural
#         "ann_return": "Annualised return",
#         "Annualized Volatility": "Annualised volatility",
#         "ann_volatility": "Annualised volatility",
#         "volatility_annualised": "Annualised volatility",
#     }

#     present = {
#         col: label for col, label in possible_metrics.items() if col in df.columns
#     }

#     if len(present) == 0:
#         raise ValueError("No recognised metric columns found in the CSV.")

#     # Determine number of subplots = number of present metrics
#     n = len(present)
#     fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)

#     # If only one metric, axes is not an array
#     if n == 1:
#         axes = [axes]

#     for ax, (col, label) in zip(axes, present.items()):
#         ax.plot(df[x_col], df[col], linewidth=1.4)
#         ax.set_ylabel(label)
#         ax.grid(True, linestyle="--", linewidth=0.4)

#     axes[-1].set_xlabel("Training iteration")
#     fig.suptitle("Metric evolution per checkpoint/iteration")
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()
