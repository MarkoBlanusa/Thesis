import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gc
from trade_env_ray_portfolio import TradingEnvironment
import re
from tqdm import tqdm
import ray
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.data import from_numpy
import logging

# Set up logger
logger = logging.getLogger(__name__)


from gymnasium.envs.registration import register

# initialize torch and neural networks
torch, nn = try_import_torch()


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

    return df_combined


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
        "include_transaction_fees": True,
        "fees": 0.0005,
        "turnover_limit": None,
        "net_exposure": True,
        "net_exposure_value": 1,
        "net_exposure_constraint_type": "Equality constraint",
        "leverage_limit": True,
        "leverage_limit_value": 5,
        "leverage_limit_constraint_type": "Inequality constraint",
        "min_weight_value": -5,
        "max_weight_value": 5,
    }

    lunar_list = ["ada", "bnb", "btc", "doge", "eos", "eth", "ltc", "trx", "xlm", "xrp"]

    # Build the features using the chosen frequency. Select manually which feature layer to include in order to correspond to the desired stage.
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
        "eosusdt",  # 4
        "ethusdt",  # 5
        "ltcusdt",  # 6
        "trxusdt",  # 7
        "xlmusdt",  # 8
        "xrpusdt",  # 9
    ]

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

    # 2) fit column-wise scalers only on the train slice
    scalers = fit_scalers(df_features, "2023-04-30")

    # 3) transform all three sets with the SAME scalers
    df_features_scaled = transform(df_features, scalers)

    df_features_scaled = df_features_scaled.loc["2020-01-01":"2025-05-20"]

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
        "train_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1.npy",
        train_X,
    )
    np.save(
        "val_portfolio_data_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1.npy",
        val_X,
    )
    np.save(
        "test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1.npy",
        test_X,
    )

    df_raw_ohlcv_train_np = df_raw_ohlcv_train.to_numpy(dtype=np.float32)
    df_raw_ohlcv_val_np = df_raw_ohlcv_val.to_numpy(dtype=np.float32)
    df_raw_ohlcv_test_np = df_raw_ohlcv_test.to_numpy(dtype=np.float32)

    np.save("train_raw_ohlcv_100_1d_NewVal2.npy", df_raw_ohlcv_train_np)
    np.save("val_raw_ohlcv_100_1d_NewVal2.npy", df_raw_ohlcv_val_np)
    np.save("test_raw_ohlcv_100_1d_NewVal2.npy", df_raw_ohlcv_test_np)

    # Define the environment creator function
    def env_creator(env_config):
        return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trade_env_ray_portfolio", env_creator)

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=2 * (1024**3))

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
        "train_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1"
    )
    val_X_ds.write_parquet(
        "val_portfolio_data_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1"
    )
    test_X_ds.write_parquet(
        "test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1"
    )

    df_raw_ohlcv_train_ds.write_parquet("train_raw_ohlcv_100_1d_NewVal2")
    df_raw_ohlcv_val_ds.write_parquet("val_raw_ohlcv_100_1d_NewVal2")
    df_raw_ohlcv_test_ds.write_parquet("test_raw_ohlcv_100_1d_NewVal2")
    print("Ray datasets saved. ")

    del train_X_ds
    del val_X_ds
    del test_X_ds
    gc.collect()
