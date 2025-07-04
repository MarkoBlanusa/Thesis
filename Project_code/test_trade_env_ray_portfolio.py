import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
from ray.data import read_parquet
import scipy.stats
import cvxpy as cp
import math
from collections import deque, Counter


# Global variables

min_size_usdt_assets = [
    5,  # ADAUSDT placeholder
    5,  # BNBUSDT placeholder
    100,  # BTCUSDT placeholder
    5,  # DOGEUSDT placeholder
    5,  # EOSUSDT placeholder
    20,  # ETHUSDT placeholder
    20,  # LTCUSDT placeholder
    5,  # TRXUSDT placeholder
    5,  # XLMUSDT placeholder
    5,  # XRPUSDT placeholder
]


min_trade_amount_assets = [
    1,  # ADAUSDT placeholder
    0.01,  # BNBUSDT placeholder
    0.001,  # BTCUSDT placeholder
    1,  # DOGEUSDT placeholder
    0.1,  # EOSUSDT placeholder
    0.001,  # ETHUSDT placeholder
    0.001,  # LTCUSDT placeholder
    1,  # TRXUSDT placeholder
    1,  # XLMUSDT placeholder
    0.1,  # XRPUSDT placeholder
]


max_market_amount_assets = [
    300000,  # ADAUSDT placeholder
    2000,  # BNBUSDT placeholder
    120,  # BTCUSDT placeholder
    30000000,  # DOGEUSDT placeholder
    120000,  # EOSUSDT placeholder
    2000,  # ETHUSDT placeholder
    5000,  # LTCUSDT placeholder
    5000000,  # TRXUSDT placeholder
    1000000,  # XLMUSDT placeholder
    2000000,  # XRPUSDT placeholder
]


min_price_change_usdt = [
    0.0001,  # ADAUSDT placeholder
    0.01,  # BNBUSDT placeholder
    0.1,  # BTCUSDT placeholder
    0.00001,  # DOGEUSDT placeholder
    0.001,  # EOSUSDT placeholder
    0.01,  # ETHUSDT placeholder
    0.01,  # LTCUSDT placeholder
    0.00001,  # TRXUSDT placeholder
    0.00001,  # XLMUSDT placeholder
    0.0001,  # XRPUSDT placeholder
]


def project_fully_invested_l1(v, L=5.0):
    """
    Euclidean projection of vector v onto
        { w | sum(w)=1  and  ||w||_1 <= L }.
    Solved exactly with a tiny CVXPY program.
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.sum_squares(w - v))
    constraints = [cp.sum(w) == 1, cp.norm1(w) <= L]
    cp.Problem(obj, constraints).solve(warm_start=True, solver=cp.ECOS)
    return w.value


TRAIN_BEG = "2020-04-10"
TRAIN_END = "2023-04-30"
VAL_START = "2023-05-01"  # 100-day burn-in after TRAIN_END
VAL_END = "2024-02-08"
TEST_BEG = "2024-02-09"  # 100-day burn-in after VAL_END
TEST_END = "2025-05-20"


class TradingEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mode="test",
        leverage=5,
        input_length=100,
        market_fee=0.0005,
        limit_fee=0.0002,
        liquidation_fee=0.0125,
        slippage_mean=0.000001,
        slippage_std=0.00005,
        initial_balance=1000,
        total_episodes=1,
        max_episodes=False,
        full_episode=True,
        episode_length=120,  # 24 hours of 5 minutes data
        max_risk=0.02,
        min_risk=0.001,
        min_profit=0,
        seed=42,
        limit_bounds=False,
        margin_mode: str = "cross",  # "cross" or "isolated"
        predict_leverage=False,
        ppo_mode=True,
        full_invest=True,
        render_mode=None,
    ):
        super(TradingEnvironment, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Load your dataset
        self.data = read_parquet(
            "data/test_portfolio_dataset_100_1d_normalized_states_add_macro_and_lunarcrush_close_NewVal2_17_stage1"
        )

        self.raw_df = np.load("data/test_raw_ohlcv_100_1d_NewVal2.npy").astype(
            np.float32
        )
        self._raw_np = self.raw_df
        print("RAW DATASET SHAPE : ", self._raw_np.shape)
        self._n_raw = self._raw_np.shape[1]  # should be 50
        assert (
            self._raw_np.ndim == 2 and self._raw_np.shape[1] % 5 == 0
        ), "raw OHLCV .npy must be (T, num_assets*5)"

        self.mode = mode
        if self.mode == "train":
            self.all_dates = pd.date_range(TRAIN_BEG, TRAIN_END)
        elif self.mode == "val":
            self.all_dates = pd.date_range(VAL_START, VAL_END)
        if self.mode == "test":
            self.all_dates = pd.date_range(TEST_BEG, TEST_END)

        # Number of assets in the portfolio
        # You must ensure that your dataset columns are arranged as explained:
        # For example, if you have 10 assets and each asset has 5 columns (Open,High,Low,Close,Volume),
        # your state should have these 50 columns for OHLCV (plus the additional static columns).
        self.num_assets = 10  # Adjust this number according to your actual dataset

        self.limit_bounds = limit_bounds
        self.predict_leverage = predict_leverage
        self.margin_mode = margin_mode
        self.ppo_mode = ppo_mode
        self.full_invest = full_invest

        print("LIMIT BOUNDS : ", self.limit_bounds)
        print("PREDICT LEVERAGE : ", self.predict_leverage)
        print("MARGIN MODE : ", self.margin_mode)
        print("PPO MODE : ", self.ppo_mode)

        self.input_length = input_length
        self.max_leverage = 125
        self.leverage = leverage
        self.market_fee = market_fee
        self.limit_fee = limit_fee
        self.liquidation_fee = liquidation_fee
        self.slippage_mean = slippage_mean
        self.slippage_std = slippage_std
        self.initial_balance = initial_balance
        self.full_episode = full_episode
        if self.full_episode:
            if self.mode == "train":
                self.episode_length = self.data.count()
            else:
                self.episode_length = self.data.count() - self.input_length
        else:
            self.episode_length = episode_length
        # self.total_episodes = total_episodes
        self.max_start_index = self.data.count() - self.episode_length
        if max_episodes:
            self.total_episodes = self.max_start_index
        else:
            self.total_episodes = total_episodes

        print("TOTAL EPISODE NOT SELF IN INIT : ", total_episodes)
        print("TOTAL EPISODES IN INIT : ", self.total_episodes)

        self.reward_log = 0

        self.current_episode = 0
        self.unrealized_pnl = 0
        self.desired_position_size = 0
        self.realized_pnl = 0
        self.position_value = 0
        self.compounded_returns = 1.0
        self.opening_fee = 0
        self.closing_fee = 0
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.min_profit = min_profit
        self.first_tier_size = 50000
        self.min_trade_btc_amount = 0.001
        self.max_open_orders = 200
        self.min_size_usdt = 100
        self.price_precision = 0.01
        self.max_market_btc_amount = 120
        self.max_limit_btc_amount = 1000
        self.mark_price_rate = 0.00025
        self.cap_rate = 0.05
        self.floor_rate = 0.05
        self.min_price_change_usdt = 0.1
        self.min_btc_value_usdt = 556.8
        self.refill_rate = 0.05
        self.technical_miss = 0.0002
        self.no_trade = 0.0001
        self.max_technical_miss = 0.002
        self.max_no_trade = 0.001
        self.consecutive_technical_miss = 0
        self.consecutive_no_trade = 0
        self.previous_max_dd = 0
        self.take_profit_price = 0
        self.stop_loss_price = 0
        self.entry_price = 0
        self.balance = self.initial_balance
        self.allowed_leverage = self.leverage
        self.margin_fee = 0
        self.current_ask = 0
        self.current_bid = 0
        self.mark_price = 0
        self.log_trading_returns = []
        self.profits = []
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.sortino_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = self.leverage
        self.rng = np.random.default_rng(seed)
        self.r_min_ema = -3.1  # running min of composite reward
        self.r_max_ema = 2.6  # running max
        self.norm_alpha = 0.001  # smoothing factor (~0.001 = 2000-step half life)
        self.aligned_raw = None
        self.aligned_norm = None
        self.wf_gap = 115  # days to slide the window
        self.wf_purge = 5  # 5-day “leakage” gap
        self._wf_ptr = 0  # pointer for walk‑forward
        self.max_start = self.data.count() - self.episode_length
        self.valid_starts = list(range(0, self.max_start, self.wf_gap + self.wf_purge))
        random.shuffle(self.valid_starts)  # shuffle once per run or per epoch
        # --- initialisation reward normalization -------------------------------------------------
        self.r_count = 1e-4  # avoids div-by-zero on first update
        self.r_mean = 0.0
        self.r_var = 1.0  # unbiased sample variance * n
        self.tanh_k = 2.0  # slope inside tanh, tweak 1-3
        # --- Mean–Vol–CVaR reward parameters ---------------------
        self.beta = 0.98  # EWMA decay for volatility
        self.alpha_tail = 0.05  # tail quantile level (5%)
        self.eta_q = 0.01  # quantile‐tracker step size
        self.lambda_vol = 0.1  # weight on σ penalty
        self.tau_tail = 10.0  # weight on CVaR penalty
        # initialize online trackers
        self.sigma2 = 0.0  # EWMA variance state
        self.q_alpha = 0.0  # running VaR estimate
        # clipping threshold for normalized reward
        self.clip_reward = 10.0
        # ---------- NEW STATE FEATURES ----------------------------------------
        self.rolling_pnl = deque(maxlen=14)  # last 14 daily PnL’s
        self.rolling_pnl_sigma = 0.0  # realised σ of those PnL’s
        self.age_of_pos = 0  # steps since any position opened
        self.sin_dow = 0.0  # sin(day-of-week)
        self.cos_dow = 0.0  # cos(day-of-week)
        self.market_regime = 0  # +1 bull / –1 bear / 0 neutral
        self.portfolio_cash_ratio = 1.0  # cash / (portfolio value)
        self.last_log_ret = 0.0
        self.mm_reqs = 0.0
        self.free_collateral = (
            self.balance + self.position_value + self.unrealized_pnl - self.mm_reqs
        )
        # ---------- ONLINE SCALING -------------------------------------------
        self._welford = {}  # name → (count, mean, M2)
        self._scale_cfg = {  # one-time declaration
            #  ––– Welford z-score group –––
            "unrealized_pnl": "z",
            "realized_pnl": "z",
            "compounded_returns": "z",
            "balance": "z",
            "position_value": "z",
            "current_position_size": "z",
            "last_log_ret": "z",
            "sortino_ratio": "z",
            "rolling_pnl_sigma": "z",
            "portfolio_cash_ratio": "z",
            "free_collateral": "z",
            "mm_reqs": "z",
            "unrealized_pnl_0": "z",
            "unrealized_pnl_1": "z",
            "unrealized_pnl_2": "z",
            "unrealized_pnl_3": "z",
            "unrealized_pnl_4": "z",
            "unrealized_pnl_5": "z",
            "unrealized_pnl_6": "z",
            "unrealized_pnl_7": "z",
            "unrealized_pnl_8": "z",
            "unrealized_pnl_9": "z",
            "margin_usage_ratio": "z",
            #  ––– min–max (fixed) –––
            "leverage": ("minmax", 1.0, 125.0),
            "age_of_pos": ("minmax", 0.0, self.episode_length),
            "liq_dist_0": ("minmax", -1.0, 1.0),
            "liq_dist_1": ("minmax", -1.0, 1.0),
            "liq_dist_2": ("minmax", -1.0, 1.0),
            "liq_dist_3": ("minmax", -1.0, 1.0),
            "liq_dist_4": ("minmax", -1.0, 1.0),
            "liq_dist_5": ("minmax", -1.0, 1.0),
            "liq_dist_6": ("minmax", -1.0, 1.0),
            "liq_dist_7": ("minmax", -1.0, 1.0),
            "liq_dist_8": ("minmax", -1.0, 1.0),
            "liq_dist_9": ("minmax", -1.0, 1.0),
            "concentration_hhi": ("minmax", 0.0, 1.0),
            #  ––– centred tanh –––
            "rel_bid": "tanh",
            "rel_ask": "tanh",
            "rel_mark": "tanh",
            "spread_pct": "tanh",  # i.e.  math.tanh(spread_pct / 0.01)
        }

        for i in range(self.num_assets):
            self._scale_cfg[f"realized_weight_{i}"] = ("minmax", -1.0, 1.0)
        self.unrealized_pnl_s = [0.0] * self.num_assets
        self.liqu_distances = [0.0] * self.num_assets

        # --- INITIALIZE EW-MINMAX PARAMETERS for selected online features ---
        # For 'portfolio_cash_ratio' (∈ [0,1])
        self.alpha_ew = 0.99
        self.pcr_min_ew = 1.0  # initialize to starting ratio =1.0
        self.pcr_max_ew = 1.0

        # Instaure a global clip value
        self.clip_val = 5

        # --- reward-shaping constants ---------------------------------
        self.survival_bonus = 0.005  # +0.005 every live step
        self.early_term_penalty_scale = -5.0  # multiplied by "time left" on blow-up

        # +++ NEW: per-asset scalar attributes expected by the scaler
        for i in range(self.num_assets):
            setattr(self, f"unrealized_pnl_{i}", 0.0)
            setattr(self, f"liq_dist_{i}", 0.0)
        self.margin_usage_ratio = 0.0
        self.concentration_hhi = 0.0

        # Initialize a positions dictionary for multiple assets
        # Each asset will have its own position data structure
        self.positions = {
            i: {
                "type": None,  # "long" or "short"
                "entry_price": 0.0,
                "size": 0.0,
                "leverage": self.leverage,
            }
            for i in range(self.num_assets)
        }

        # Define the action space for multiple assets:
        # If limit_bounds=True: each asset has (weight, stop_loss, take_profit, leverage) = 4 parameters
        # If limit_bounds=False: each asset has (weight, leverage) = 2 parameters
        # Build action_space depending on flags:
        if self.limit_bounds:
            if self.predict_leverage:
                # [w, sl, tp, lev] per asset
                dim = self.num_assets * 4
            else:
                # [w, sl, tp] per asset; leverage = fixed
                dim = self.num_assets * 3
        else:
            if self.predict_leverage:
                # [w, lev] per asset; no limits
                dim = self.num_assets * 2
            else:
                # only weights
                dim = self.num_assets
        self.action_space = spaces.Box(low=-1, high=1, shape=(dim,), dtype=np.float32)

        # Initialize metrics
        if not hasattr(self, "metrics"):
            self.metrics = {
                "returns": [],
                "num_margin_calls": [],
                "risk_taken": [],
                "sharpe_ratios": [],
                "drawdowns": [],
                "num_trades": [],
                "leverage_used": [],
                "final_balance": [],
                "compounded_returns": [],
                "log_returns": [],
                "profit_risk_ratio": [],
                "sharpe_ratio": [],
                "rewards": [],
            }

        # Initialize metrics
        if not hasattr(self, "checkpoint_metrics"):
            self.checkpoint_metrics = {
                "Cumulative Return": np.nan,
                "Annualized Return": np.nan,
                "Annualized Volatility": np.nan,
                "Annualized Downside Semi-Std": np.nan,
                "Annualized Upside Semi-Std": np.nan,
                "Sharpe Ratio": np.nan,
                "Sortino Ratio": np.nan,
                "Maximum Drawdown": np.nan,
                "Lowest Point (% of Initial Capital)": np.nan,
                "VaR (99%)": np.nan,
                "CVaR (99%)": np.nan,
                "Calmar Ratio": np.nan,
                "Upside Potential Ratio": np.nan,
                "Omega Ratio": np.nan,
                "Skewness": np.nan,
                "Kurtosis": np.nan,
            }

        self.weight_dev_stats = {  # cumulative |Δw| per cause
            "Slippage": 0.0,
            "Trade_size": 0.0,
            "Leverage limit": 0.0,
            "Insufficient collateral": 0.0,
            "Margin tier": 0.0,
        }
        self.weight_dev_causes = []  # list of dicts per step  {asset_i: "cause"}
        self.weight_dev_counts = Counter()

        self.sequence_buffer = []
        self.state, info = self.reset()

        # Initialize default static values depending on limit_bounds
        # These static values are appended at the end of each observation

        # ---------- default_static_values (same order as additional_state) -----
        # NB: we insert *scaled* zeros so that the first few steps are centred
        unrealized_pnl_s0 = (
            np.clip(self._scale("unrealized_pnl", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        realized_pnl_s0 = (
            np.clip(self._scale("realized_pnl", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        comp_ret_s0 = (
            np.clip(
                self._scale("compounded_returns", 0.0), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        balance_s0 = (
            np.clip(
                self._scale("balance", self.initial_balance),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        pos_val_s0 = (
            np.clip(self._scale("position_value", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        current_ps_s0 = (
            np.clip(
                self._scale("current_position_size", 0.0), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        last_log_ret_s0 = (
            np.clip(self._scale("last_log_ret", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        max_dd_s0 = self.previous_max_dd
        rel_mark_s0 = (
            np.clip(self._scale("rel_mark", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        sortino_s0 = (
            np.clip(self._scale("sortino_ratio", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        pnl_sigma_s0 = (
            np.clip(
                self._scale("rolling_pnl_sigma", 0.0), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        age_pos_s0 = (
            np.clip(self._scale("age_of_pos", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val

        pcr_norm = (self.portfolio_cash_ratio - self.pcr_min_ew) / (
            self.pcr_max_ew - self.pcr_min_ew + 1e-8
        )
        cash_ratio_s0 = 2.0 * pcr_norm - 1.0  # → [-1, 1]

        sin_dow0 = 0.0
        cos_dow0 = 1.0
        regime_flag0 = 0.0
        spread_pct0 = (
            np.clip(self._scale("spread_pct", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        # ---------------- NEW LIQUIDATION RELATED FEATURES ---------------------------
        # ---- per-asset (10 assets × 1) -------------------
        list_liq_dist_i_s0 = []
        list_unrealized_pnl_i_s0 = []
        for i in range(self.num_assets):
            liq_dist_i0 = self.liqu_distances[i]  # pre-computed
            liq_dist_i_s0 = self._scale(f"liq_dist_{i}", liq_dist_i0)
            list_liq_dist_i_s0.append(liq_dist_i_s0)

            unrealized_pnl_i0 = self.unrealized_pnl_s[i]
            unrealized_pnl_i_s0 = self._scale(f"unrealized_pnl_{i}", unrealized_pnl_i0)
            list_unrealized_pnl_i_s0.append(unrealized_pnl_i_s0)

        if self.limit_bounds:
            self.default_static_values = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price,
                    self.stop_loss_price,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    0,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sortino_ratio,
                ]
            )
        else:
            self.default_static_values = np.array(
                [
                    unrealized_pnl_s0,
                    realized_pnl_s0,
                    comp_ret_s0,
                    balance_s0,
                    pos_val_s0,
                    current_ps_s0,
                    last_log_ret_s0,
                    max_dd_s0,
                    rel_mark_s0,
                    sortino_s0,
                    pnl_sigma_s0,
                    age_pos_s0,
                    cash_ratio_s0,
                    sin_dow0,
                    cos_dow0,
                    regime_flag0,
                    spread_pct0,
                ],
                dtype=np.float32,
            )
        num_features = self.state.shape[1]
        self.num_state_features = len(self.default_static_values)

        # Define observation space with appropriate dimensions
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.input_length, num_features),
            dtype=np.float32,
        )

    def reset(self, seed=None, **kwargs):

        super().reset(seed=seed, **kwargs)  # Call to super to handle seeding properly

        if self.full_episode:
            print(f"LEN EPISODE : {self.episode_length}")
            if self.mode == "train":
                self.start_idx = 0
            else:
                self.start_idx = self.input_length
        else:

            ##### Walk-forward with purging and worker offset and random
            if len(self.valid_starts) == 0:  # new epoch
                self.valid_starts = list(
                    range(0, self.max_start, self.wf_gap + self.wf_purge)
                )
                random.shuffle(self.valid_starts)

            base_idx = self.valid_starts.pop()  # pick a random leakage-free slice
            wid = kwargs.get("worker_idx", 0)  # each rollout-worker starts staggered
            self.start_idx = (base_idx + wid * self.wf_gap) % self.max_start

        self.current_step = self.start_idx

        # slice out exactly as many dates as history length will grow
        # history will end up length = episode_length+1
        if self.mode == "train":
            end = self.start_idx + self.episode_length + 1
        else:
            end = self.start_idx + self.episode_length
        self.current_dates = self.all_dates[self.start_idx : end]

        self.reward_log = 0

        self.unrealized_pnl = 0
        self.position_value = 0
        self.desired_position_size = 0
        self.compounded_returns = 1.0
        self.margin_call_triggered = False
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.history = []
        self.trading_returns = []  # Initialize trading returns
        self.log_trading_returns = []
        self.final_returns = []
        self.stop_loss_levels = {}
        self.take_profit_levels = {}
        self.cumulative_transaction_costs = 0
        self.previous_action = [0, 0, 0, 0]  # Initial dummy action
        self.opening_fee = 0
        self.closing_fee = 0
        self.consecutive_technical_miss = 0
        self.consecutive_no_trade = 0
        self.previous_max_dd = 0
        self.take_profit_price = 0
        self.stop_loss_price = 0
        self.entry_price = 0
        self.allowed_leverage = self.leverage
        self.margin_fee = 0
        self.current_ask = 0
        self.current_bid = 0
        self.mark_price = 0
        self.log_trading_returns = []
        self.sequence_buffer = []
        self.action_history = []
        self.profits = []
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.sortino_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = self.leverage
        self.aligned_raw = []
        self.aligned_norm = []
        self.rolling_pnl.clear()
        # --- reset risk trackers for the new episode -------------
        self.sigma2 = 0.0  # EWMA variance
        self.q_alpha = 0.0  # running VaR
        # ---------- NEW STATE FEATURES ----------------------------------------
        self.rolling_pnl = deque(maxlen=14)  # last 14 daily PnL’s
        self.rolling_pnl_sigma = 0.0  # realised σ of those PnL’s
        self.age_of_pos = 0  # steps since any position opened
        self.sin_dow = 0.0  # sin(day-of-week)
        self.cos_dow = 0.0  # cos(day-of-week)
        self.market_regime = 0  # +1 bull / –1 bear / 0 neutral
        self.portfolio_cash_ratio = 1.0  # cash / (portfolio value)
        self.last_log_ret = 0.0
        self.mm_reqs = 0.0
        self.free_collateral = (
            self.balance + self.position_value + self.unrealized_pnl - self.mm_reqs
        )
        self.unrealized_pnl_s = [0.0] * self.num_assets
        self.liqu_distances = [0.0] * self.num_assets

        # +++ NEW: per-asset scalar attributes expected by the scaler
        for i in range(self.num_assets):
            setattr(self, f"unrealized_pnl_{i}", 0.0)
            setattr(self, f"liq_dist_{i}", 0.0)
        self.margin_usage_ratio = 0.0
        self.concentration_hhi = 0.0

        # Reset positions for all assets
        self.positions = {
            i: {
                "type": None,  # "long" or "short"
                "entry_price": 0.0,
                "size": 0.0,
                "leverage": self.leverage,
            }
            for i in range(self.num_assets)
        }

        # Initialize episode-specific metrics
        self.episode_metrics = {
            "returns": [],
            "compounded_returns": [],
            "num_margin_calls": 0,
            "list_margin_calls": [],
            "risk_taken": [],
            "sharpe_ratios": [],
            "drawdowns": [],
            "num_trades": 0,
            "list_trades": [],
            "leverage_used": [],
            "stop_loss_hits": 0,
            "rewards": [],
            "sharpe_ratio": [],
            "profit_risk_ratio": [],
            "log_returns": [],
            "risk_adjusted_step_return": [],
            "rolling_sharpe": [],
            "step_risk": [],
            "weights": [],
            "true_weights": [],
            "weight_mae": [],
            "desired_leverage": [],
            "realized_leverage": [],
        }
        self.rejected_trade_count = 0  # ✓ NEW

        # Initialize metrics
        self.checkpoint_metrics = {
            "Cumulative Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Annualized Downside Semi-Std": np.nan,
            "Annualized Upside Semi-Std": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Lowest Point (% of Initial Capital)": np.nan,
            "VaR (99%)": np.nan,
            "CVaR (99%)": np.nan,
            "Calmar Ratio": np.nan,
            "Upside Potential Ratio": np.nan,
            "Omega Ratio": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
        }

        self.current_episode += 1

        # Initialize default static values depending on limit_bounds
        # These static values are appended at the end of each observation

        # ---------- default_static_values (same order as additional_state) -----
        # NB: we insert *scaled* zeros so that the first few steps are centred
        # NB: we insert *scaled* zeros so that the first few steps are centred
        unrealized_pnl_s0 = (
            np.clip(self._scale("unrealized_pnl", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        realized_pnl_s0 = (
            np.clip(self._scale("realized_pnl", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        comp_ret_s0 = (
            np.clip(
                self._scale("compounded_returns", 0.0), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        balance_s0 = (
            np.clip(
                self._scale("balance", self.initial_balance),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        pos_val_s0 = (
            np.clip(self._scale("position_value", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        current_ps_s0 = (
            np.clip(
                self._scale("current_position_size", 0.0), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        last_log_ret_s0 = (
            np.clip(self._scale("last_log_ret", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        max_dd_s0 = self.previous_max_dd
        rel_mark_s0 = (
            np.clip(self._scale("rel_mark", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        sortino_s0 = (
            np.clip(self._scale("sortino_ratio", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        pnl_sigma_s0 = (
            np.clip(
                self._scale("rolling_pnl_sigma", 0.0), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        age_pos_s0 = (
            np.clip(self._scale("age_of_pos", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val

        pcr_norm = (self.portfolio_cash_ratio - 1) / (1 - 1 + 1e-8)
        cash_ratio_s0 = 2.0 * pcr_norm - 1.0  # → [-1, 1]

        sin_dow0 = 0.0
        cos_dow0 = 1.0
        regime_flag0 = 0.0
        spread_pct0 = (
            np.clip(self._scale("spread_pct", 0.0), -self.clip_val, self.clip_val)
        ) / self.clip_val
        # ---------------- NEW LIQUIDATION RELATED FEATURES ---------------------------
        # ---- per-asset (10 assets × 1) -------------------
        list_liq_dist_i_s0 = []
        list_unrealized_pnl_i_s0 = []
        for i in range(self.num_assets):
            liq_dist_i0 = self.liqu_distances[i]  # pre-computed
            liq_dist_i_s0 = self._scale(f"liq_dist_{i}", liq_dist_i0)
            list_liq_dist_i_s0.append(liq_dist_i_s0)

            unrealized_pnl_i0 = self.unrealized_pnl_s[i]
            unrealized_pnl_i_s0 = self._scale(f"unrealized_pnl_{i}", unrealized_pnl_i0)
            list_unrealized_pnl_i_s0.append(unrealized_pnl_i_s0)

        if self.limit_bounds:
            self.default_static_values = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price,
                    self.stop_loss_price,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    0,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sortino_ratio,
                ]
            )
        else:
            self.default_static_values = np.array(
                [
                    unrealized_pnl_s0,
                    realized_pnl_s0,
                    comp_ret_s0,
                    balance_s0,
                    pos_val_s0,
                    current_ps_s0,
                    last_log_ret_s0,
                    max_dd_s0,
                    rel_mark_s0,
                    sortino_s0,
                    pnl_sigma_s0,
                    age_pos_s0,
                    cash_ratio_s0,
                    sin_dow0,
                    cos_dow0,
                    regime_flag0,
                    spread_pct0,
                ],
                dtype=np.float32,
            )

        self.num_state_features = len(self.default_static_values)

        # Prepare the windowed dataset for the episode
        self.end_idx = self.start_idx + self.episode_length

        # Split the dataset at self.start_idx
        datasets = self.data.split_at_indices([self.start_idx])
        # Use the dataset starting from self.start_idx
        sliced_data = datasets[1]

        # Initialize the iterator over the windowed dataset
        self.iterator = iter(
            sliced_data.iter_batches(
                batch_size=1,  # Each data point is a pre-sequenced array
                prefetch_batches=1,
                drop_last=False,
            )
        )

        # self.iterator = itertools.islice(self.iterator, self.start_idx, None)

        self.sequence_buffer.clear()
        self.load_initial_sequences()

        self.state = self.sequence_buffer[0]

        return self.state, {}

    # ----  required by RLlib for checkpointing -------------
    def get_state(self):
        """Return env state, including reward‐Welford and feature‐Welford."""
        # convert _welford tuples to lists so it's JSON serializable
        welford_serial = {k: list(v) for k, v in self._welford.items()}
        return {
            # reward‐normalization stats
            "r_count": self.r_count,
            "r_mean": self.r_mean,
            "r_var": self.r_var,
            "tanh_k": self.tanh_k,
            # per-feature Welford stats
            "feature_welford": welford_serial,
            # per-feature EW stats
            "pcr_min_ew": self.pcr_min_ew,
            "pcr_max_ew": self.pcr_max_ew,
        }

    def set_state(self, state):
        """Restore both reward‐Welford and feature‐Welford from a checkpoint."""
        # reward stats
        self.r_count = state["r_count"]
        self.r_mean = state["r_mean"]
        self.r_var = state["r_var"]
        self.tanh_k = state.get("tanh_k", 2.0)
        # rebuild feature Welford dict: lists → tuples
        fw = state.get("feature_welford", {})
        self._welford = {k: tuple(v) for k, v in fw.items()}
        # rebuild featur EW
        self.pcr_min_ew = state["pcr_min_ew"]
        self.pcr_max_ew = state["pcr_max_ew"]

    def load_initial_sequences(self):
        """Initialize the sequence buffer with the initial sequences for the rolling window."""

        self.sequence_buffer.clear()

        for _ in range(self.input_length):
            try:
                batch = next(self.iterator)
                initial_data = batch["data"][0].copy()
                initial_data[:, -self.num_state_features :] = self.default_static_values
                self.sequence_buffer.append(initial_data)
            except StopIteration:
                break  # In case there are fewer batches than the input_length

    def update_sequences(self, new_data):
        # new_data should contain the updated last 14 features for the most current observation
        # This method will cascade this new_data through the last 14 features of each sequence's corresponding observation
        # Handle the addition of a completely new sequence and retire the oldest sequence
        try:
            batch = next(self.iterator)
            new_sequence = batch["data"][0].copy()
            # Replace the oldest sequence with a new one if available
            self.sequence_buffer.append(new_sequence)
            self.sequence_buffer.pop(0)
            # Update the last row's last 18 features in each sequence appropriately
            for i in range(len(self.sequence_buffer)):
                # Update the (length-i-1)'th observation in the i'th sequence in the buffer
                self.sequence_buffer[i][-i - 1, -self.num_state_features :] = new_data
            next_state = self.sequence_buffer[0]
            self.state = next_state

            # Check if the episode has ended
            terminated = self.current_step > self.end_idx
            if terminated:
                self.metrics["compounded_returns"].append(
                    self.episode_metrics["compounded_returns"][-1]
                )
            return next_state, terminated, {}

        except StopIteration:

            # Even if no more sequences are left for the iterator, finish the remaining sequences inside the buffer
            if len(self.sequence_buffer) > 1:
                self.sequence_buffer.pop(0)
                for i in range(len(self.sequence_buffer)):
                    self.sequence_buffer[i][
                        -i - 1, -self.num_state_features :
                    ] = new_data
                next_state = self.sequence_buffer[0]
                self.state = next_state
                terminated = self.current_step > self.end_idx
                if terminated:
                    self.metrics["compounded_returns"].append(
                        self.episode_metrics["compounded_returns"][-1]
                    )
                return next_state, terminated, {}

            else:
                # Reset the episode if the batch buffer ends
                # self.get_aggregated_trade_info()
                # next_state, info = self.reset()
                self.metrics["compounded_returns"].append(
                    self.episode_metrics["compounded_returns"][-1]
                )
                # next_state, info = self.reset()
                next_state = self.sequence_buffer[0]
                self.state = next_state
                terminated = True
                return self.state, terminated, {}

    def get_std_dev_from_volume(
        self,
        volume,
        min_std=0.001,
        max_std=0.01,
        scaling_factor=7000,
        fallback_std=0.005,
    ):

        # Handle NaN or zero volume cases
        if np.isnan(volume) or volume == 0:
            return fallback_std

        # Calculate the inverse volume effect
        raw_std_dev = 1 / (volume / scaling_factor)

        # Normalize to a range between min_std and max_std
        normalized_std_dev = min_std + (max_std - min_std) * (
            raw_std_dev / (1 + raw_std_dev)
        )

        # If normalized std_dev is NaN or inf, fallback to fixed std_dev
        if np.isnan(normalized_std_dev) or np.isinf(normalized_std_dev):
            return fallback_std

        return normalized_std_dev

    def approximate_bid_ask(
        self,
        high_price,
        low_price,
        close_price,
        volume,
        bid_ask_std_base=0.0015,
        scaling_factor=1000,
        fallback_std_dev=0.0025,  # Fixed std_dev for fallback mechanism
    ):

        range_price = high_price - low_price

        # Check for NaN in high, low, or volume to prevent NaN results
        if (
            np.isnan(high_price)
            or np.isnan(low_price)
            or np.isnan(volume)
            or volume == 0
        ):
            # Use fallback method if inputs are invalid
            return self.fallback_approximation(close_price, fallback_std_dev)

        # Adjust std_dev based on volume
        std_dev = bid_ask_std_base / (volume / scaling_factor)

        # Check if std_dev is NaN or infinity
        if np.isnan(std_dev) or np.isinf(std_dev):
            return self.fallback_approximation(close_price, fallback_std_dev)

        bid_spread = self.rng.normal(0, std_dev) * range_price
        ask_spread = self.rng.normal(0, std_dev) * range_price

        bid_price = close_price - bid_spread
        ask_price = close_price + ask_spread

        # Check if bid_price and ask_price is NaN or infinity
        if (
            np.isnan(bid_price)
            or np.isnan(ask_price)
            or np.isinf(bid_price)
            or np.isinf(ask_price)
        ):
            return self.fallback_approximation(close_price, fallback_std_dev)

        return bid_price, ask_price

    def fallback_approximation(self, current_price, fixed_std_dev):
        """
        Fallback method to approximate bid and ask prices if NaN is encountered.
        Uses the current_price and applies fixed bid/ask spreads based on a fixed std_dev.
        """
        range_price = current_price * 0.01  # Assume a 1% range for spread approximation

        # Generate fixed spreads using a normal distribution with fixed_std_dev
        bid_spread = self.rng.normal(0, fixed_std_dev) * range_price
        ask_spread = self.rng.normal(0, fixed_std_dev) * range_price

        # Calculate fallback bid and ask prices
        bid_price = current_price - bid_spread
        ask_price = current_price + ask_spread

        return bid_price, ask_price

    # ---------- online mean / variance (Welford) --------------------------
    def _welford_update(self, name: str, x: float):
        c, m, s = self._welford.get(name, (0, 0.0, 0.0))
        c += 1
        delta = x - m
        m += delta / c
        s += delta * (x - m)
        self._welford[name] = (c, m, s)

    def _welford_update_reward(self, x: float):
        """Incremental Welford algorithm (keeps mean/var online)."""
        self.r_count += 1.0
        delta = x - self.r_mean
        self.r_mean += delta / self.r_count
        delta2 = x - self.r_mean
        self.r_var += delta * delta2  # var * n

    def _welford_norm(self, name: str, x: float, eps=1e-8) -> float:
        c, m, s = self._welford.get(name, (1, 0.0, 0.0))
        var = s / max(c - 1, 1)
        return (x - m) / math.sqrt(var + eps)

    # ---------- generic scaler -------------------------------------------
    def _scale(self, fname: str, raw: float) -> float:
        spec = self._scale_cfg.get(fname)
        if spec == "z":
            return self._welford_norm(fname, raw)
        if spec == "tanh":
            return math.tanh(raw)
        if isinstance(spec, tuple) and spec[0] == "minmax":
            _, lo, hi = spec
            return (raw - lo) / (hi - lo + 1e-8)
        return raw

    def _append_true_weights(self, bids: list, asks: list) -> None:
        """
        Compute signed realised weights after all trades & PnL updates,
        then append to episode_metrics['true_weights'].
        """
        port_val = max(self.portfolio_value, 1e-8)  # avoid /0
        realised_w = np.zeros(self.num_assets, dtype=np.float32)

        for i, pos in self.positions.items():
            if pos["type"] is None or pos["size"] == 0:
                continue  # flat asset
            # MtM value = notional + unrealised PnL already tracked
            mtm_val = pos["size"] + self.unrealized_pnl_s[i]
            sign = 1 if pos["type"] == "long" else -1
            realised_w[i] = sign * mtm_val / port_val

        self.episode_metrics["true_weights"].append(realised_w.copy())

    def step(self, action):

        # Multi-asset action extraction
        # Assuming limit_bounds = False for simplicity, and that you receive [weight1, leverage1, weight2, leverage2, ..., weightN, leverageN]
        num_assets = self.num_assets
        print("REWARD COUNT : ", self.r_count)
        print("REWARD MEAN : ", self.r_mean)
        print("REWARD VAR : ", self.r_var)
        print("TANH_K : ", self.tanh_k)
        # Parse the action depending on limit_bounds
        action = np.array(action)

        # Initialize placeholders
        stop_losses = np.zeros(num_assets, dtype=np.float32)
        take_profits = np.zeros(num_assets, dtype=np.float32)
        levs = np.full(num_assets, self.leverage, dtype=np.float32)

        if self.limit_bounds:
            if self.predict_leverage:
                act = action.reshape(num_assets, 4)
                weights, stop_losses, take_profits, levs = act.T
            else:
                act = action.reshape(num_assets, 3)
                weights, stop_losses, take_profits = act.T
                # levs remains the fixed self.leverage
        else:
            if self.predict_leverage:
                act = action.reshape(num_assets, 2)
                weights, levs = act.T
            else:
                weights = action
                # stop_losses, take_profits, leverages stay defaults

        cause_this_step = {i: "none" for i in range(self.num_assets)}

        if self.ppo_mode:  # Weights comes from a trained PPO agent
            # 1) map raw weights to signed exposures
            #    project on ∑|w| <= L_max, ∑w unconstrained here
            if self.margin_mode == "cross":
                # cross‐margin → one global leverage cap
                if self.full_invest:
                    if self.leverage == 1:
                        norm = np.sum(np.abs(weights))
                        if norm > 0:
                            weights *= self.leverage / norm
                    else:
                        weights *= self.leverage
                        weights = project_fully_invested_l1(weights, L=self.leverage)
                else:
                    # first scale to ∑|w| <= self.leverage
                    weights *= self.leverage
                    norm = np.sum(np.abs(weights))
                    if norm > 0:
                        weights *= self.leverage / norm
                levs = np.full(num_assets, self.leverage)
            else:
                # isolated‐margin
                if self.predict_leverage:
                    # explicit from agent: map [-1,1]→[1,max_leverage]
                    levs = ((levs + 1) / 2) * (self.leverage - 1) + 1
                    levs = np.ceil(levs).astype(int)
                    for i in range(num_assets):
                        weights[i] *= levs[i]
                    norm = np.sum(np.abs(weights))
                    if norm > 0:
                        for j in range(num_assets):
                            weights[j] *= levs[j] / norm
                else:
                    # implicit dynamic leverage under hard cap = ∑|w|
                    weights *= self.leverage
                    norm = np.sum(np.abs(weights))
                    if norm > 0:
                        weights *= self.leverage / norm
                    for i in range(num_assets):
                        levs[i] = max(np.abs(weights[i]), 1)
                    # levs = np.full(num_assets, np.sum(np.abs(weights)))
                    levs = np.ceil(levs).astype(int)
        else:  # Weights comes from an optimizer method
            if self.margin_mode == "cross":
                # cross‐margin → one global leverage cap
                # first scale to ∑|w| <= self.leverage
                levs = np.full(num_assets, self.leverage)
            else:
                # implicit dynamic leverage under hard cap = ∑|w|
                for i in range(num_assets):
                    levs[i] = max(np.abs(weights[i]), 1)
                # levs = np.full(num_assets, np.sum(np.abs(weights)))
                levs = np.ceil(levs).astype(int)

        print("NET EXPOSURE WEIGHTS : ", np.sum(weights))
        print("LEVERAGE WEIGHTS : ", np.sum(np.abs(weights)))

        # Collect weights into self.episode_metrics["weights"]
        self.episode_metrics["weights"].append(
            weights.copy()
        )  # Store a copy to avoid reference issues

        self.previous_action = weights
        self.action_history.append(
            {
                "Episode": self.current_episode,
                "Step": self.current_step,
                "Actions": weights.tolist(),
            }
        )

        print(f"CURRENT STEP : {self.current_step}")
        print(f"TOTAL LEVERAGE FROM THE WEIGHTS : {np.sum(np.abs(weights))}")
        print(f"TOTAL NET EXPOSURE FROM THE WEIGHTS : {np.sum(weights)}")

        allowed_leverages = [None, None, None, None, None, None, None, None, None, None]
        desired_position_sizes = []
        current_position_sizes = []

        # ------- NEW -------------------------------------------
        if self.mode == "train":
            raw_idx = self.current_step + (self.input_length - 1)
            print("WE ARE IN TRAIN MODE !")
        else:
            # for validation & test
            raw_idx = self.current_step
        print("START INDEX : ", self.start_idx)
        print("CURRENT STEP : ", self.current_step)
        print("RAW INDEX : ", raw_idx)
        row_raw = self._raw_np[raw_idx]  # 1-D vector length 50
        asset_idx = 3
        ohlcv_field = 3
        raw_val = float(self._raw_np[raw_idx, asset_idx * 5 + ohlcv_field])
        norm_val = float(self.state[-1, asset_idx * 5 + ohlcv_field])
        self.aligned_raw.append(raw_val)
        self.aligned_norm.append(norm_val)

        current_opens = []
        current_highs = []
        current_lows = []
        current_closes = []
        current_volumes = []

        for i in range(self.num_assets):
            base = i * 5
            current_opens.append(float(row_raw[base + 0]))
            current_highs.append(float(row_raw[base + 1]))
            current_lows.append(float(row_raw[base + 2]))
            current_closes.append(float(row_raw[base + 3]))
            current_volumes.append(float(row_raw[base + 4]))

        # Approximate bid/ask for each asset
        bids = []
        asks = []
        for i in range(num_assets):
            bid_i, ask_i = self.approximate_bid_ask(
                current_highs[i], current_lows[i], current_closes[i], current_volumes[i]
            )
            bids.append(bid_i)
            asks.append(ask_i)

        # Weighted by position size or notional
        total_size = sum(
            pos["size"] for pos in self.positions.values() if pos["size"] > 0
        )
        if total_size > 0:
            weighted_sum_bid = 0
            weighted_sum_ask = 0
            for i, pos in self.positions.items():
                if pos["size"] > 0:
                    # weight by pos["size"] or another factor
                    weighted_sum_bid += bids[i] * pos["size"]
                    weighted_sum_ask += asks[i] * pos["size"]

            aggregated_bid = weighted_sum_bid / total_size
            aggregated_ask = weighted_sum_ask / total_size
        else:
            aggregated_bid = np.mean(bids)  # Fallback if no positions
            aggregated_ask = np.mean(asks)

        self.current_bid = aggregated_bid  # reference
        self.current_ask = aggregated_ask

        # Compute a mark price simulation for each asset and take average
        mark_prices = []
        for i in range(num_assets):
            mid_price_i = 0.5 * (bids[i] + asks[i])
            mark_std = self.get_std_dev_from_volume(current_volumes[i])
            normal_factor = np.random.normal(0, mark_std)
            mark_price_i = mid_price_i * (1 + normal_factor)
            mark_prices.append(mark_price_i)
        self.mark_price = np.mean(mark_prices)

        # Reset realized_pnl and margin_fee for this step
        self.realized_pnl = 0
        self.margin_fee = 0

        # Compute initial unrealized PnL and position_value
        self.update_unrealized_pnl_all_assets(bids, asks)
        self.portfolio_value = self.balance + self.position_value + self.unrealized_pnl
        print("UNREALIZED PNL NEW STEP : ", self.unrealized_pnl)
        print("POSITION VALUE NEW STEP : ", self.position_value)
        print("BALANCE NEW STEP : ", self.balance)
        print("PORTFOLIO VALUE NEW STEP : ", self.portfolio_value)

        desired_position_sizes.clear()
        current_position_sizes.clear()

        # Not end of episode, proceed with actions per asset
        reward = 0
        print("REWARD FROM SURVIVAL BONUS : ", self.survival_bonus)
        reward += self.survival_bonus

        # Margin call check before acting
        print("CHECKING MARGIN CALL")
        self.check_margin_call(max(current_highs), min(current_lows))
        if self.margin_call_triggered:
            cause_this_step = ["Margin call"] * self.num_assets
            self.rejected_trade_count += self.num_assets
        if self.limit_bounds:
            self.check_limits(current_highs, current_lows)

        # Use fixed order for testing (e.g., sequential from 0 to N)
        asset_order = list(range(num_assets))

        # For each asset, apply the action
        epsilon = 1e-10
        for idx in asset_order:
            self.update_unrealized_pnl_all_assets(bids, asks)
            i = idx
            mark_price_i = mark_prices[i]
            print(f"BEGINNING THE LOOP FOR ASSET {i}")
            weight_i = weights[i]
            leverage_i = levs[i]
            print(f"WEIGHTS OF ASSET {i} : ", weight_i)
            print(f"LEVERAGE OF ASSET {i} : ", leverage_i)
            stop_loss_i = stop_losses[i]
            take_profit_i = take_profits[i]
            # current_price_state = current_closes_state[i]
            current_price = current_closes[i]
            # print(f"CURRENT PRICE FROM STATE : ", current_price_state)
            print(f"CURRENT PRICE FROM RAW DATA : ", current_price)
            pos = self.positions[i]
            print(f"POSITIONS OF ASSET {i} : ", self.positions[i])

            # --------------New Unrealized pnls and liquidation distances per assets-----------------------
            if pos["type"] is not None and pos["size"] > 0:
                if pos["type"] == "long":
                    if self.margin_mode == "isolated":
                        # per‐asset free collateral
                        eq_i = (pos["size"] / pos["leverage"]) + self.unrealized_pnl_s[
                            i
                        ]
                        _, mm_rate_i, mm_amt_i, *_ = self.get_margin_tier(
                            i, pos["size"]
                        )
                        mm_req_i = max(pos["size"] * mm_rate_i - mm_amt_i, 0.0)
                        free_coll_i = max(eq_i - mm_req_i, 0.0)
                        liq_price_i = (
                            pos["entry_price"] * (-(free_coll_i) / pos["size"])
                        ) + pos["entry_price"]
                    else:
                        # cross‐margin uses global pool
                        liq_price_i = (
                            pos["entry_price"] * (-(self.free_collateral) / pos["size"])
                        ) + pos["entry_price"]

                    upnl_i = (
                        pos["size"]
                        * ((bids[i] - pos["entry_price"]) / pos["entry_price"])
                        if pos["entry_price"] != 0
                        else 0
                    )

                else:  # short
                    if self.margin_mode == "isolated":
                        eq_i = (pos["size"] / pos["leverage"]) + self.unrealized_pnl_s[
                            i
                        ]
                        _, mm_rate_i, mm_amt_i, *_ = self.get_margin_tier(
                            i, pos["size"]
                        )
                        mm_req_i = max(pos["size"] * mm_rate_i - mm_amt_i, 0.0)
                        free_coll_i = max(eq_i - mm_req_i, 0.0)
                        liq_price_i = pos["entry_price"] - (
                            pos["entry_price"] * (-(free_coll_i) / pos["size"])
                        )
                    else:
                        liq_price_i = pos["entry_price"] - (
                            pos["entry_price"] * (-(self.free_collateral) / pos["size"])
                        )

                    upnl_i = (
                        pos["size"]
                        * ((pos["entry_price"] - asks[i]) / pos["entry_price"])
                        if pos["entry_price"] != 0
                        else 0
                    )

                self.unrealized_pnl_s[i] = upnl_i
                liq_price_i = max(liq_price_i, 0)
                liq_distance_pct_i = (
                    (mark_price_i - liq_price_i) / mark_price_i
                    if mark_price_i != 0
                    else 0
                )
                self.liqu_distances[i] = liq_distance_pct_i
            else:
                self.unrealized_pnl_s[i] = 0
                self.liqu_distances[i] = 0

            # Determine final desired positions from weights
            mm_reqs = 0.0
            for j, pos_j in self.positions.items():
                if pos_j["size"] > 0:
                    notional = pos_j["size"]
                    _, mm_rate, mm_amt, *_ = self.get_margin_tier(j, notional)
                    mm_reqs += max(notional * mm_rate - mm_amt, 0.0)

            # --- global free collateral for cross only ---
            if self.margin_mode == "cross":
                equity = self.balance + self.position_value + self.unrealized_pnl
                self.free_collateral = max(equity - mm_reqs, 0.0)

            if self.margin_mode == "isolated":
                # asset-specific equity = size / leverage + unrealized_pnl_i
                pos_i = self.positions[i]
                eq_i = (
                    self.balance
                    + (pos_i["size"] / pos_i["leverage"])
                    + self.unrealized_pnl_s[i]
                )
                # its own maintenance margin
                _, mm_rate_i, mm_amt_i, *_ = self.get_margin_tier(i, pos_i["size"])
                mm_req_i = max(pos_i["size"] * mm_rate_i - mm_amt_i, 0.0)
                free_coll_i = max(eq_i - mm_req_i, 0.0)

                print(f"MARGIN REQ {i} : {mm_req_i}")
                print(f"ASSET VALUE {i} : {eq_i}")
                print(f"FREE COLLATERAL ASSET {i} : {free_coll_i}")

                desired_position_i = weight_i * free_coll_i

            else:
                # cross-margin
                desired_position_i = weight_i * self.free_collateral

            print("BALANCE : ", self.balance)
            print("POSITION VALUE : ", self.position_value)
            print("UNREALIZED PNL : ", self.unrealized_pnl)
            print("MAINTENANCE MARGIN REQUIREMENTS : ", mm_reqs)
            print(f"DESIRED POSITION SIZE FOR ASSET {i} : ", desired_position_i)

            # Current position details
            current_size = pos["size"]
            current_position_sizes.append(current_size)
            current_direction = pos["type"]

            # difference_size determines how we adjust
            if current_direction == "long":
                effective_current_size = current_size
            elif current_direction == "short":
                effective_current_size = -current_size
            else:
                effective_current_size = 0

            difference_size = desired_position_i - effective_current_size
            print(f"DIFFERENCE SIZE OF ASSET {i} : ", difference_size)

            # Apply slippage to this asset's effective prices
            slippage = np.random.normal(self.slippage_mean, self.slippage_std)
            # If difference_size > 0: we are buying more (for long) or closing short (buy side) => use ask price with slippage
            # If difference_size < 0: we are selling (for long) or opening short => use bid price with slippage
            if difference_size > 0:
                # Buying side => use ask price adjusted by slippage
                effective_ask_price = asks[i] * (1 + slippage)
                trade_price = (
                    round(effective_ask_price / min_price_change_usdt[i])
                    * min_price_change_usdt[i]
                )
                print("current trade price : ", trade_price)
                print("current price : ", asks[i])
                print("effective current price : ", effective_ask_price)
            else:
                # Selling side => use bid price adjusted by slippage
                effective_bid_price = bids[i] * (1 - slippage)
                trade_price = (
                    round(effective_bid_price / min_price_change_usdt[i])
                    * min_price_change_usdt[i]
                )
                print("current trade price : ", trade_price)
                print("current price : ", bids[i])
                print("effective current price : ", effective_bid_price)

            # ------------------------------------------------------------
            # First handle position closing steps (no leverage/trade checks needed)
            # ------------------------------------------------------------

            # Handle partial/full closes first
            partially_closed = False
            fully_closed = False

            # Adjust positions:
            # If difference_size > 0 and we currently have a short, close it first
            if difference_size > 0 and current_direction == "short":
                closing_size = min(difference_size, abs(effective_current_size))
                if (closing_size) < abs(effective_current_size):

                    realized_pnl = closing_size * (
                        (pos["entry_price"] - trade_price) / pos["entry_price"]
                    )

                    self.opening_fee -= closing_size * self.market_fee
                    self.closing_fee = closing_size * self.market_fee
                    remaining_size = abs(effective_current_size) - closing_size
                    print(
                        f"REMAINING SIZE OF THE SHORT POSITION OF ASSET {i} : ",
                        remaining_size,
                    )

                    # Update balances
                    self.balance += (realized_pnl - 2 * self.closing_fee) + (
                        closing_size / pos["leverage"]
                    )
                    self.realized_pnl = realized_pnl - 2 * self.closing_fee
                    print(f"REALIZED PNL OF THE PARTIAL CLOSE : ", self.realized_pnl)
                    pos["size"] = remaining_size
                    pos["type"] = "short" if remaining_size > 0 else None
                    # After partial close: difference_size=0, no new position
                    difference_size = 0
                    partially_closed = True

                    # Update pnl and stop loss if needed
                    self.update_unrealized_pnl_all_assets(bids, asks)
                    if self.limit_bounds and pos["type"] is not None:
                        self.update_stop_loss_if_needed(
                            i, stop_loss_i, take_profit_i, mark_prices[i]
                        )

                    self.closing_fee = 0
                    self.consecutive_technical_miss = 0
                    self.consecutive_no_trade = 0

                else:

                    print(f"CLOSING THE SHORT POSITION FOR ASSET {i}")
                    # Close short at trade_price
                    pnl = current_size * (
                        (pos["entry_price"] - trade_price) / pos["entry_price"]
                    )
                    print(f"PNL OF ASSET {i} : ", pnl)
                    closing_fee = current_size * self.market_fee
                    self.balance += (pnl - closing_fee - self.opening_fee) + (
                        current_size / pos["leverage"]
                    )
                    print(f"BALANCE AFTER CLOSING SHORT OF ASSET {i} : ", self.balance)
                    self.realized_pnl += pnl - closing_fee - self.opening_fee
                    self.opening_fee = 0
                    pos["size"] = 0
                    pos["type"] = None
                    fully_closed = True
                    # Adjust difference_size by subtracting the closed size
                    # We used `abs(effective_current_size)` from difference_size
                    difference_size = difference_size - abs(effective_current_size)

            # If difference_size < 0 and we currently have a long, close it first
            if (
                difference_size < 0
                and current_direction == "long"
                and not partially_closed
                and not fully_closed
            ):
                closing_size = min(abs(difference_size), abs(effective_current_size))
                if (closing_size) < abs(effective_current_size):

                    realized_pnl = closing_size * (
                        (trade_price - pos["entry_price"]) / pos["entry_price"]
                    )
                    self.opening_fee -= closing_size * self.market_fee
                    self.closing_fee = closing_size * self.market_fee
                    remaining_size = abs(effective_current_size) - closing_size
                    print(
                        f"REMAINING SIZE OF THE LONG POSITION OF ASSET {i} : ",
                        remaining_size,
                    )

                    self.balance += (realized_pnl - 2 * self.closing_fee) + (
                        closing_size / pos["leverage"]
                    )
                    self.realized_pnl = realized_pnl - 2 * self.closing_fee
                    print(f"REALIZED PNL OF THE PARTIAL CLOSE : ", self.realized_pnl)
                    pos["size"] = remaining_size
                    pos["type"] = "long" if remaining_size > 0 else None
                    difference_size = 0  # after partial close no new position
                    partially_closed = True

                    self.update_unrealized_pnl_all_assets(bids, asks)
                    if self.limit_bounds and pos["type"] is not None:
                        self.update_stop_loss_if_needed(
                            i, stop_loss_i, take_profit_i, mark_prices[i]
                        )

                    self.closing_fee = 0
                    self.consecutive_technical_miss = 0
                    self.consecutive_no_trade = 0

                else:
                    print(f"CLOSING THE LONG POSITION FOR ASSET {i}")
                    # Close long at trade_price
                    pnl = current_size * (
                        (trade_price - pos["entry_price"]) / pos["entry_price"]
                    )
                    print(f"PNL OF ASSET {i} : ", pnl)
                    closing_fee = current_size * self.market_fee
                    self.balance += (pnl - closing_fee - self.opening_fee) + (
                        current_size / pos["leverage"]
                    )
                    print(f"BALANCE AFTER CLOSING LONG OF ASSET {i} : ", self.balance)
                    self.realized_pnl += pnl - closing_fee - self.opening_fee
                    self.opening_fee = 0
                    pos["size"] = 0
                    pos["type"] = None
                    fully_closed = True
                    # Adjust difference_size by subtracting the closed size
                    difference_size = difference_size + abs(
                        effective_current_size
                    )  # difference_size <0, add abs to increase difference_size

            if partially_closed or fully_closed:
                # Partial close done, no new position open
                # Move to next asset
                self.consecutive_technical_miss = 0
                self.consecutive_no_trade = 0
                continue

            if difference_size == 0 and not fully_closed:
                # No change in position size, possibly update stop-loss if needed
                if pos["type"] is not None and self.limit_bounds:
                    self.update_stop_loss_if_needed(
                        i, stop_loss_i, take_profit_i, mark_prices[i]
                    )
                if pos["type"] is None:
                    # No position & no trade => penalty no trade
                    penalty = min(
                        self.no_trade
                        + (0.5 * self.no_trade * self.consecutive_no_trade),
                        self.max_no_trade,
                    )
                    reward -= penalty
                    print("REWARD PENALTY FROM NO TRADE : ", penalty)
                    self.consecutive_no_trade += 1
                continue

            if difference_size == 0 and fully_closed:
                self.consecutive_technical_miss = 0
                continue

            # Now handle opening new position (or increasing existing same-direction position)
            # For opening new position or increasing, we do margin/trade checks
            # difference_size!=0 at this point means no partial close was done

            # Check trade size constraints
            desired_unit_size = abs(difference_size) / current_price
            asset_min_size_usdt = min_size_usdt_assets[i]
            asset_min_trade_amount = min_trade_amount_assets[
                i
            ]  # If needed, interpret as min units of asset
            asset_max_market_amount = max_market_amount_assets[i]

            # Check margin tier for allowed leverage
            allowed_leverage, mm_rate, mm_amount, _, _ = self.get_margin_tier(
                i, abs(desired_position_i)
            )
            allowed_leverages[i] = allowed_leverage

            if leverage_i > allowed_leverage:
                print("LEVERAGE OUTPUTED IS NOT ALLOWED : ", leverage_i)
                print("ALLOWED LEVERAGE : ", allowed_leverage)
                # Penalty for too high leverage
                penalty = min(
                    self.technical_miss
                    + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                    self.max_technical_miss,
                )
                reward -= penalty
                print("REWARD PENALTY FROM WRONG LEVERAGE : ", penalty)
                self.consecutive_technical_miss += 1
                self.rejected_trade_count += 1
                cause_this_step[i] = "Leverage limit"
                # Skip this asset's position adjustment
                continue

            if not (
                abs(difference_size) >= asset_min_size_usdt
                and desired_unit_size >= asset_min_trade_amount
                and desired_unit_size <= asset_max_market_amount
            ):
                print("TRADE IS NOT VALID IN SIZE. ")
                # Trade not valid in size
                penalty = min(
                    self.technical_miss
                    + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                    self.max_technical_miss,
                )
                reward -= penalty
                print("REWARD PENALTY FROM WRONG SIZE : ", penalty)
                self.consecutive_technical_miss += 1
                self.rejected_trade_count += 1
                cause_this_step[i] = "Trade size"
                # Possibly update stop loss if limit_bounds and position exists
                if self.limit_bounds and current_direction is not None:
                    self.update_stop_loss_if_needed(
                        i, stop_loss_i, take_profit_i, mark_prices[i]
                    )
                continue

            required_margin = abs(desired_position_i) / leverage_i
            # choose the same pool used for sizing
            if self.margin_mode == "isolated":
                collateral_pool = free_coll_i
            else:
                collateral_pool = self.free_collateral

            if collateral_pool < required_margin:
                print(
                    "NOT ENOUGH BALANCE FOR THE REQUIRED MARGIN OF : ",
                    required_margin,
                )
                print("BALANCE : ", self.balance)
                # Not enough balance
                penalty = min(
                    self.technical_miss
                    + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                    self.max_technical_miss,
                )
                reward -= penalty
                print("REWARD PENALTY FROM INSUFFICIENT COLLATERAL : ", penalty)
                self.consecutive_technical_miss += 1
                self.rejected_trade_count += 1
                cause_this_step[i] = "Insufficient collateral"
                continue

            # Now open/increase position in the direction of difference_size
            self.balance -= abs(difference_size) / leverage_i
            print(
                f"BALANCE AFTER OPENING POSITION (MINUS THE REQUIRED MARGIN) OF ASSET {i} : ",
                self.balance,
            )
            self.opening_fee += abs(difference_size) * self.market_fee

            new_size = abs(difference_size) + (
                pos["size"]
                if pos["type"] in ["long", "short"]
                and (
                    (pos["type"] == "long" and difference_size > 0)
                    or (pos["type"] == "short" and difference_size < 0)
                )
                else 0
            )

            print(f"NEW SIZE OF THE OPENED POSITION OF ASSET {i} : ", new_size)
            if difference_size > 0:
                # Going long
                if pos["type"] == "long":
                    # Weighted average entry
                    old_size = pos["size"]
                    new_entry_price = (
                        pos["entry_price"] * old_size
                        + trade_price * abs(difference_size)
                    ) / new_size
                    print(
                        f"NEW ENTRY PRICE OF THE INCREASED LONG POSITION OF ASSET {i} : ",
                        new_entry_price,
                    )
                    pos["entry_price"] = new_entry_price
                    pos["size"] = new_size
                    pos["leverage"] = leverage_i
                else:
                    pos["type"] = "long"
                    pos["entry_price"] = trade_price
                    print(
                        f"NEW ENTRY PRICE OF THE NEW LONG POSITION OF ASSET {i} : ",
                        trade_price,
                    )
                    pos["size"] = abs(difference_size)
                    pos["leverage"] = leverage_i

            elif difference_size < 0:
                # Going short
                if pos["type"] == "short":
                    # Weighted average entry
                    old_size = pos["size"]
                    new_entry_price = (
                        pos["entry_price"] * old_size
                        + trade_price * abs(difference_size)
                    ) / new_size
                    print(
                        f"NEW ENTRY PRICE OF THE INCREASED SHORT POSITION OF ASSET {i} : ",
                        new_entry_price,
                    )
                    pos["entry_price"] = new_entry_price
                    pos["size"] = new_size
                    pos["leverage"] = leverage_i
                else:
                    pos["type"] = "short"
                    pos["entry_price"] = trade_price
                    print(
                        f"NEW ENTRY PRICE OF THE NEW SHORT POSTION OF ASSET {i} : ",
                        trade_price,
                    )
                    pos["size"] = abs(difference_size)
                    pos["leverage"] = leverage_i

            # Update unrealized pnl after trade
            self.update_unrealized_pnl_all_assets(bids, asks)

            # If limit_bounds, set stop_loss/take_profit for this asset
            if self.limit_bounds and pos["type"] is not None:
                self.update_stop_loss_if_needed(
                    i, stop_loss_i, take_profit_i, mark_prices[i]
                )

            # Successful trade
            self.episode_metrics["num_trades"] += 1
            self.consecutive_technical_miss = 0
            self.consecutive_no_trade = 0

        # Recalculate portfolio metrics
        self.update_unrealized_pnl_all_assets(bids, asks)
        # --------------New Unrealized pnls and liquidation distances per assets-----------------------
        if pos["type"] is not None and pos["size"] > 0:
            if pos["type"] == "long":
                if self.margin_mode == "isolated":
                    # per‐asset free collateral
                    eq_i = (pos["size"] / pos["leverage"]) + self.unrealized_pnl_s[i]
                    _, mm_rate_i, mm_amt_i, *_ = self.get_margin_tier(i, pos["size"])
                    mm_req_i = max(pos["size"] * mm_rate_i - mm_amt_i, 0.0)
                    free_coll_i = max(eq_i - mm_req_i, 0.0)
                    liq_price_i = (
                        pos["entry_price"] * (-(free_coll_i) / pos["size"])
                    ) + pos["entry_price"]
                else:
                    # cross‐margin uses global pool
                    liq_price_i = (
                        pos["entry_price"] * (-(self.free_collateral) / pos["size"])
                    ) + pos["entry_price"]

                upnl_i = (
                    pos["size"] * ((bids[i] - pos["entry_price"]) / pos["entry_price"])
                    if pos["entry_price"] != 0
                    else 0
                )

            else:  # short
                if self.margin_mode == "isolated":
                    eq_i = (pos["size"] / pos["leverage"]) + self.unrealized_pnl_s[i]
                    _, mm_rate_i, mm_amt_i, *_ = self.get_margin_tier(i, pos["size"])
                    mm_req_i = max(pos["size"] * mm_rate_i - mm_amt_i, 0.0)
                    free_coll_i = max(eq_i - mm_req_i, 0.0)
                    liq_price_i = pos["entry_price"] - (
                        pos["entry_price"] * (-(free_coll_i) / pos["size"])
                    )
                else:
                    liq_price_i = pos["entry_price"] - (
                        pos["entry_price"] * (-(self.free_collateral) / pos["size"])
                    )

                upnl_i = (
                    pos["size"] * ((pos["entry_price"] - asks[i]) / pos["entry_price"])
                    if pos["entry_price"] != 0
                    else 0
                )

            self.unrealized_pnl_s[i] = upnl_i
            liq_price_i = max(liq_price_i, 0)
            liq_distance_pct_i = (
                (mark_price_i - liq_price_i) / mark_price_i if mark_price_i != 0 else 0
            )
            self.liqu_distances[i] = liq_distance_pct_i
        else:
            self.unrealized_pnl_s[i] = 0
            self.liqu_distances[i] = 0

        # ---------------------------------------------------
        for i in range(self.num_assets):
            setattr(self, f"unrealized_pnl_{i}", self.unrealized_pnl_s[i])  # ✓ NEW
            setattr(self, f"liq_dist_{i}", self.liqu_distances[i])  # ✓ NEW

        print("UNREALIZED PNL END OF STEP : ", self.unrealized_pnl)
        print("POSITION VALUE END OF STEP : ", self.position_value)
        print("BALANCE END OF STEP : ", self.balance)
        self.portfolio_value = self.balance + self.position_value + self.unrealized_pnl
        self.portfolio_value = round(self.portfolio_value, 5)
        print("PORTFOLIO VALUE END OF STEP : ", self.portfolio_value)
        self.history.append(self.portfolio_value)

        # ---------- track realised (true) weights ---------------------------
        self._append_true_weights(bids, asks)

        # ---- step-wise MAE between desired (weights) and realised --------
        realized_w = self.episode_metrics["true_weights"][-1]
        desired_w = self.episode_metrics["weights"][-1]
        # compute leverage
        desired_lev = np.sum(np.abs(desired_w))
        realized_lev = np.sum(np.abs(realized_w))
        # append
        self.episode_metrics["desired_leverage"].append(desired_lev)
        self.episode_metrics["realized_leverage"].append(realized_lev)
        print("realized_w shape : ", len(realized_w))
        print("desired_w shape : ", len(desired_w))
        mae_t = np.abs(desired_w - realized_w).sum()
        self.episode_metrics["weight_mae"].append(mae_t)
        # -------- Δw and attribution ---------------------------
        abs_delta = np.abs(desired_w - realized_w)
        for j in range(self.num_assets):
            if (
                cause_this_step[j] == "none"
            ):  # trade went through -> residual is slippage
                if abs_delta[j] > 1e-8:
                    cause_this_step[j] = "Slippage"
            self.weight_dev_stats.setdefault(cause_this_step[j], 0.0)
            self.weight_dev_stats[cause_this_step[j]] += abs_delta[j]
            # count occurrences
            if abs_delta[j] > 1e-8 or cause_this_step[j] != "none":
                self.weight_dev_counts[cause_this_step[j]] += 1

        # keep per-step logs
        self.weight_dev_causes.append(cause_this_step)

        # ---------- NEW FEATURE CALCULATIONS ----------------------------------

        # Maintenance requirements and free collateral related features
        self.mm_reqs = mm_reqs

        self.margin_usage_ratio = self.mm_reqs / max(self.portfolio_value, 1e-8)
        self.concentration_hhi = np.square(self.previous_action).sum()

        # 2. age-of-position
        self.age_of_pos = 0 if self.current_position_size == 0 else self.age_of_pos + 1

        # 3. sin/cos day-of-week (no leakage)
        dow = (raw_idx // 1) % 7  # raw_idx maps 1-to-1 to calendar days in 1-D freq
        self.sin_dow = math.sin(2 * math.pi * dow / 7)
        self.cos_dow = math.cos(2 * math.pi * dow / 7)

        # 4. market-regime flag (slope of 200-SMA on BTC close)
        if raw_idx >= 210:
            sma_now = np.mean(self._raw_np[raw_idx - 199 : raw_idx + 1, 3])  # BTC close
            sma_old = np.mean(self._raw_np[raw_idx - 209 : raw_idx - 9, 3])
            self.market_regime = (
                1 if sma_now > sma_old else -1 if sma_now < sma_old else 0
            )
        else:
            self.market_regime = 0

        # 5. cash / equity ratio
        eq = max(self.portfolio_value, 1e-8)
        self.portfolio_cash_ratio = self.balance / eq

        # 6. Construct additional_state arrays
        last_log_ret = (
            self.log_trading_returns[-1] if len(self.log_trading_returns) > 0 else 0
        )
        self.last_log_ret = last_log_ret
        # --- new instantaneous features ------------------------------
        mid_price = 0.5 * (self.current_bid + self.current_ask)
        spread_pct = (self.current_ask - self.current_bid) / max(mid_price, 1e-8)
        spread_pct_s = (
            np.clip(
                self._scale("spread_pct", spread_pct), -self.clip_val, self.clip_val
            )
        ) / self.clip_val
        # Update compounded returns
        self.compounded_returns *= 1 + last_log_ret

        # 1. rolling 14-day σ of portfolio PnL
        self.rolling_pnl.append(last_log_ret)
        if len(self.rolling_pnl) == 14:
            pnl_arr = np.array(self.rolling_pnl, dtype=np.float32)
            self.rolling_pnl_sigma = float(pnl_arr.std(ddof=1))

        # 30‑day rolling window
        window = np.array(self.trading_returns[-30:])
        if len(window) < 30:  # need at least a week
            sortino = 0
        else:
            rf_daily = 0.005 / 252
            excess = window.mean() - rf_daily
            down_dev = np.sqrt(np.mean(np.square(np.minimum(window - rf_daily, 0))))
            sortino = excess / down_dev if down_dev else 0
        self.sortino_ratio = sortino

        # Log metrics for the current step
        self.log_step_metrics()

        # Trading returns
        if len(self.history) > 1:
            if not self.margin_call_triggered:
                previous_val = self.history[-2]
                current_val = self.history[-1]
                trading_return = (
                    0
                    if previous_val == 0
                    else (current_val - previous_val) / previous_val
                )
                self.trading_returns.append(trading_return)
                self.episode_metrics["returns"].append(trading_return)
                log_trading_return = (
                    0
                    if (current_val <= 0 or previous_val <= 0)
                    else np.log(current_val / previous_val)
                )
                self.log_trading_returns.append(log_trading_return)
                final_return = (current_val - self.history[0]) / self.history[0]
                self.final_returns.append(final_return)
                max_dd = calculate_max_drawdown(self.history)
                reward += self.calculate_reward()
                self.episode_metrics["compounded_returns"].append(
                    self.compounded_returns - 1
                )
            else:
                # Margin call triggered this step
                max_dd = calculate_max_drawdown(self.history)
                reward += self.calculate_reward()
                self.episode_metrics["compounded_returns"].append(
                    self.compounded_returns - 1
                )
                self.margin_call_triggered = False
        else:
            reward += 0
            max_dd = 0

        self.previous_max_dd = max_dd

        self.episode_metrics["rewards"].append(reward)
        self.metrics["rewards"].append(reward)

        total_size = sum(
            pos["size"] for pos in self.positions.values() if pos["size"] > 0
        )
        if total_size > 0:

            allowed_leverage_count = 0
            sum_allowed_leverage = 0
            for pos in self.positions.values():
                if allowed_leverages[allowed_leverage_count] is not None:
                    sum_allowed_leverage += (
                        allowed_leverages[allowed_leverage_count] * pos["size"]
                    )
                    allowed_leverage_count += 1
            weighted_entry_price = (
                sum(
                    pos["entry_price"] * pos["size"]
                    for pos in self.positions.values()
                    if pos["size"] > 0
                )
                / total_size
            )
            self.entry_price = weighted_entry_price
        else:
            self.leverage = self.leverage
            self.allowed_leverage = self.leverage
            self.entry_price = 0

        self.desired_position_size = np.sum(desired_position_sizes)
        self.current_position_size = np.sum(current_position_sizes)

        # ---------- ONLINE SCALER UPDATES -------------------------------------
        for fname, flag in self._scale_cfg.items():
            if flag == "z":
                # every feature lives as an attribute → access via getattr
                self._welford_update(fname, float(getattr(self, fname)))

        # ---------- SCALED FEATURE PACK --------------------------------------
        unrealized_pnl_s = (
            np.clip(
                self._scale("unrealized_pnl", self.unrealized_pnl),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        realized_pnl_s = (
            np.clip(
                self._scale("realized_pnl", self.realized_pnl),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        comp_ret_s = (
            np.clip(
                self._scale("compounded_returns", self.compounded_returns - 1),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        balance_s = (
            np.clip(self._scale("balance", self.balance), -self.clip_val, self.clip_val)
        ) / self.clip_val
        pos_val_s = (
            np.clip(
                self._scale("position_value", self.position_value),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        current_ps_s = (
            np.clip(
                self._scale("current_position_size", self.current_position_size),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        last_log_ret_s = (
            np.clip(
                self._scale("last_log_ret", self.last_log_ret),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        max_dd_s = self.previous_max_dd
        rel_mark_s = (
            np.clip(
                self._scale(
                    "rel_mark",
                    (self.mark_price - self.entry_price) / (self.entry_price + 1e-8),
                ),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        sortino_s = (
            np.clip(
                self._scale("sortino_ratio", self.sortino_ratio),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        pnl_sigma_s = (
            np.clip(
                self._scale("rolling_pnl_sigma", self.rolling_pnl_sigma),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val
        age_pos_s = (
            np.clip(
                self._scale("age_of_pos", self.age_of_pos),
                -self.clip_val,
                self.clip_val,
            )
        ) / self.clip_val

        self.pcr_min_ew = min(
            self.portfolio_cash_ratio,
            self.alpha_ew * self.pcr_min_ew
            + (1 - self.alpha_ew) * self.portfolio_cash_ratio,
        )
        self.pcr_max_ew = max(
            self.portfolio_cash_ratio,
            self.alpha_ew * self.pcr_max_ew
            + (1 - self.alpha_ew) * self.portfolio_cash_ratio,
        )
        pcr_norm = (self.portfolio_cash_ratio - self.pcr_min_ew) / (
            self.pcr_max_ew - self.pcr_min_ew + 1e-8
        )
        cash_ratio_s = 2.0 * pcr_norm - 1.0  # → [-1, 1]

        sin_dow = self.sin_dow
        cos_dow = self.cos_dow
        regime_flag = self.market_regime

        # ---------------- NEW LIQUIDATION RELATED FEATURES ---------------------------
        # ---- per-asset (10 assets × 1) -------------------
        list_liq_dist_i_s = []
        list_unrealized_pnl_i_s = []
        for i in range(self.num_assets):
            liq_dist_i = self.liqu_distances[i]  # pre-computed
            liq_dist_i_s = self._scale(f"liq_dist_{i}", liq_dist_i)
            list_liq_dist_i_s.append(liq_dist_i_s)

            unrealized_pnl_i = self.unrealized_pnl_s[i]
            unrealized_pnl_i_s = self._scale(f"unrealized_pnl_{i}", unrealized_pnl_i)
            list_unrealized_pnl_i_s.append(unrealized_pnl_i_s)

        if self.limit_bounds:
            additional_state = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price if hasattr(self, "take_profit_price") else 0,
                    self.stop_loss_price if hasattr(self, "stop_loss_price") else 0,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    last_log_ret,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sortino_ratio,
                ]
            )
        else:
            additional_state = np.array(
                [
                    # ----- original ----------
                    unrealized_pnl_s,
                    realized_pnl_s,
                    comp_ret_s,
                    balance_s,
                    pos_val_s,
                    current_ps_s,
                    last_log_ret_s,
                    max_dd_s,
                    rel_mark_s,
                    sortino_s,
                    # ----- NEW ---------------
                    pnl_sigma_s,
                    age_pos_s,
                    cash_ratio_s,
                    sin_dow,
                    cos_dow,
                    regime_flag,
                    spread_pct_s,
                ],
                dtype=np.float32,
            )

        self.current_step += 1
        next_state, terminated, info = self.update_sequences(additional_state)
        self.state = next_state

        if self.portfolio_value < 0.1 * self.initial_balance:
            print("EPISODE TERMINATED EARLY DUE TO LOW PORTFOLIO VALUE. ")
            remain = 1.0 - (self.current_step - self.start_idx) / self.episode_length
            reward += self.early_term_penalty_scale * remain  # big negative
            print(
                "REWARD PENALTY FROM EARLY EPISODE STOPPING : ",
                self.early_term_penalty_scale * remain,
            )
            # terminated = True

        if abs(reward) > 10000000:
            terminated = True

        if terminated and abs(reward) <= 10000000:
            print("EPISODE TERMINATED")
            print("CURRENT STEP : ", self.current_step)
            print("START INDEX : ", self.start_idx)
            print("END INDEX : ", self.end_idx)
            print("TOTAL EPISODES : ", self.total_episodes)
            print("CURRENT EPISODE : ", self.current_episode)
            # next_state, info = self.data[self.current_step - 1], {}
            # self.state = next_state
            self.log_episode_metrics()

            if self.current_episode < self.total_episodes:
                terminated = False
                self.reset()
            else:
                print("\nFINAL PERFORMANCE METRICS OVER ALL EPISODES:")
                for metric in self.metrics:
                    if metric not in ["rewards", "returns"]:
                        value = np.mean(self.metrics[metric])
                        print(f"{metric}: {value:.4f}")
                # print("EPISODE METRICS : ", episode_metrics)

                # print("SELF METRICS : ", self.metrics)

                avg_compounded_returns = np.mean(self.metrics["compounded_returns"])
                print(
                    f"AVERAGE COMPOUNDED RETURNS OVER {self.total_episodes} EPISODES : {avg_compounded_returns}"
                )
                total_rewards = np.sum(self.metrics["rewards"])
                mean_rewards = np.mean(self.metrics["rewards"])
                std_rewards = np.std(self.metrics["rewards"])
                skew_rewards = calculate_skewness(self.metrics["rewards"])
                kurt_rewards = calculate_kurtosis(self.metrics["rewards"])
                print(f"STATS ON REWARDS OVER {self.total_episodes} EPISODES : ")
                print()
                print("TOTAL REWARDS : ", total_rewards)
                print("MEAN REWARDS : ", mean_rewards)
                print("STD REWARDS : ", std_rewards)
                print("SKEWNESS REWARDS : ", skew_rewards)
                print("KURTOSIS REWARDS : ", kurt_rewards)
                print()
                print()
                if len(self.metrics["returns"]) != 0:
                    total_returns = np.sum(self.metrics["returns"])
                    mean_returns = np.mean(self.metrics["returns"])
                    std_returns = np.std(self.metrics["returns"])
                    skew_returns = calculate_skewness(self.metrics["returns"])
                    kurt_returns = calculate_kurtosis(self.metrics["returns"])
                    print(f"STATS ON RETURNS OVER {self.total_episodes} EPISODES : ")
                    print()
                    print("TOTAL RETURNS : ", total_returns)
                    print("MEAN RETURNS : ", mean_returns)
                    print("STD RETURNS : ", std_returns)
                    print("SKEWNESS RETURNS : ", skew_returns)
                    print("KURTOSIS RETURNS : ", kurt_returns)
                print(
                    "METRICS FINAL COMPOUNDED RETURN : ",
                    self.metrics["compounded_returns"],
                )

                # Convert history to cumulative returns
                portfolio_values = pd.Series(self.history)
                portfolio_returns = portfolio_values.pct_change().fillna(0)
                portfolio_cum_returns = (1 + portfolio_returns).cumprod()

                # Compute metrics
                episode_metrics = compute_metrics(
                    portfolio_cum_returns, portfolio_returns, self.initial_balance
                )
                self.checkpoint_metrics = episode_metrics
                print(
                    "TOTAL ACCUMULATED REWARD ONLY FROM LOGRETURNS : ", self.reward_log
                )

        truncated = False

        # 4) ---------- NEW: running mean-std normalisation ----------
        #    a) Welford update
        self.r_count += 1.0
        delta = reward - self.r_mean
        self.r_mean += delta / self.r_count
        delta2 = reward - self.r_mean
        self.r_var += delta * delta2  # unbiased * n
        print("TOTAL REWARD BEFORE NORMALIZATION : ", reward)

        #    b) z-score
        z = (reward - self.r_mean) / math.sqrt(
            self.r_var / max(self.r_count - 1, 1) + 1e-8
        )

        #    c) clip ±5 σ
        z_clip = max(min(z, self.clip_val), -self.clip_val)

        #    d) rescale to [–1,1]
        r_norm = z_clip / self.clip_val

        reward = r_norm

        print("REWARD END OF STEP : ", reward)

        return next_state, reward, terminated, truncated, info

    def set_limits(
        self,
        asset_id: int,
        current_price: float,
        take_profit: float,
        position_type: str,
        adjusted_stop_loss: float,
        mark_price: float,
    ):
        """
        asset_id: which asset in self.positions
        """
        pos = self.positions[asset_id]
        if pos["type"] is None or pos["size"] <= 0:
            return

        entry_price = pos["entry_price"]

        # compute prices
        if position_type == "long":
            slp = current_price * (1 - adjusted_stop_loss)
            tpp = current_price * (
                1
                + max(
                    (2 * self.opening_fee + self.min_profit) / pos["size"], take_profit
                )
            )
        else:
            slp = current_price * (1 + adjusted_stop_loss)
            tpp = current_price * (
                1
                - max(
                    (2 * self.opening_fee + self.min_profit) / pos["size"], take_profit
                )
            )

        # round
        slp = round(slp / self.price_precision) * self.price_precision
        tpp = round(tpp / self.price_precision) * self.price_precision

        # store under (asset_id, entry_price) to avoid collisions
        self.stop_loss_levels[(asset_id, entry_price)] = (slp, position_type)
        if take_profit <= 1 - ((1 - self.cap_rate) * (mark_price / entry_price)):
            self.take_profit_levels[(asset_id, entry_price)] = (tpp, position_type)

    def check_limits(self, high_price: float, low_price: float):
        for (aid, entry), (slp, ptype) in list(self.stop_loss_levels.items()):
            # get matching take-profit if any
            tpp = self.take_profit_levels.get((aid, entry), (None,))[0]

            # decide if we hit SL or TP
            hit_sl = (ptype == "long" and low_price <= slp) or (
                ptype == "short" and high_price >= slp
            )
            hit_tp = tpp is not None and (
                (ptype == "long" and high_price >= tpp)
                or (ptype == "short" and low_price <= tpp)
            )
            if hit_sl or hit_tp:
                price = slp if hit_sl else tpp
                self.episode_metrics["stop_loss_hits"] += int(hit_sl)
                # proper execute_order signature:
                self.execute_order(
                    order_type="sell" if ptype == "long" else "buy",
                    execution_price=price,
                    entry_price=entry,
                    position_type=ptype,
                    position_size=self.positions[aid]["size"],
                    previous_leverage=self.positions[aid]["leverage"],
                    is_margin_call=False,
                )
                # cleanup
                self.stop_loss_levels.pop((aid, entry), None)
                self.take_profit_levels.pop((aid, entry), None)

    def execute_order(
        self,
        order_type: str,
        execution_price: float,
        entry_price: float,
        position_type: str,
        position_size: float,
        previous_leverage: int,
        is_margin_call: bool = False,
        asset_id: int = None,
    ):
        """
        asset_id: which asset we're liquidating
        """
        if asset_id is None:
            # try to infer from entry_price, but better to _always_ pass it
            asset_id = next(
                i for i, p in self.positions.items() if p["entry_price"] == entry_price
            )

        # PnL computation
        if not is_margin_call:
            if position_type == "long" and order_type == "sell":
                pnl = position_size * ((execution_price - entry_price) / entry_price)
            elif position_type == "short" and order_type == "buy":
                pnl = position_size * ((entry_price - execution_price) / entry_price)
            else:
                return  # unsupported
            fee = position_size * self.limit_fee
            self.realized_pnl = pnl - fee - self.opening_fee
            self.balance += self.realized_pnl + (position_size / previous_leverage)
        else:
            # same as before, but now we know asset_id
            pnl = (
                position_size * ((execution_price - entry_price) / entry_price)
                if position_type == "long"
                else position_size * ((entry_price - execution_price) / entry_price)
            )
            fee = position_size * self.limit_fee
            mfee = position_size * self.liquidation_fee
            self.realized_pnl = pnl - fee - self.opening_fee - mfee
            self.balance += max(
                self.realized_pnl + (position_size / previous_leverage), -self.balance
            )

        # reset this asset
        self.positions[asset_id] = {
            "type": None,
            "entry_price": 0.0,
            "size": 0.0,
            "leverage": self.leverage,
        }
        # reset env‐wide
        self.unrealized_pnl = 0
        self.position_value = sum(
            p["size"] / p["leverage"] for p in self.positions.values()
        )
        self.portfolio_value = self.balance + self.unrealized_pnl + self.position_value
        self.closing_fee = 0
        self.opening_fee = 0

    def calculate_liquidation_price(self):
        """
        Adapted calculation of a liquidation price for a multi-asset portfolio.
        This is a simplified approach:
        - Compute total notional value from all open positions.
        - Use that to get maintenance_margin_rate and required_margin.
        - Assume the 'worst-case scenario' is reflected by the first asset (or any chosen reference asset).
        For a real multi-asset scenario, a single liquidation price is not truly representative.
        """

        # If no positions, no margin price
        active_positions = [
            pos
            for pos in self.positions.values()
            if pos["type"] is not None and pos["size"] > 0
        ]
        if not active_positions:
            return 0

        # Compute total notional value
        total_notional = sum(pos["size"] for pos in active_positions)

        # Get margin tier from total_notional
        leverage, maintenance_margin_rate, maintenance_amount, _, _ = (
            self.get_margin_tier(total_notional)
        )
        required_margin = maintenance_margin_rate * total_notional - maintenance_amount

        # One-way mode, cross-margin logic: we need a reference asset for margin_price computation
        # Let's pick the first asset that has a position as reference:
        for i, pos in self.positions.items():
            if pos["type"] is not None and pos["size"] > 0:
                reference_asset_index = i
                reference_pos = pos
                break

        entry_price = reference_pos["entry_price"]
        position_type = reference_pos["type"]
        position_size = reference_pos["size"]

        # Calculate margin call price using reference asset logic
        # This mirrors the single asset logic but on the chosen reference position
        # margin_price = same formula as single asset:
        if position_type == "long":
            margin_price = (
                entry_price
                * (-(self.portfolio_value - required_margin) / position_size)
            ) + entry_price
        else:
            margin_price = entry_price - (
                entry_price
                * (-(self.portfolio_value - required_margin) / position_size)
            )

        margin_price = max(margin_price, 0)
        return margin_price

    def update_stop_loss_if_needed(
        self, asset_index: int, stop_loss: float, take_profit: float, mark_price: float
    ):
        """
        Enforce that the new SL/TP for asset_index lies within our
        per-asset risk bounds, then call set_limits.
        """

        if not self.limit_bounds:
            return

        pos = self.positions[asset_index]
        if pos["type"] is None or pos["size"] <= 0:
            return

        # 1) per-unit bounds
        equity = self.balance + self.position_value
        max_loss = (self.max_risk * equity - 2 * self.opening_fee) / pos["size"]
        min_loss = (self.min_risk * equity - 2 * self.opening_fee) / pos["size"]

        # 2) cap via cap_rate
        if pos["entry_price"] != 0:
            restricted = 1 - (1 - self.cap_rate) * (mark_price / pos["entry_price"])
        else:
            restricted = 0.0

        final_max = min(max_loss, restricted)
        adj_sl = max(min(stop_loss, final_max), min_loss)

        if adj_sl > 0 and adj_sl >= min_loss:
            # pass asset_index and everything
            self.set_limits(
                asset_id=asset_index,
                current_price=mark_price,  # or the current close price if you prefer
                take_profit=take_profit,
                position_type=pos["type"],
                adjusted_stop_loss=adj_sl,
                mark_price=mark_price,
            )

    def update_unrealized_pnl_all_assets(self, bids, asks):
        """
        Recalculate the unrealized PnL and position value for all assets after changes.
        This method updates self.unrealized_pnl and self.position_value based on current bids/asks.
        bids and asks are arrays of current bid/ask prices for each asset.
        """

        total_unrealized_pnl = 0
        total_position_value = 0
        for i, pos in self.positions.items():
            if pos["type"] is not None and pos["size"] > 0:
                if pos["type"] == "long":
                    upnl = (
                        pos["size"]
                        * ((bids[i] - pos["entry_price"]) / pos["entry_price"])
                        if pos["entry_price"] != 0
                        else 0
                    )

                else:  # short
                    upnl = (
                        pos["size"]
                        * ((pos["entry_price"] - asks[i]) / pos["entry_price"])
                        if pos["entry_price"] != 0
                        else 0
                    )

                pos_value = pos["size"] / pos["leverage"]
                total_unrealized_pnl += upnl
                total_position_value += pos_value
                self.unrealized_pnl_s[i] = upnl

        self.unrealized_pnl = total_unrealized_pnl
        self.position_value = total_position_value

    def check_margin_call(self, high_price, low_price):
        if not self.positions:
            return

        if self.margin_mode == "cross":
            total_maintenance_margin = 0
            for asset_id, pos in self.positions.items():
                if pos["type"] is not None and pos["size"] > 0:
                    notional_value = pos["size"]
                    max_lev, mm_rate, mm_amt, low_s, up_s = self.get_margin_tier(
                        asset_id, notional_value
                    )
                    maintenance_margin = mm_rate * notional_value - mm_amt
                    if maintenance_margin < 0:
                        maintenance_margin = 0
                    total_maintenance_margin += maintenance_margin

            current_portfolio_value = self.portfolio_value
            if current_portfolio_value < total_maintenance_margin:
                # Margin call triggered
                any_long = any(p["type"] == "long" for p in self.positions.values())
                any_short = any(p["type"] == "short" for p in self.positions.values())

                if any_long and any_short:
                    worst_case_price = low_price  # or pick whichever scenario you consider more realistic
                elif any_long:
                    worst_case_price = low_price
                else:
                    worst_case_price = high_price

                self.handle_margin_call(worst_case_price)

        else:
            for i, pos in self.positions.items():
                if pos["type"] and pos["size"] > 0:
                    _, mm_rate, mm_amt, *_ = self.get_margin_tier(i, pos["size"])
                    maintenance = max(mm_rate * pos["size"] - mm_amt, 0.0)
                    # equity allocated to this asset:
                    equity_i = abs(pos["size"]) / pos["leverage"]
                    if equity_i < maintenance:
                        # liquidate this one asset only
                        price = low_price if pos["type"] == "long" else high_price
                        self.execute_order(
                            order_type="sell" if pos["type"] == "long" else "buy",
                            execution_price=price,
                            entry_price=pos["entry_price"],
                            position_type=pos["type"],
                            position_size=pos["size"],
                            previous_leverage=pos["leverage"],
                            is_margin_call=True,
                        )

    def handle_margin_call(self, worst_case_price):
        self.margin_call_triggered = True

        if not self.positions:
            return

        if self.margin_mode == "cross":

            # Liquidate all positions
            for asset_id, position in list(self.positions.items()):
                position_type = position["type"]
                entry_price = position["entry_price"]
                position_size = position["size"]
                previous_leverage = position["leverage"]

                if position_type is not None and position_size > 0:
                    order_type = "sell" if position_type == "long" else "buy"
                    self.execute_order(
                        order_type=order_type,
                        execution_price=worst_case_price,
                        entry_price=entry_price,
                        position_type=position_type,
                        position_size=position_size,
                        previous_leverage=previous_leverage,
                        is_margin_call=True,
                    )

            # After deleting all, re-init positions to maintain keys
            self.positions = {
                i: {
                    "type": None,
                    "entry_price": 0.0,
                    "size": 0.0,
                    "leverage": self.leverage,
                }
                for i in range(self.num_assets)
            }

            if self.limit_bounds:
                self.stop_loss_levels.clear()
                self.take_profit_levels.clear()

            self.episode_metrics["num_margin_calls"] += 1
            self.episode_metrics["list_margin_calls"].append(
                self.episode_metrics["num_margin_calls"]
            )

            # Log trading return (unchanged)
            if len(self.history) > 0:
                if self.history[-1] == 0:
                    trading_return = 0
                else:
                    trading_return = (
                        self.portfolio_value - self.history[-1]
                    ) / self.history[-1]
                self.trading_returns.append(trading_return)
                self.episode_metrics["returns"].append(trading_return)
                if self.portfolio_value <= 0 or self.history[-1] <= 0:
                    log_trading_return = 0
                else:
                    log_trading_return = np.log(self.portfolio_value / self.history[-1])
                self.log_trading_returns.append(log_trading_return)
                final_return = (self.portfolio_value - self.history[0]) / self.history[
                    0
                ]
                self.final_returns.append(final_return)
        else:
            return

    def get_margin_tier(self, asset_id, notional_value):
        # Store tiers in a dictionary keyed by asset_id
        # Each value is a list of tuples: (lower_bound, upper_bound, max_leverage, mm_rate, mm_amount)
        tiers = {
            0: [  # adausdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            1: [  # bnbusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            2: [  # btcusdt
                (0, 50000, 125, 0.004, 0),
                (50000, 600000, 100, 0.005, 50),
                (600000, 3000000, 75, 0.0065, 950),
                (3000000, 12000000, 50, 0.01, 11450),
                (12000000, 70000000, 25, 0.02, 131450),
                (70000000, 100000000, 20, 0.025, 481450),
                (100000000, 230000000, 10, 0.05, 2981450),
                (230000000, 480000000, 5, 0.1, 14481450),
                (480000000, 600000000, 4, 0.125, 26481450),
                (600000000, 800000000, 3, 0.15, 41481450),
                (800000000, 1200000000, 2, 0.25, 121481450),
                (1200000000, 1800000000, 1, 0.5, 421481450),
            ],
            3: [  # dogeusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.007, 20),
                (50000, 750000, 40, 0.01, 170),
                (750000, 2000000, 25, 0.02, 7670),
                (2000000, 4000000, 20, 0.025, 17670),
                (4000000, 20000000, 10, 0.05, 117670),
                (20000000, 40000000, 5, 0.1, 1117670),
                (40000000, 50000000, 4, 0.125, 2117670),
                (50000000, 100000000, 2, 0.25, 8367670),
                (100000000, 200000000, 1, 0.5, 33367670),
            ],
            4: [  # eousdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            5: [  # ethusdt
                (0, 50000, 125, 0.004, 0),
                (50000, 600000, 100, 0.005, 50),
                (600000, 3000000, 75, 0.0065, 950),
                (3000000, 12000000, 50, 0.01, 11450),
                (12000000, 50000000, 25, 0.02, 131450),
                (50000000, 65000000, 20, 0.025, 381450),
                (65000000, 150000000, 10, 0.05, 2006450),
                (150000000, 320000000, 5, 0.1, 9506450),
                (320000000, 400000000, 4, 0.125, 17506450),
                (400000000, 530000000, 3, 0.15, 27506450),
                (530000000, 800000000, 2, 0.25, 80506450),
                (800000000, 1200000000, 1, 0.5, 280506450),
            ],
            6: [  # ltcusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            7: [  # trxusdt
                (0, 10000, 75, 0.0065, 0),
                (10000, 90000, 50, 0.01, 35),
                (90000, 120000, 40, 0.015, 485),
                (120000, 650000, 25, 0.02, 1085),
                (650000, 800000, 20, 0.025, 4335),
                (800000, 3000000, 10, 0.05, 24335),
                (3000000, 6000000, 5, 0.1, 174335),
                (6000000, 12000000, 4, 0.125, 324335),
                (12000000, 20000000, 2, 0.25, 1824335),
                (20000000, 30000000, 1, 0.5, 6824335),
            ],
            8: [  # xlmusdt
                (0, 10000, 75, 0.01, 0),
                (10000, 100000, 50, 0.015, 50),
                (100000, 500000, 25, 0.02, 550),
                (500000, 1000000, 20, 0.025, 3050),
                (1000000, 5000000, 10, 0.05, 28050),
                (5000000, 10000000, 5, 0.1, 278050),
                (10000000, 12500000, 4, 0.125, 528050),
                (12500000, 25000000, 2, 0.25, 2090550),
                (25000000, 50000000, 1, 0.5, 8340550),
            ],
            9: [  # xrpusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 20000, 50, 0.0065, 15),
                (20000, 160000, 40, 0.01, 85),
                (160000, 1000000, 25, 0.02, 1685),
                (1000000, 2000000, 20, 0.025, 6685),
                (2000000, 10000000, 10, 0.05, 56685),
                (10000000, 20000000, 5, 0.1, 556685),
                (20000000, 25000000, 4, 0.125, 1056685),
                (25000000, 50000000, 2, 0.25, 4181685),
                (50000000, 100000000, 1, 0.5, 16681685),
            ],
        }

        # Retrieve the tier list for the given asset_id
        tier_list = tiers[asset_id]
        # Iterate through tiers to find the one matching the notional_value
        for low, high, max_lev, mm_rate, mm_amt in tier_list:
            if low <= notional_value <= high:
                return max_lev, mm_rate, mm_amt, low, high

        # If notional_value exceeds the highest tier, return the last tier's values
        last_tier = tier_list[-1]
        return last_tier[2], last_tier[3], last_tier[4], last_tier[0], last_tier[1]

    def calculate_reward(self):
        # 1) get today's log‐return
        v_t = self.log_trading_returns[-1]
        print("REWARD FROM LOG RETURNS : ", v_t)

        # 2) update EWMA variance for volatility penalty
        self.sigma2 = self.beta * self.sigma2 + (1.0 - self.beta) * (v_t**2)
        sigma = np.sqrt(self.sigma2)

        # 3) compute raw Mean–Vol–CVaR reward
        r_raw = v_t - self.lambda_vol * sigma
        print("REWARD PENALTY FROM VARIANCE COMPONENT : ", -self.lambda_vol * sigma)
        print("REWARD R_RAW : ", r_raw)

        return float(r_raw)

    def plot_alignment(self):
        """
        At the end of an episode, call this to see the two series.
        """
        raw, norm = self.aligned_raw, self.aligned_norm
        steps_raw = range(len(raw))
        steps_norm = range(len(norm))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(steps_raw, raw, label="raw OHLCV")
        ax1.set_ylabel("raw price")
        ax1.legend()

        ax2.plot(steps_norm, norm, label="normalized feature")
        ax2.set_xlabel("step")
        ax2.set_ylabel("normalized value")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    #  VISUAL SANITY-CHECK ─ raw OHLCV  vs  last row of each normalised seq
    # ----------------------------------------------------------------------
    def check_alignment(
        self,
        asset_idx: int = 2,  # choose asset (0-9)
        ohlcv_field: int = 3,  # 0=open … 3=close 4=volume
        max_points: int = 500,  # trim plot for speed
    ):
        """
        Plot the selected raw OHLCV series **and** the corresponding values
        from the last row of every sequence currently in `self.sequence_buffer`.

        Call it right after `reset()` (or before calling `step()` the first
        time) so the buffer is fully populated.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if len(self.sequence_buffer) == 0:
            print("⚠️  sequence_buffer is empty – call this right after reset()")
            return

        # ------- raw -------------------------------------------------------
        raw_series_full = self._raw_np[:, asset_idx * 5 + ohlcv_field].astype(
            np.float32
        )

        # ------- normalised (last rows) -----------------------------------
        seq_array = np.stack(self.sequence_buffer)  # (input_len, L, D)
        norm_series = seq_array[:, -1, asset_idx * 5 + ohlcv_field]

        # drop the first input_length-1 raw rows so indices match
        aligned_raw = raw_series_full[
            self.input_length : self.input_length + len(norm_series)
        ]

        # trimming
        N = min(len(aligned_raw), max_points)
        aligned_raw = aligned_raw[:N]
        norm_series = norm_series[:N]

        # ------- plotting  -------------------------------------------------
        fig, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax_raw.plot(aligned_raw, color="tab:blue", label="raw")
        ax_raw.set_title(f"RAW  – asset {asset_idx}, field {ohlcv_field}")
        ax_raw.legend()

        ax_norm.plot(norm_series, color="tab:green", label="normalised\n(last row)")
        ax_norm.set_title("NORMALISED sequence values")
        ax_norm.legend()

        plt.tight_layout()
        plt.show()

    def log_step_metrics(self):
        if self.limit_bounds and self.positions:
            # Compute risk taken with stop loss levels for each asset and average or sum them
            # We'll just append the average risk taken across all positions with a stop loss
            total_risk_taken = []
            for asset, pos in self.positions.items():
                if pos["type"] is not None and pos["size"] > 0:
                    entry_price = pos["entry_price"]
                    # Assume stop_loss_levels is keyed by (asset, entry_price)
                    if (asset, entry_price) in self.stop_loss_levels:
                        stop_loss_price, _ = self.stop_loss_levels[(asset, entry_price)]
                        current_risk = (
                            pos["size"]
                            * (abs(entry_price - stop_loss_price) / entry_price)
                        ) / self.portfolio_value
                        if not np.isnan(current_risk):
                            total_risk_taken.append(current_risk)
            if total_risk_taken:
                avg_risk = np.mean(total_risk_taken)
                self.episode_metrics["risk_taken"].append(avg_risk)

        # Log drawdowns
        drawdown = calculate_max_drawdown(self.history)
        if not np.isnan(drawdown):
            self.episode_metrics["drawdowns"].append(drawdown)

    def log_episode_metrics(self):
        # Convert history to cumulative returns
        portfolio_values = pd.Series(self.history)
        portfolio_returns = portfolio_values.pct_change().fillna(0)
        portfolio_cum_returns = (1 + portfolio_returns).cumprod()
        print("CUM RETURNS PORTFOLIO : ", portfolio_cum_returns)
        print("TRUE PORTFOLIO CUM RETURNS : ", self.portfolio_value)

        # Compute metrics
        episode_metrics = compute_metrics(
            portfolio_cum_returns, portfolio_returns, self.initial_balance
        )

        # Store metrics
        for key, value in episode_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # Existing metrics
        self.metrics["compounded_returns"].append(self.compounded_returns - 1)
        self.metrics["rewards"].append(self.episode_metrics.get("rewards", [0])[-1])

    def plot_episode_metrics(self):
        plt.style.use("ggplot")

        # 1. Cumulative Portfolio Value
        start_date = self.current_dates[0]
        end_date = self.current_dates[-1]

        print("START DATE : ", start_date)
        print("END DATE : ", end_date)

        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(
            self.current_dates,
            self.history,
            color="#1f77b4",
            linewidth=2.5,
            alpha=0.9,
        )
        ax1.set_ylabel("Portfolio Value", fontsize=12)
        ax1.set_title("Cumulative Portfolio Value Over Time", fontsize=14, pad=20)
        ax1.grid(True, linestyle=":", alpha=0.7)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Format x-axis as dates
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig1.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("output1.png", dpi=150)
        plt.show()

        # 2. Portfolio Returns
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.bar(
            self.current_dates[: len(self.episode_metrics["returns"])],
            np.array(self.episode_metrics["returns"]) * 100,
            color="#2ca02c",
            width=0.8,
            alpha=0.7,
        )
        ax2.set_ylabel("Return (%)", fontsize=12)
        ax2.set_title("Portfolio Returns per Step", fontsize=14, pad=20)
        ax2.grid(True, linestyle=":", alpha=0.7)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig2.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("output2.png", dpi=150)
        plt.show()

        # 3. Asset Allocation Weights
        # map from index positions to short asset names
        asset_names = [
            "ada",
            "bnb",
            "btc",
            "doge",
            "eos",
            "eth",
            "ltc",
            "trx",
            "xlm",
            "xrp",
        ]

        # build the weights DataFrame with the proper column names
        weights_df = pd.DataFrame(
            self.episode_metrics["weights"],
            index=self.current_dates[: len(self.episode_metrics["weights"])],
            columns=asset_names,
        )
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(asset_names)))
        ax3.stackplot(
            weights_df.index,
            weights_df.values.T * 100,
            labels=weights_df.columns,
            alpha=0.8,
            colors=colors,
        )
        ax3.set_ylabel("Asset Allocation Weight (%)", fontsize=12)
        ax3.set_title("Desired Portfolio Composition Over Time", fontsize=14, pad=20)
        ax3.legend(
            ncol=3,
            fontsize=9,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            borderpad=1,
        )
        ax3.grid(True, linestyle=":", alpha=0.7)
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig3.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("output3.png", dpi=150)
        plt.show()

        # 4. Realized Allocation Weights
        realised_w_df = pd.DataFrame(
            self.episode_metrics["true_weights"],
            index=self.current_dates[: len(self.episode_metrics["true_weights"])],
            columns=asset_names,
        )
        csv_path = f"realized_weights.csv"
        realised_w_df.to_csv(csv_path, index=True)
        desired_w_df = pd.DataFrame(
            self.episode_metrics["weights"],
            index=self.current_dates[: len(self.episode_metrics["weights"])],
            columns=asset_names,
        )
        print(f"✓ realised weights saved to {csv_path}")

        fig4, ax4 = plt.subplots(figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(asset_names)))
        ax4.stackplot(
            realised_w_df.index,
            realised_w_df.values.T * 100,
            labels=realised_w_df.columns,
            alpha=0.8,
            colors=colors,
        )
        ax4.set_ylabel("Realised Weight (%)")
        ax4.set_title("Realized Portfolio Composition Over Time")
        ax4.legend(
            ncol=3,
            fontsize=9,
            framealpha=0.8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
        )
        ax4.grid(True, linestyle=":", alpha=0.7)
        ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig4.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("output4.png", dpi=150)
        plt.show()

        # 4. Divergence between desired & realised weights (MAE)
        mae_series = pd.Series(
            self.episode_metrics["weight_mae"],
            index=self.current_dates[: len(self.episode_metrics["weight_mae"])],
        )
        fig5, ax5 = plt.subplots(figsize=(14, 5))
        ax5.plot(mae_series.index, mae_series.values, color="#d62728", linewidth=2)
        ax5.set_ylabel("Σ |Δw|")
        ax5.set_title("Step-wise MAE between desired and realised weights")
        ax5.grid(True, linestyle=":", alpha=0.7)
        ax5.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig5.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("output5.png", dpi=150)
        plt.show()
        # ---- numeric summaries --------------------------------
        te = np.sqrt(
            np.mean(
                np.square(
                    np.array(self.episode_metrics["weights"])
                    - np.array(self.episode_metrics["true_weights"])
                )
            )
        )
        rej_rate = self.rejected_trade_count / (
            max(len(self.action_history), 1) * self.num_assets
        )
        print(f"Tracking-Error (weights): {te:.4f}")
        print(
            f"Rejected-trade ratio   : {rej_rate:.2%} ({self.rejected_trade_count} rejects / {(max(len(self.action_history), 1) * self.num_assets)} total)"
        )
        is_shortfall = (
            (desired_w_df - realised_w_df).abs().sum(axis=1).mean()
        )  # Implementation Shortfall proxy
        mad_weights = (
            (desired_w_df - realised_w_df).abs().values.mean()
        )  # Mean-Abs-Dev over all t,i
        print(f"Implementation-Shortfall (mean |Δw| per step): {is_shortfall:.4%}")
        print(f"Mean-Absolute-Deviation of weights:            {mad_weights:.4%}")

        # ---- breakdown by cause -------------------------------
        dev_table = pd.Series(self.weight_dev_stats).sort_values(ascending=False)
        print("\nCumulative |Δw| split by cause:")
        print(dev_table.to_string(float_format=lambda x: f"{x:.4f}"))

        fig6, ax6 = plt.subplots(figsize=(14, 5))
        dev_table.plot(kind="bar", ax=ax6)
        ax6.set_ylabel("|Δw| contribution")
        ax6.set_title("Weight-deviation attribution by cause")

        print("\nPercentage of cumulative |Δw| split by cause:")
        dev_share = dev_table / dev_table.sum()
        print(dev_share.to_string(float_format=lambda x: f"{x:.2%}"))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig("output6.png", dpi=150)
        plt.show()

        # at the end of plot_episode_metrics()
        fig4, ax4 = plt.subplots(figsize=(14, 5))
        dates = self.current_dates[: len(self.episode_metrics["desired_leverage"])]
        ax4.plot(
            dates,
            self.episode_metrics["desired_leverage"],
            label="Desired Leverage",
            linewidth=2,
        )
        ax4.plot(
            dates,
            self.episode_metrics["realized_leverage"],
            label="Realized Leverage",
            linewidth=2,
        )
        ax4.set_ylabel("Leverage")
        ax4.set_title("Desired vs. Realized Portfolio Leverage")
        ax4.legend()
        ax4.grid(True, linestyle=":")
        fig4.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("output7.png", dpi=150)
        plt.show()

        lev_df = pd.DataFrame(
            {
                "date": self.current_dates[
                    : len(self.episode_metrics["desired_leverage"])
                ],
                "desired_leverage": self.episode_metrics["desired_leverage"],
                "realized_leverage": self.episode_metrics["realized_leverage"],
            }
        )
        csv_path_lev = f"leverages_series.csv"
        lev_df.set_index("date", inplace=True)
        lev_df.to_csv("leverages_series.csv", index=True)
        print("Leverage series saved. ")

    def plot_metrics(self):
        plt.figure(figsize=(14, 10))

        plt.subplot(4, 2, 1)
        plt.plot(self.metrics["returns"], label="Returns")
        plt.title("Returns Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 2)
        plt.plot(self.metrics["sharpe_ratios"], label="Sharpe Ratio")
        plt.title("Sharpe Ratio Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 3)
        plt.plot(self.metrics["drawdowns"], label="Max Drawdown")
        plt.title("Max Drawdown Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 4)
        plt.plot(self.metrics["risk_taken"], label="Risk Taken")
        plt.title("Risk Taken Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 5)
        plt.plot(self.metrics["num_margin_calls"], label="Margin Calls")
        plt.title("Number of Margin Calls Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 6)
        plt.plot(self.metrics["num_trades"], label="Number of Trades")
        plt.title("Number of Trades Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 7)
        plt.plot(self.metrics["leverage_used"], label="Leverage Used")
        plt.title("Leverage Used Over Iterations")
        plt.legend()

        plt.subplot(4, 2, 8)
        plt.plot(self.metrics["final_balance"], label="Final Balance")
        plt.title("Final Balance Over Iterations")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def render(self):
        self.plot_episode_metrics()


# Example functions for risk measures
def calculate_empirical_var(returns, confidence_level):
    if len(returns) == 0:
        return 0
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    return var


def calculate_empirical_es(returns, confidence_level):
    if len(returns) == 0:
        return 0
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    tail_returns = sorted_returns[:index]
    es = -np.mean(tail_returns) if len(tail_returns) > 0 else 0
    return es


def calculate_max_drawdown(portfolio_values):
    if len(portfolio_values) == 0:
        return 0
    drawdowns = []
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak != 0 else 0
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns) if drawdowns else 0
    return max_drawdown


def calculate_skewness(returns):
    if len(returns) < 2:
        return 0
    std_returns = np.std(returns)
    if std_returns == 0:
        return 0
    return np.mean((returns - np.mean(returns)) ** 3) / std_returns**3


def calculate_kurtosis(returns):
    if len(returns) < 2:
        return 0
    std_returns = np.std(returns)
    if std_returns == 0:
        return 0
    return np.mean((returns - np.mean(returns)) ** 4) / std_returns**4


def compute_drawdowns(portfolio_cum_returns):
    rolling_max = portfolio_cum_returns.cummax()
    drawdown = (portfolio_cum_returns - rolling_max) / rolling_max
    return drawdown


def compute_metrics(
    portfolio_cum_returns, portfolio_returns, initial_value, risk_free_rate=0.0
):
    metrics = {}
    if portfolio_cum_returns.empty or portfolio_returns.empty:
        print("Portfolio data empty.")
        return {
            "Cumulative Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Annualized Downside Semi-Std": np.nan,
            "Annualized Upside Semi-Std": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Lowest Point (% of Initial Capital)": np.nan,
            "VaR (99%)": np.nan,
            "CVaR (99%)": np.nan,
            "Calmar Ratio": np.nan,
            "Upside Potential Ratio": np.nan,
            "Omega Ratio": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
        }

    # Cumulative Return
    cumulative_return = (portfolio_cum_returns.iloc[-1]) - 1
    metrics["Cumulative Return"] = cumulative_return

    # Annualized Return (assuming 365 steps per "year" in your environment)
    metrics["Annualized Return"] = np.mean(portfolio_returns) * 365

    # Annualized Volatility
    metrics["Annualized Volatility"] = np.std(portfolio_returns) * np.sqrt(365)

    # Annualized Downside Semi-Std (threshold = 0)
    downside_diff = np.clip(portfolio_returns, np.NINF, 0)
    downside_diff_squared = np.square(downside_diff)
    downside_semi_std = np.sqrt(np.mean(downside_diff_squared)) * np.sqrt(365)
    metrics["Annualized Downside Semi-Std"] = downside_semi_std

    # Annualized Upside Semi-Std (threshold = 0)
    upside_diff = np.clip(portfolio_returns, 0, np.inf)
    upside_diff_squared = np.square(upside_diff)
    upside_semi_std = np.sqrt(np.mean(upside_diff_squared)) * np.sqrt(365)
    metrics["Annualized Upside Semi-Std"] = upside_semi_std

    # Sharpe Ratio
    metrics["Sharpe Ratio"] = (
        (metrics["Annualized Return"] - risk_free_rate)
        / metrics["Annualized Volatility"]
        if metrics["Annualized Volatility"] != 0
        else np.nan
    )

    # Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = (
        np.std(downside_returns) * np.sqrt(365)
        if not downside_returns.empty
        else np.nan
    )
    metrics["Sortino Ratio"] = (
        (metrics["Annualized Return"] - risk_free_rate) / downside_std
        if downside_std != 0
        else np.nan
    )

    # Maximum Drawdown
    drawdown = compute_drawdowns(portfolio_cum_returns)
    metrics["Maximum Drawdown"] = drawdown.min()

    # Lowest Point (% of Initial Capital)
    metrics["Lowest Point (% of Initial Capital)"] = portfolio_cum_returns.min()

    # VaR and CVaR at 99%
    metrics["VaR (99%)"] = (
        np.percentile(portfolio_returns, 1) if not portfolio_returns.empty else np.nan
    )
    metrics["CVaR (99%)"] = (
        portfolio_returns[portfolio_returns <= metrics["VaR (99%)"]].mean()
        if not portfolio_returns.empty
        else np.nan
    )

    # Calmar Ratio
    metrics["Calmar Ratio"] = (
        metrics["Annualized Return"] / abs(metrics["Maximum Drawdown"])
        if metrics["Maximum Drawdown"] != 0
        else np.nan
    )

    # Upside Potential Ratio
    positive_returns = portfolio_returns[portfolio_returns > 0]
    downside_std = np.std(portfolio_returns[portfolio_returns < 0])
    metrics["Upside Potential Ratio"] = (
        np.mean(positive_returns) / downside_std if downside_std != 0 else np.nan
    )

    # Omega Ratio
    threshold = 0
    gains = portfolio_returns[portfolio_returns > threshold] - threshold
    losses = threshold - portfolio_returns[portfolio_returns < threshold]
    metrics["Omega Ratio"] = gains.sum() / losses.sum() if losses.sum() != 0 else np.nan

    # Skewness and Kurtosis
    metrics["Skewness"] = (
        scipy.stats.skew(portfolio_returns) if not portfolio_returns.empty else np.nan
    )
    metrics["Kurtosis"] = (
        scipy.stats.kurtosis(portfolio_returns)
        if not portfolio_returns.empty
        else np.nan
    )

    return metrics
