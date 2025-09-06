from collections import Counter
from pathlib import Path
from typing import Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from database.db_loader import load_tob_range, HISTORY_DB


# --------------------------
# 1) Resample + base features
# --------------------------

def make_1s_frame(df_tob: pd.DataFrame) -> pd.DataFrame:
    idx = df_tob.set_index("ts")
    out = pd.DataFrame({
        "mid": idx[["bid", "ask"]].mean(axis=1).resample("1s").last().ffill(),
        "bid": idx["bid"].resample("1s").last().ffill(),
        "ask": idx["ask"].resample("1s").last().ffill(),
        "spread": (idx["ask"] - idx["bid"]).resample("1s").mean().ffill(),
        "bid_size": idx["bidSize"].resample("1s").mean().fillna(0.0),
        "ask_size": idx["askSize"].resample("1s").mean().fillna(0.0),
        "tick_count": idx["bid"].resample("1s").size().fillna(0.0),
    })
    # res = pd.DataFrame({
    #     "mid": idx["mid"].resample("1s").last().ffill(),
    #     "spread": idx["spread"].resample("1s").mean().ffill(),
    #     "bid_size": idx["bidSize"].resample("1s").mean().fillna(0.0),
    #     "ask_size": idx["askSize"].resample("1s").mean().fillna(0.0),
    #     "tick_count": idx["mid"].resample("1s").size()
    # })
    out = out.reset_index()  # keep 'ts' column
    return out


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds your original 5 features:
      - ret1, ret5, spread_bp, imbalance, tick_count
    """
    df = df.copy()
    df["ret1"] = df["mid"].pct_change().fillna(0.0)
    df["ret5"] = df["mid"].pct_change(5).fillna(0.0)
    df["imb"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1e-9)
    # (spread / mid), keep as fraction (multiply by 1e4 later in plots if you want pips/bps)
    df["spread_bp"] = (df["spread"] / df["mid"]).fillna(0.0)
    return df

# def add_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["ret1"] = df["mid"].pct_change().fillna(0.0)
#     df["ret5"] = df["mid"].pct_change(5).fillna(0.0)
#     df["imb"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1e-9)
#     df["spread_bp"] = (df["spread"] / df["mid"]).fillna(0.0)
#     return df



# --------------------------
# 2) Technical indicators
# --------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(price: pd.Series, n: int = 14) -> pd.Series:
    delta = price.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _stoch(series: pd.Series, n: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Approximate stochastic using rolling min/max of mid (no OHLC available).
    Returns: %K, %D, williams %R, rolling range
    """
    low = series.rolling(n).min()
    high = series.rolling(n).max()
    rng = (high - low).replace(0.0, np.nan)
    k = 100.0 * (series - low) / (rng + 1e-12)
    dline = k.rolling(d).mean()
    willr = -100.0 * (high - series) / (rng + 1e-12)
    return k, dline, willr, rng


def _cci(series: pd.Series, n: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI) (using mid for Typical Price).
    """
    tp = series  # approximate Typical Price as mid (no OHLC)
    ma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda s: np.mean(np.abs(s - s.mean())), raw=False)
    cci = (tp - ma) / (0.015 * (mad + 1e-12))
    return cci


def _rolling_autocorr(x: pd.Series, window: int, lag: int = 1) -> pd.Series:
    # safe rolling autocorr; if variance ~ 0 returns 0
    def _ac(s):
        s = s.dropna()
        if len(s) <= lag or s.std(ddof=0) == 0:
            return 0.0
        return s.autocorr(lag=lag)
    return x.rolling(window).apply(_ac, raw=False)

def add_technical_indicators(df: pd.DataFrame, pip: float = 1e-4) -> pd.DataFrame:
    """
    Adds a large set of indicators derived from mid/bid/ask/bid_size/ask_size/tick_count.
    All NaNs are left in-place here; the Dataset will fillna(0.0) after assembling sequences.
    """
    out = df.copy()
    price = out["mid"]

    # ---- Momentum / Returns in pips and %
    for w in [1, 2, 5, 10, 20, 30, 60, 120]:
        out[f"ret{w}"] = price.pct_change(w)
        out[f"mom{w}_pips"] = price.diff(w) / pip

    # ---- SMA / EMA (seconds, since we resampled to 1s)
    for w in [5, 10, 20, 30, 60, 120, 300]:
        out[f"sma{w}"] = price.rolling(w).mean()
        out[f"ema{w}"] = _ema(price, w)
        out[f"zmid{w}"] = (price - out[f"sma{w}"]) / (price.rolling(w).std() + 1e-9)

    # ---- MACD (use second-based windows; classic is (12, 26, 9) bars)
    macd_fast, macd_slow, macd_sig = 12, 26, 9
    macd = _ema(price, macd_fast) - _ema(price, macd_slow)
    macd_signal = _ema(macd, macd_sig)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # ---- Bollinger (k=2)
    for w in [20, 60, 120]:
        ma = price.rolling(w).mean()
        sd = price.rolling(w).std()
        upper = ma + 2 * sd
        lower = ma - 2 * sd
        out[f"bb{w}_pctb"] = (price - lower) / (upper - lower + 1e-12)
        out[f"bb{w}_width_pips"] = (upper - lower) / pip

    # ---- RSI (Wilder)
    for w in [14, 30, 60]:
        out[f"rsi{w}"] = _rsi(price, w)

    # ---- Stochastic / Williams %R
    for w in [14, 30, 60]:
        k, dline, willr, rng = _stoch(price, n=w, d=3)
        out[f"stochK{w}"] = k
        out[f"stochD{w}"] = dline
        out[f"willr{w}"] = willr
        out[f"rng{w}_pips"] = rng / pip

    # ---- CCI
    for w in [20, 60, 120]:
        out[f"cci{w}"] = _cci(price, n=w)

    # ---- Volatility proxies
    out["ret1"] = price.pct_change()
    for w in [30, 60, 120, 300]:
        out[f"rv{w}"] = out["ret1"].rolling(w).std()              # realized vol (pct)
        out[f"vol{w}_pips"] = price.diff().rolling(w).std() / pip # realized vol (pips)

    # ---- Spread features
    out["spread_frac"] = (out["spread"] / price)
    out["spread_pips"] = out["spread"] / pip
    for w in [30, 60, 120, 300]:
        m = out["spread_pips"].rolling(w).mean()
        s = out["spread_pips"].rolling(w).std()
        out[f"spread_mean{w}_pips"] = m
        out[f"spread_z{w}"] = (out["spread_pips"] - m) / (s + 1e-9)

    # ---- Microprice & imbalance flavors
    out["microprice"] = (out["ask"] * out["bid_size"] + out["bid"] * out["ask_size"]) / (
        out["bid_size"] + out["ask_size"] + 1e-9
    )
    out["micro_dev_pips"] = (out["microprice"] - price) / pip
    out["imb"] = (out["bid_size"] - out["ask_size"]) / (out["bid_size"] + out["ask_size"] + 1e-9)
    for w in [30, 60, 120]:
        out[f"imb_mean{w}"] = out["imb"].rolling(w).mean()

    # ---- OBV-like using tick_count as "volume" proxy
    sign = np.sign(price.diff().fillna(0.0))
    out["obv_ticks"] = (sign * out["tick_count"]).cumsum()

    # ---- Autocorrelation of returns
    for w in [60, 120, 300]:
        out[f"autocorr_ret1_w{w}"] = _rolling_autocorr(out["ret1"], window=w, lag=1)

    # ---- Time-of-day cyclical encodings
    sec = (pd.to_datetime(out["ts"]).dt.hour * 3600
           + pd.to_datetime(out["ts"]).dt.minute * 60
           + pd.to_datetime(out["ts"]).dt.second)
    out["tod_sin"] = np.sin(2 * np.pi * sec / 86400.0)
    out["tod_cos"] = np.cos(2 * np.pi * sec / 86400.0)

    # ---- Fibonacci position within rolling window ranges
    for w in [300, 900, 1800]:  # 5m, 15m, 30m
        low = price.rolling(w).min()
        high = price.rolling(w).max()
        rng = (high - low) + 1e-12
        pos = (price - low) / rng   # 0..1
        out[f"fib_pos{w}"] = pos
        for lvl, name in [(0.236, "236"), (0.382, "382"), (0.5, "500"), (0.618, "618")]:
            out[f"fib_dist{w}_{name}"] = (pos - lvl)

    return out


# --------------------------
# 3) Sequence builder
# --------------------------


def class_counts(y: np.ndarray):
    vals, counts = np.unique(y, return_counts=True)
    d = {int(v): int(c) for v, c in zip(vals, counts)}
    # ensure all 0,1,2 present in dict for readability
    for k in [0, 1, 2]:
        d.setdefault(k, 0)
    return d


def build_sequences_by_cols(
    df_1s: pd.DataFrame,
    feat_cols: Sequence[str],
    lookback: int = 120,
    horizon: int = 10,
    stride: int = 1,
    classify: bool = True,
    tau: float = 0.00010,     # label threshold on future return
    burn_in: Optional[int] = None,  # skip first N secs to let indicators warm up
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build (X, y) from any set of feature columns in df_1s, with an optional burn-in.
    """
    df = df_1s.copy()
    if burn_in is None:
        # Use a conservative burn equal to the largest rolling window we used (e.g., 1800)
        burn_in = 300  # 5min; raise to 1800 if you want max warmup
    if burn_in > 0 and len(df) > burn_in:
        df = df.iloc[burn_in:].copy()

    feats = df[list(feat_cols)].astype(np.float32).fillna(0.0).values
    mid = df["mid"].astype(np.float64).values

    X, y = [], []
    for i in range(lookback, len(df) - horizon, stride):
        X.append(feats[i - lookback:i])
        r = (mid[i + horizon] - mid[i]) / (mid[i] + 1e-12)
        if classify:
            if r > tau:
                lab = 2  # up
            elif r < -tau:
                lab = 0  # down
            else:
                lab = 1  # flat
            y.append(lab)
        else:
            y.append(r)

    X = np.stack(X) if X else np.zeros((0, lookback, len(feat_cols)), dtype=np.float32)
    y = np.array(y, dtype=np.int64 if classify else np.float32)

    # Normalization stats over the features
    norm = {
        "feat_cols": list(feat_cols),
        "feat_mean": feats.mean(axis=0).tolist() if len(feats) else [0.0] * len(feat_cols),
        "feat_std": (feats.std(axis=0) + 1e-9).tolist() if len(feats) else [1.0] * len(feat_cols),
        "lookback": lookback,
        "horizon": horizon,
        "classify": classify,
        "tau": tau,
        "burn_in": burn_in,
    }
    return X, y, norm


# def build_sequences(df_1s: pd.DataFrame, lookback: int = 120, horizon: int = 10,
#                     stride: int = 1, classify: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
#     """
#     Targets: future return over `horizon` seconds on mid: r = (mid[t+h]-mid[t])/mid[t]
#     Classification: label {-1,0,1} by threshold τ; else regression.
#     """
#     feats = df_1s[["ret1", "ret5", "spread_bp", "imb", "tick_count"]].values.astype(np.float32)
#     mid = df_1s["mid"].values.astype(np.float64)
#
#     X, y = [], []
#     for i in range(lookback, len(df_1s) - horizon, stride):
#         X.append(feats[i - lookback:i])
#         r = (mid[i + horizon] - mid[i]) / mid[i]
#         if classify:
#             tau = 0.00010  # 0.2 bp ~ tweak later
#             if r > tau:
#                 lab = 2  # up
#             elif r < -tau:
#                 lab = 0  # down
#             else:
#                 lab = 1  # flat
#             y.append(lab)
#         else:
#             y.append(r)
#
#     X = np.stack(X) if X else np.zeros((0, lookback, feats.shape[1]), dtype=np.float32)
#     y = np.array(y, dtype=np.int64 if classify else np.float32)
#     norm = {
#         "feat_mean": feats.mean(axis=0).tolist(),
#         "feat_std": (feats.std(axis=0) + 1e-9).tolist(),
#         "lookback": lookback,
#         "horizon": horizon,
#         "classify": classify
#     }
#
#     return X, y, norm


# --------------------------
# 4) Dataset
# --------------------------

class EurusdTickDataset(Dataset):
    """
    When feature_set is:
      - "basic": use the original 5 features: ['ret1','ret5','spread_bp','imb','tick_count']
      - "full":  use the large technical set defined in add_technical_indicators
      - a list/tuple of column names: use exactly those columns (they must exist)
    """
    def __init__(
        self,
        symbol: str,
        start_iso: str,
        end_iso: str,
        lookback: int = 120,
        horizon: int = 10,
        classify: bool = True,
        stride: int = 1,
        db_path: Optional[Path] = None,
        normalize: bool = True,
        norm_stats: Optional[dict] = None,
        balance: str = "undersample",        # "none" | "undersample" | "oversample"
        random_state: int = 42,
        feature_set: Union[str, Sequence[str]] = "full",
        pip: float = 1e-4,
    ):
        db = db_path or HISTORY_DB
        raw = load_tob_range(symbol, start_iso, end_iso, db, pip=1.0)
        if raw.empty:
            # Empty fallback
            self.X = np.zeros((0, lookback, 5), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.int64 if classify else np.float32)
            self.norm = {
                "feat_cols": ["ret1", "ret5", "spread_bp", "imb", "tick_count"],
                "feat_mean": [0] * 5,
                "feat_std": [1] * 5,
                "lookback": lookback,
                "horizon": horizon,
                "classify": classify,
            }
            self.feat_cols = self.norm["feat_cols"]
            return

        # 1s frame + basic features
        f1s = make_1s_frame(raw)
        f1s = add_basic_features(f1s)

        # add extended indicators if desired
        if feature_set == "full":
            f1s = add_technical_indicators(f1s, pip=pip)
            # choose all numeric columns that are *features*
            # keep 'ts' and raw book columns out; keep tick_count as a feature
            exclude = {"ts", "bid", "ask", "spread", "bid_size", "ask_size", "mid"}
            feat_cols = [c for c in f1s.columns if c not in exclude]
        elif feature_set == "basic":
            feat_cols = ["ret1", "ret5", "spread_bp", "imb", "tick_count"]
        else:
            # custom list
            feat_cols = list(feature_set)

        # Assemble sequences (with burn-in to let indicators warm)
        X, y, norm = build_sequences_by_cols(
            f1s, feat_cols,
            lookback=lookback, horizon=horizon, stride=stride,
            classify=classify, tau=0.00010, burn_in=300
        )

        # Optionally balance classes
        if balance != "none" and len(y) > 0 and classify:
            print("Before Balancing:", class_counts(y))
            rng = np.random.default_rng(random_state)
            counts = Counter(y)
            min_count = min(counts.values())
            max_count = max(counts.values())
            indices = []

            if balance == "undersample":
                for c in np.unique(y):
                    idx = np.where(y == c)[0]
                    if len(idx) > min_count:
                        idx = rng.choice(idx, min_count, replace=False)
                    indices.extend(idx)
            elif balance == "oversample":
                for c in np.unique(y):
                    idx = np.where(y == c)[0]
                    if len(idx) < max_count:
                        extra = rng.choice(idx, max_count - len(idx), replace=True)
                        idx = np.concatenate([idx, extra])
                    indices.extend(idx)
            else:
                # "none"
                indices = np.arange(len(y))

            rng.shuffle(indices)
            X, y = X[indices], y[indices]
            print("After Balancing:", class_counts(y))

        # Normalization
        if normalize and len(X):
            if norm_stats is None:
                mean = np.array(norm["feat_mean"], dtype=np.float32)
                std = np.array(norm["feat_std"], dtype=np.float32)
            else:
                mean = np.array(norm_stats["feat_mean"], dtype=np.float32)
                std = np.array(norm_stats["feat_std"], dtype=np.float32)

            X = (X - mean) / std
            self.norm = {
                "feat_cols": norm["feat_cols"],
                "feat_mean": mean.tolist(),
                "feat_std": std.tolist(),
                "lookback": lookback,
                "horizon": horizon,
                "classify": classify,
                "tau": norm["tau"],
                "burn_in": norm["burn_in"],
            }
        else:
            self.norm = norm

        self.X, self.y = X, y
        self.feat_cols = list(feat_cols)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])
#
# class EurusdTickDataset(Dataset):
#     def __init__(self, symbol: str, start_iso: str, end_iso: str,
#                  lookback=120, horizon=10, classify=True, stride=1,
#                  db_path: Optional[Path] = None, normalize=True, norm_stats=None,
#                  balance="undersample", random_state: int = 42):
#         db = db_path or HISTORY_DB
#         raw = load_tob_range(symbol, start_iso, end_iso, db, pip=1.0)
#         if raw.empty:
#             self.X = np.zeros((0, lookback, 5), dtype=np.float32)
#             self.y = np.zeros((0,), dtype=np.int64 if classify else np.float32)
#             self.norm = {"feat_mean": [0] * 5, "feat_std": [1] * 5, "lookback": lookback, "horizon": horizon,
#                          "classify": classify}
#         else:
#             f1s = make_1s_frame(raw)
#             f1s = add_features(f1s)
#             # print(f1s.iloc[100])
#             X, y, norm = build_sequences(f1s, lookback, horizon, stride, classify)
#             print("Before Balancing:", class_counts(y))
#             # 🔹 Balance here
#             if balance != "none" and len(y) > 0:
#                 rng = np.random.default_rng(random_state)
#                 counts = Counter(y)
#                 min_count = min(counts.values())
#                 max_count = max(counts.values())
#                 indices = []
#
#                 if balance == "undersample":
#                     for c in np.unique(y):
#                         idx = np.where(y == c)[0]
#                         if len(idx) > min_count:
#                             idx = rng.choice(idx, min_count, replace=False)
#                         indices.extend(idx)
#                 elif balance == "oversample":
#                     for c in np.unique(y):
#                         idx = np.where(y == c)[0]
#                         if len(idx) < max_count:
#                             extra = rng.choice(idx, max_count - len(idx), replace=True)
#                             idx = np.concatenate([idx, extra])
#                         indices.extend(idx)
#
#                 rng.shuffle(indices)
#                 X, y = X[indices], y[indices]
#                 print("After Balancing:", class_counts(y))
#
#             if normalize and len(X):
#                 mean = np.array(norm_stats["feat_mean"] if norm_stats else norm["feat_mean"], dtype=np.float32)
#                 std = np.array(norm_stats["feat_std"] if norm_stats else norm["feat_std"], dtype=np.float32)
#                 X = (X - mean) / std
#                 self.norm = {"feat_mean": mean.tolist(), "feat_std": std.tolist(),
#                              "lookback": lookback, "horizon": horizon, "classify": classify}
#             else:
#                 self.norm = norm
#
#             self.X, self.y = X, y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, i):
#         return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])
