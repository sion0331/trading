from collections import Counter
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from database.db_loader import load_tob_range

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "trading.db"


def make_1s_frame(df_tob: pd.DataFrame) -> pd.DataFrame:
    idx = df_tob.set_index("ts")
    res = pd.DataFrame({
        "mid": idx["mid"].resample("1s").last().ffill(),
        "spread": idx["spread"].resample("1s").mean().ffill(),
        "bid_size": idx["bidSize"].resample("1s").mean().fillna(0.0),
        "ask_size": idx["askSize"].resample("1s").mean().fillna(0.0),
        "tick_count": idx["mid"].resample("1s").size()
    })
    res = res.reset_index()
    return res


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["mid"].pct_change().fillna(0.0)
    df["ret5"] = df["mid"].pct_change(5).fillna(0.0)
    df["imb"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1e-9)
    df["spread_bp"] = (df["spread"] / df["mid"]).fillna(0.0)
    return df


def class_counts(y: np.ndarray):
    vals, counts = np.unique(y, return_counts=True)
    d = {int(v): int(c) for v, c in zip(vals, counts)}
    # ensure all 0,1,2 present in dict for readability
    for k in [0, 1, 2]:
        d.setdefault(k, 0)
    return d


def build_sequences(df_1s: pd.DataFrame, lookback: int = 120, horizon: int = 10,
                    stride: int = 1, classify: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Targets: future return over `horizon` seconds on mid: r = (mid[t+h]-mid[t])/mid[t]
    Classification: label {-1,0,1} by threshold τ; else regression.
    """
    feats = df_1s[["ret1", "ret5", "spread_bp", "imb", "tick_count"]].values.astype(np.float32)
    mid = df_1s["mid"].values.astype(np.float64)

    X, y = [], []
    for i in range(lookback, len(df_1s) - horizon, stride):
        X.append(feats[i - lookback:i])
        r = (mid[i + horizon] - mid[i]) / mid[i]
        if classify:
            tau = 0.00010  # 0.2 bp ~ tweak later
            if r > tau:
                lab = 2  # up
            elif r < -tau:
                lab = 0  # down
            else:
                lab = 1  # flat
            y.append(lab)
        else:
            y.append(r)

    X = np.stack(X) if X else np.zeros((0, lookback, feats.shape[1]), dtype=np.float32)
    y = np.array(y, dtype=np.int64 if classify else np.float32)
    norm = {
        "feat_mean": feats.mean(axis=0).tolist(),
        "feat_std": (feats.std(axis=0) + 1e-9).tolist(),
        "lookback": lookback,
        "horizon": horizon,
        "classify": classify
    }

    return X, y, norm


class EurusdTickDataset(Dataset):
    def __init__(self, symbol: str, start_iso: str, end_iso: str,
                 lookback=120, horizon=10, classify=True, stride=1,
                 db_path: Optional[Path] = None, normalize=True, norm_stats=None,
                 balance="undersample", random_state: int = 42):
        db = db_path or DB_PATH
        raw = load_tob_range(symbol, start_iso, end_iso, db)
        if raw.empty:
            self.X = np.zeros((0, lookback, 5), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.int64 if classify else np.float32)
            self.norm = {"feat_mean": [0] * 5, "feat_std": [1] * 5, "lookback": lookback, "horizon": horizon,
                         "classify": classify}
        else:
            f1s = make_1s_frame(raw)
            f1s = add_features(f1s)
            # print(f1s.iloc[100])
            X, y, norm = build_sequences(f1s, lookback, horizon, stride, classify)
            print("Before Balancing:", class_counts(y))
            # 🔹 Balance here
            if balance != "none" and len(y) > 0:
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

                rng.shuffle(indices)
                X, y = X[indices], y[indices]
                print("After Balancing:", class_counts(y))

            if normalize and len(X):
                mean = np.array(norm_stats["feat_mean"] if norm_stats else norm["feat_mean"], dtype=np.float32)
                std = np.array(norm_stats["feat_std"] if norm_stats else norm["feat_std"], dtype=np.float32)
                X = (X - mean) / std
                self.norm = {"feat_mean": mean.tolist(), "feat_std": std.tolist(),
                             "lookback": lookback, "horizon": horizon, "classify": classify}
            else:
                self.norm = norm

            self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])
