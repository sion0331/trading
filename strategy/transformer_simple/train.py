from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import EurusdTickDataset
from model import TinyTransformer

# ---------- Paths ----------
OUT_DIR = Path(__file__).resolve().parents[2] / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = OUT_DIR / "eurusd_transformer_metrics.jsonl"
CONFIG_PATH = OUT_DIR / "eurusd_transformer_config.json"
CKPT_PATH = OUT_DIR / "eurusd_transformer.pt"
NORM_PATH = OUT_DIR / "eurusd_transformer.norm.json"


# ---------- Config ----------
@dataclass
class TrainConfig:
    symbol: str = "EUR"
    start_date: str = "2025-08-08"
    end_date: str = "2025-08-08"
    lookback: int = 120
    horizon: int = 10
    batch_size: int = 256
    epochs: int = 10
    val_frac: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    # model
    in_dim: int = 5
    num_classes: int = 3
    # notes
    label_names: tuple[str, ...] = ("down", "flat", "up")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def save_config_once(cfg: TrainConfig, path: Path = CONFIG_PATH) -> None:
    if not path.exists():
        path.write_text(cfg.to_json(), encoding="utf-8")


def log_metrics(epoch: int, train_loss: float, val_loss: float, val_acc: float,
                metrics_path: Path = METRICS_PATH) -> None:
    row = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "time": float(time.time()),
        "saved_path": str(CKPT_PATH) if CKPT_PATH.exists() else "",
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------- Data / Model builders ----------
def make_dataset(cfg: TrainConfig) -> EurusdTickDataset:
    ds = EurusdTickDataset(
        symbol=cfg.symbol,
        start_iso=f"{cfg.start_date}T00:00:00+00:00",
        end_iso=f"{cfg.end_date}T23:59:59+00:00",
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        classify=True,
    )
    if len(ds) < 1000:
        raise SystemExit(f"Not enough samples ({len(ds)}). Extend date range.")
    return ds


def make_loaders(ds: EurusdTickDataset, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(cfg.seed)
    n_val = int(cfg.val_frac * len(ds))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    return dl_train, dl_val


def build_model(cfg: TrainConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(in_dim=cfg.in_dim, num_classes=cfg.num_classes)
    return model.to(device)


# ---------- Train / Eval ----------
def _ensure_ce_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Make sure labels are in {0,1,2}. If dataset yields {-1,0,1}, shift by +1.
    """
    if torch.min(y) < 0:
        y = y + 1
    return y


def train_epoch(model: torch.nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: str) -> float:
    model.train()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(device), _ensure_ce_labels(y.to(device))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss.item()) * len(x)
    return tot / max(1, len(loader.dataset))


@torch.no_grad()
def eval_epoch(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    tot, correct = 0.0, 0
    n = max(1, len(loader.dataset))
    for x, y in loader:
        x, y = x.to(device), _ensure_ce_labels(y.to(device))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        tot += float(loss.item()) * len(x)
        pred = logits.argmax(-1)
        correct += int((pred == y).sum().item())
    return tot / n, correct / n


# ---------- Orchestration ----------
def run_training(cfg: TrainConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", "gpu" if device == "cuda" else device)

    # config snapshot (once)
    save_config_once(cfg)

    # data
    ds = make_dataset(cfg)
    dl_train, dl_val = make_loaders(ds, cfg)

    # model / opt
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = np.inf
    for ep in range(1, cfg.epochs + 1):
        tr = train_epoch(model, dl_train, opt, device)
        va, acc = eval_epoch(model, dl_val, device)

        print(f"epoch {ep:02d} | train {tr:.4f} | val {va:.4f} | acc {acc:.3f}")
        log_metrics(ep, tr, va, acc)

        if va < best:
            best = va
            torch.save({"state_dict": model.state_dict()}, CKPT_PATH)
            with open(NORM_PATH, "w", encoding="utf-8") as f:
                json.dump(getattr(ds, "norm", {}), f)
            print("saved:", CKPT_PATH)


# ---------- Main ----------
if __name__ == "__main__":
    cfg = TrainConfig(
        symbol="EUR",
        start_date="2025-08-08",
        end_date="2025-08-08",
        lookback=120,
        horizon=10,
        batch_size=256,
        epochs=10,
        val_frac=0.10,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        in_dim=5,
        num_classes=3,
        label_names=("down", "flat", "up"),
    )
    run_training(cfg)
