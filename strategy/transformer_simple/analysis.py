# strategy/transformer_simple/analyze_training.py
from __future__ import annotations

import json
# ---- put this at the very top, before importing numpy/torch/matplotlib ----
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicate OpenMP (unsafe but practical)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce oversubscription
# optional once torch is imported: torch.set_num_threads(1); torch.set_num_interop_threads(1)

import numpy as np
import torch
from torch.utils.data import DataLoader

# local imports
from dataset import EurusdTickDataset
from model import TinyTransformer

# -------- Paths (must match train.py) --------
OUT_DIR = Path(__file__).resolve().parents[2] / "models"
CKPT_PATH = OUT_DIR / "eurusd_transformer.pt"
NORM_PATH = OUT_DIR / "eurusd_transformer.norm.json"
CONFIG_PATH = OUT_DIR / "eurusd_transformer_config.json"
METRICS_PATH = OUT_DIR / "eurusd_transformer_metrics.jsonl"

FIG_LOSS = OUT_DIR / "training_loss.png"
FIG_ACC = OUT_DIR / "training_acc.png"
FIG_CM = OUT_DIR / "confusion_matrix.png"
FIG_PHIST = OUT_DIR / "prob_hist.png"
REPORT_MD = OUT_DIR / "eurusd_transformer_report.md"


# -------- Utilities --------
def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# -------- Plot helpers (matplotlib, no style/colors set) --------
def plot_loss(metrics: List[Dict], out_path: Path):
    import matplotlib.pyplot as plt

    if not metrics:
        return
    ep = [m["epoch"] for m in metrics]
    tr = [m["train_loss"] for m in metrics]
    va = [m["val_loss"] for m in metrics]

    plt.figure(figsize=(6, 4))
    plt.plot(ep, tr, label="train")
    plt.plot(ep, va, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_acc(metrics: List[Dict], out_path: Path):
    import matplotlib.pyplot as plt

    if not metrics:
        return
    ep = [m["epoch"] for m in metrics]
    acc = [m.get("val_acc", float("nan")) for m in metrics]

    plt.figure(figsize=(6, 4))
    plt.plot(ep, acc, label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Pred")
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_prob_hist(probs: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt

    # probs: [N, C]
    if probs.size == 0:
        return
    plt.figure(figsize=(6, 4))
    maxp = probs.max(axis=1)
    plt.hist(maxp, bins=20)
    plt.xlabel("max class probability")
    plt.ylabel("count")
    plt.title("Prediction Confidence (max prob)")
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -------- Metrics helpers (no sklearn dependency) --------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # per-class
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        rec = np.where(true_pos > 0, tp / true_pos, 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


# -------- Core evaluation --------
@dataclass
class EvalConfig:
    symbol: str
    start_date: str
    end_date: str
    lookback: int
    horizon: int
    batch_size: int = 512
    label_names: Tuple[str, str, str] = ("down", "flat", "up")


def make_eval_dataset(cfg: EvalConfig, norm_stats: dict | None) -> EurusdTickDataset:
    return EurusdTickDataset(
        symbol=cfg.symbol,
        start_iso=f"{cfg.start_date}T00:00:00+00:00",
        end_iso=f"{cfg.end_date}T23:59:59+00:00",
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        classify=True,
        stride=1,
        normalize=True,
        norm_stats=norm_stats,
    )


@torch.no_grad()
def eval_model(model: torch.nn.Module, dl: DataLoader, device: str) -> Dict[str, np.ndarray | float]:
    model.eval()
    ys, ps, pr = [], [], []

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)

        ys.append(yb.cpu().numpy())
        pr.append(pred.cpu().numpy())
        ps.append(probs.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(pr) if pr else np.zeros((0,), dtype=np.int64)
    probs = np.concatenate(ps) if ps else np.zeros((0, 3), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    return {"y_true": y_true, "y_pred": y_pred, "probs": probs, "accuracy": acc}


# -------- Report writer --------
def write_report(
        cfg_train: dict,
        metrics_rows: List[dict],
        eval_cfg: EvalConfig,
        eval_out: Dict[str, np.ndarray | float],
        cm: np.ndarray,
        prec: np.ndarray,
        rec: np.ndarray,
        f1: np.ndarray,
):
    lines = []
    lines.append("# EURUSD Transformer — Training Analysis\n")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(cfg_train, indent=2))
    lines.append("```")

    if metrics_rows:
        best_row = min(metrics_rows, key=lambda m: m["val_loss"])
        lines.append("## Best Validation")
        lines.append(f"- Epoch: **{best_row['epoch']}**")
        lines.append(f"- Val Loss: **{best_row['val_loss']:.4f}**")
        lines.append(f"- Val Acc: **{best_row.get('val_acc', float('nan')):.3f}**\n")

    lines.append("## Evaluation")
    lines.append(f"- Eval range: `{eval_cfg.start_date}` → `{eval_cfg.end_date}`")
    lines.append(f"- Lookback: `{eval_cfg.lookback}` | Horizon: `{eval_cfg.horizon}`")
    lines.append(f"- Samples: `{len(eval_out['y_true'])}`")
    lines.append(f"- Accuracy: **{eval_out['accuracy']:.3f}**\n")

    lines.append("### Per-class metrics")
    labels = list(eval_cfg.label_names)
    lines.append("| class | precision | recall | f1 |")
    lines.append("|---|---:|---:|---:|")
    for i, name in enumerate(labels):
        lines.append(f"| {name} | {prec[i]:.3f} | {rec[i]:.3f} | {f1[i]:.3f} |")

    lines.append("\n### Confusion matrix")
    lines.append(f"Saved to: `{FIG_CM}`")

    lines.append("\n### Training curves")
    lines.append(f"- Loss: `{FIG_LOSS}`")
    lines.append(f"- Accuracy: `{FIG_ACC}`")

    lines.append("\n### Confidence histogram")
    lines.append(f"- Max-prob histogram: `{FIG_PHIST}`")

    _ensure_dir(REPORT_MD)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report: {REPORT_MD}")


# -------- Main --------
if __name__ == "__main__":
    # Load artifacts
    cfg_train = _load_json(CONFIG_PATH, default={}) or {}
    metrics_rows = _load_metrics_jsonl(METRICS_PATH)
    norm_stats = _load_json(NORM_PATH, default={"feat_mean": [0] * 5, "feat_std": [1] * 5})

    # Eval config (defaults to train dates; change here if you want a wider eval window)
    eval_cfg = EvalConfig(
        symbol=cfg_train.get("symbol", "EUR"),
        start_date=cfg_train.get("start_date", "2025-08-08"),
        end_date=cfg_train.get("end_date", "2025-08-08"),
        lookback=int(cfg_train.get("lookback", 120)),
        horizon=int(cfg_train.get("horizon", 10)),
        batch_size=512,
        label_names=tuple(cfg_train.get("label_names", ("down", "flat", "up"))),
    )

    # Dataset + loader
    ds_eval = make_eval_dataset(eval_cfg, norm_stats)
    if len(ds_eval) == 0:
        raise SystemExit("No evaluation samples. Adjust date range.")

    dl_eval = DataLoader(ds_eval, batch_size=eval_cfg.batch_size, shuffle=False)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(in_dim=5, num_classes=3).to(device)
    sd = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(sd["state_dict"])

    # Evaluate
    eval_out = eval_model(model, dl_eval, device)
    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]
    probs = eval_out["probs"]

    # Metrics / figures
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    prec, rec, f1 = precision_recall_f1(cm)

    plot_loss(metrics_rows, FIG_LOSS)
    plot_acc(metrics_rows, FIG_ACC)
    plot_confusion_matrix(cm, list(eval_cfg.label_names), FIG_CM)
    plot_prob_hist(probs, FIG_PHIST)

    # Report
    write_report(cfg_train, metrics_rows, eval_cfg, eval_out, cm, prec, rec, f1)

    print("done.")
