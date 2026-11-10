"""Shared helpers: paths, data loading and metrics."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             log_loss, matthews_corrcoef, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "study.yaml").read_text())
FEATURE_SETS = json.loads((ROOT / "configs" / "features.json").read_text())
SEED, TARGET = CFG["seed"], CFG["target"]
SPLITS = ROOT / "data" / "processed" / "splits"
OUT = ROOT / "outputs"
for sub in ("models", "tables", "reports", "logs", "predictions"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)


def load(split, feature_set="main", extra=("started_at",)):
    """Load one split with the sample that belongs to a feature set.

    main / main_plus_month : dock-to-dock trips only (both ends measured at a station)
    origin_only            : all dock-start trips, no destination information
    The test split must be requested explicitly by the final evaluation script.
    """
    feats = FEATURE_SETS[feature_set]
    cols = [TARGET, "end_dock", *extra, *feats]
    d = pd.read_parquet(SPLITS / f"{split}.parquet", columns=list(dict.fromkeys(cols)))
    if feature_set != "origin_only":
        d = d[d["end_dock"] == 1]
    return d.reset_index(drop=True)


def calibration_slope_intercept(y, p, eps=1e-6):
    from sklearn.linear_model import LogisticRegression
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def metrics(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    yhat = (p >= threshold).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "brier": brier_score_loss(y, p),
            "balanced_acc": balanced_accuracy_score(y, yhat), "f1": f1_score(y, yhat),
            "mcc": matthews_corrcoef(y, yhat), "cal_slope": slope, "cal_intercept": intercept,
            "mean_pred": float(p.mean()), "observed_rate": float(y.mean()), "threshold": threshold, "n": int(len(y))}
