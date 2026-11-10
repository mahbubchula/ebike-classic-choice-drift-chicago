"""Step 08: ONE-TIME evaluation on the 2026 test set.

Test-once guard: on first run, the SHA-256 of every model file is written to
outputs/reports/08_TEST_LOCK.json. Later runs are allowed only if the models are byte-identical
(re-computation), so no model can be changed after the test results have been seen.

Reports for LightGBM and the spline logistic regression, per feature set:
  overall metrics with day-block bootstrap 95% CIs (days resampled with replacement),
  paired bootstrap CI for the AUC difference LightGBM - logistic regression,
  metrics by month, by rider type and by price regime (before / from 16 March 2026),
  calibration in the large by month.
The decision threshold comes from validation (step 07), never from test data.

Run:  python src/step_08_test_once.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, SEED, TARGET, load, metrics  # noqa: E402

SETS = ["main", "main_plus_supply", "origin_only"]
N_BOOT = 200
REGIME_CHANGE = pd.Timestamp("2026-03-16")
REP, TAB, PRED = OUT / "reports", OUT / "tables", OUT / "predictions"

# ------------------------------------------------------------------ test-once guard
files = sorted([OUT / "models" / f"lgbm_{s}.txt" for s in SETS] + [OUT / "models" / f"logreg_{s}.joblib" for s in SETS])
hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
lock = REP / "08_TEST_LOCK.json"
if lock.exists():
    old = json.loads(lock.read_text())["model_sha256"]
    if old != hashes:
        sys.exit("STOP: models changed after the test set was evaluated. Test results must not drive model changes.")
    print("Lock found and models unchanged: recomputing identical test results.")
else:
    lock.write_text(json.dumps({"first_evaluated": pd.Timestamp.now().isoformat(timespec="seconds"),
                                "model_sha256": hashes}, indent=2))
    print("Test lock written. This is the first evaluation of the test set.")

rng = np.random.default_rng(SEED)


def boot_ci(y, p_dict, day):
    """Day-block bootstrap: resample days, keep all trips of a sampled day."""
    days = np.unique(day)
    idx_by_day = pd.Series(np.arange(len(day))).groupby(day).apply(np.array)
    out = {k: {"roc_auc": [], "pr_auc": [], "log_loss": [], "brier": []} for k in p_dict}
    diff = []
    for _ in range(N_BOOT):
        idx = np.concatenate(idx_by_day.loc[rng.choice(days, len(days), replace=True)].values)
        yb = y[idx]
        aucs = {}
        for k, p in p_dict.items():
            pb = p[idx]
            aucs[k] = roc_auc_score(yb, pb)
            out[k]["roc_auc"].append(aucs[k])
            out[k]["pr_auc"].append(average_precision_score(yb, pb))
            out[k]["log_loss"].append(log_loss(yb, np.clip(pb, 1e-6, 1 - 1e-6)))
            out[k]["brier"].append(brier_score_loss(yb, pb))
        diff.append(aucs["LightGBM"] - aucs["LogReg"])
    ci = {k: {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in d.items()}
          for k, d in out.items()}
    return ci, (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))


overall, groups = [], []
for s in SETS:
    feats = FEATURE_SETS[s]
    te = load("test", s, extra=("started_at", "is_member"))
    params = json.loads((REP / f"07_best_params_{s}.json").read_text())
    thr = params["threshold"]
    booster = lgb.Booster(model_file=str(OUT / "models" / f"lgbm_{s}.txt"))
    lr = joblib.load(OUT / "models" / f"logreg_{s}.joblib")
    y = te[TARGET].values
    p = {"LightGBM": booster.predict(te[feats]), "LogReg": lr.predict_proba(te[feats])[:, 1]}
    day = te["started_at"].dt.normalize().values
    ci, diff_ci = boot_ci(y, p, day)
    for k, pk in p.items():
        m = metrics(y, pk, threshold=thr)
        m.update(feature_set=s, model=k, **{f"{mm}_ci95": ci[k][mm] for mm in ci[k]})
        if k == "LightGBM":
            m["auc_diff_vs_logreg_ci95"] = diff_ci
        overall.append(m)
    te["month"] = te["started_at"].dt.to_period("M").astype(str)
    te["regime"] = np.where(te["started_at"] < REGIME_CHANGE, "before 16 Mar 2026", "from 16 Mar 2026")
    te["rider"] = np.where(te["is_member"] == 1, "member", "casual")
    for gcol in ("month", "rider", "regime"):
        for gval, idx in te.groupby(gcol).indices.items():
            for k, pk in p.items():
                m = metrics(y[idx], pk[idx], threshold=thr)
                m.update(feature_set=s, model=k, group=gcol, value=gval)
                groups.append(m)
    pd.DataFrame({"started_at": te["started_at"], "y": y, "is_member": te["is_member"],
                  "p_lgbm": p["LightGBM"].astype("float32"), "p_logreg": p["LogReg"].astype("float32")}
                 ).to_parquet(PRED / f"test_predictions_{s}.parquet", index=False)
    print(f"[{s}] LightGBM test AUC {overall[-2]['roc_auc']:.4f} {ci['LightGBM']['roc_auc']}, "
          f"LogReg {overall[-1]['roc_auc']:.4f}; AUC diff CI {diff_ci}")

ov = pd.DataFrame(overall)
ov.to_csv(TAB / "Table8_test_overall.csv", index=False)
pd.DataFrame(groups).to_csv(TAB / "Table9_test_by_group.csv", index=False)
cols = ["feature_set", "model", "roc_auc", "roc_auc_ci95", "pr_auc", "log_loss", "brier", "balanced_acc", "f1",
        "mcc", "cal_slope", "cal_intercept", "mean_pred", "observed_rate"]
print(ov[cols].round(4).to_string(index=False))
