"""Step 12: robustness checks (post hoc; the locked models and the locked test result are unchanged).

R1  Month as a feature (main_plus_month): same hyperparameters and number of trees as the main model (tuned on
    the main set only), trained on final_train, evaluated on the 2026 test set; season-matched drift
    decomposition repeated with this model (point estimates).
R2  January 2025 (1 Jan - 4 Feb 2025), before the 5 Feb 2025 price change and excluded from training:
    locked main model applied out of period; observed vs predicted by rider.
R3  Members only and casual only: are conclusions the same within each rider group? (locked model, test set)
R4  Threshold-free summary of the locked main model on test trips excluding round trips (a different trip type).

Output: outputs/tables/Table16_robustness.csv and outputs/reports/12_robustness.json

Run:  python src/step_12_robustness.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, SEED, TARGET, load, metrics  # noqa: E402

BP = json.loads((OUT / "reports" / "07_best_params_main.json").read_text())
locked = lgb.Booster(model_file=str(OUT / "models" / "lgbm_main.txt"))
FM, FMM = FEATURE_SETS["main"], FEATURE_SETS["main_plus_month"]
assert set(FMM) - set(FM) == {"month"}
rows, rep = [], {}


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def add(check, subset, y, p, **kw):
    m = metrics(y, p, threshold=BP["threshold"])
    rows.append(dict(check=check, subset=subset, **m, **kw))
    print(f"{check:28s} {subset:22s} n={m['n']:>9,} AUC {m['roc_auc']:.4f} logloss {m['log_loss']:.4f} "
          f"intercept {m['cal_intercept']:+.3f} mean_pred {m['mean_pred']:.3f} obs {m['observed_rate']:.3f}", flush=True)


# ------------------------------------------------------------------ R1 month as a feature
ft = load("final_train", "main_plus_month")
p = dict(BP["lgbm_params"])
bm = lgb.train(p, lgb.Dataset(ft[FMM], ft[TARGET]), num_boost_round=BP["n_trees"])
bm.save_model(str(OUT / "models" / "robust_lgbm_main_plus_month.txt"))
del ft
te = load("test", "main_plus_month", extra=("started_at", "is_member"))
add("R1 locked main model", "test, all", te[TARGET].values, locked.predict(te[FM]))
add("R1 main + month", "test, all", te[TARGET].values, bm.predict(te[FMM]))
smp = pd.read_parquet(OUT / "xai" / "sample.parquet")
c = bm.predict(smp[FMM], pred_contrib=True)
imp = pd.Series(np.abs(c[smp.year.values == 2026, :-1]).mean(0), index=FMM).sort_values(ascending=False)
rep["R1_top10_importance_main_plus_month"] = imp.head(10).round(4).to_dict()
rep["R1_month_rank"] = int(list(imp.index).index("month") + 1)
share = []
pm = bm.predict(smp[FMM])
for mo in range(2, 11):
    a = (smp.year == 2025) & (smp.month == mo)
    b = (smp.year == 2026) & (smp.month == mo)
    share.append(((logit(pm[b].mean()) - logit(pm[a].mean())), (logit(smp.loc[b, TARGET].mean()) - logit(smp.loc[a, TARGET].mean()))))
share = np.array(share)
rep["R1_share_explained_main_plus_month"] = float(share[:, 0].mean() / share[:, 1].mean())
print("R1 month rank", rep["R1_month_rank"], "share explained", round(rep["R1_share_explained_main_plus_month"], 3))

# ------------------------------------------------------------------ R2 January 2025
ex = load("excluded_pre", "main", extra=("started_at", "is_member"))
assert ex["started_at"].max() < pd.Timestamp("2025-02-05")
pe = locked.predict(ex[FM])
add("R2 January 2025 (pre-price)", "all", ex[TARGET].values, pe)
for m, nm in ((1, "members"), (0, "casual")):
    k = ex["is_member"].values == m
    add("R2 January 2025 (pre-price)", nm, ex[TARGET].values[k], pe[k])

# ------------------------------------------------------------------ R3, R4 on the locked test result
tp = pd.read_parquet(OUT / "predictions" / "test_predictions_main.parquet")
assert len(tp) == len(te) and (tp["started_at"].values == te["started_at"].values).all()
for m, nm in ((1, "members"), (0, "casual")):
    k = tp["is_member"].values == m
    add("R3 rider subgroup", nm, tp["y"].values[k], tp["p_lgbm"].values[k])
te_rt = load("test", "main", extra=("started_at",))
assert (te_rt["started_at"].values == tp["started_at"].values).all()
k = te_rt["round_trip"].values == 0
add("R4 one-way trips only", "test", tp["y"].values[k], tp["p_lgbm"].values[k])

pd.DataFrame(rows).to_csv(OUT / "tables" / "Table16_robustness.csv", index=False)
(OUT / "reports" / "12_robustness.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
