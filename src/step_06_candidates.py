"""Step 06: untuned candidate models. Fit on inner_train, score on valid.

The test split is NOT loaded here.
Models: Dummy (prior), logistic regression (splines), LightGBM, XGBoost, CatBoost.
Boosting models use early stopping on the validation set (allowed: validation is for model selection).
Logistic regression is fitted on a random 500,000-trip subsample of inner_train (compute limit);
all boosting models use the full inner_train.

Outputs: outputs/tables/Table6_candidate_models_valid.csv, predictions/valid_candidates.parquet,
         logs/06_candidates.log

Run:  python src/step_06_candidates.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, SEED, TARGET, load, metrics  # noqa: E402

from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--set", default="main", choices=list(FEATURE_SETS))
FSET = ap.parse_args().set
FEATS = FEATURE_SETS[FSET]
tr, va = load("inner_train", FSET), load("valid", FSET)
assert tr["started_at"].max() < va["started_at"].min()
Xtr, ytr, Xva, yva = tr[FEATS], tr[TARGET].values, va[FEATS], va[TARGET].values
print(f"inner_train {Xtr.shape}, valid {Xva.shape}, features {len(FEATS)}")

binary = [f for f in FEATS if set(pd.unique(Xtr[f].dropna())) <= {0, 1}]
contin = [f for f in FEATS if f not in binary]
results, preds = [], {"y": yva}


def record(name, p, secs, note=""):
    m = metrics(yva, p)
    m.update(model=name, fit_seconds=round(secs, 1), note=note)
    results.append(m)
    preds[name] = p.astype("float32")
    print(f"{name:<10} AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  logloss {m['log_loss']:.4f}  "
          f"Brier {m['brier']:.4f}  ({secs:.0f}s)")


# Dummy: predicts the inner-train prior
t = time.time()
dm = DummyClassifier(strategy="prior").fit(Xtr, ytr)
record("Dummy", dm.predict_proba(Xva)[:, 1], time.time() - t)

# Logistic regression with natural splines on continuous features (preprocessing fitted on training only)
t = time.time()
sub = tr.sample(n=500_000, random_state=SEED)
pre = ColumnTransformer([
    ("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("spl", SplineTransformer(n_knots=5, degree=3)),
                       ("sc", StandardScaler())]), contin),
    ("bin", SimpleImputer(strategy="most_frequent"), binary)])
lr = Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=2000))])
lr.fit(sub[FEATS], sub[TARGET].values)
record("LogReg", lr.predict_proba(Xva)[:, 1], time.time() - t, "splines, 500k subsample")

# LightGBM
t = time.time()
lgb = LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
record("LightGBM", lgb.predict_proba(Xva)[:, 1], time.time() - t, f"best_iter={lgb.best_iteration_}")

# XGBoost
t = time.time()
xgb = XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", eval_metric="logloss", early_stopping_rounds=100,
                    random_state=SEED, n_jobs=-1)
xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
record("XGBoost", xgb.predict_proba(Xva)[:, 1], time.time() - t, f"best_iter={xgb.best_iteration}")

# CatBoost
t = time.time()
cb = CatBoostClassifier(iterations=5000, learning_rate=0.08, depth=8, loss_function="Logloss",
                        random_seed=SEED, thread_count=-1, verbose=0, od_type="Iter", od_wait=100)
cb.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
record("CatBoost", cb.predict_proba(Xva)[:, 1], time.time() - t, f"best_iter={cb.get_best_iteration()}")

res = pd.DataFrame(results).set_index("model").round(4)
res.to_csv(OUT / "tables" / f"Table6_candidate_models_valid_{FSET}.csv")
pd.DataFrame(preds).to_parquet(OUT / "predictions" / f"valid_candidates_{FSET}.parquet", index=False)
print(res[["roc_auc", "pr_auc", "log_loss", "brier", "cal_slope", "mean_pred", "observed_rate", "note"]].to_string())
