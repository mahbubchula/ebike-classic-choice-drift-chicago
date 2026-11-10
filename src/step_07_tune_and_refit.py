"""Step 07: tune LightGBM on validation, then refit on final_train (all of 2025 from 5 Feb).

Search : Optuna TPE, N_TRIALS trials, learning rate 0.1, objective = validation log loss.
         Each trial fits on a fixed random 1,000,000-trip subsample of inner_train (compute limit).
Final  : best parameters, learning rate 0.05, full inner_train, early stopping on validation
         -> number of trees. Then refit on final_train with that fixed number of trees.
Baseline: spline logistic regression refitted on a 500,000-trip subsample of final_train.
The test split is NOT loaded here.

Outputs: outputs/models/lgbm_{set}.txt, logreg_{set}.joblib, outputs/tables/Table7_tuning_{set}.csv,
         outputs/reports/07_best_params_{set}.json

Run:  python src/step_07_tune_and_refit.py --set main
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, SEED, TARGET, load, metrics  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="main", choices=list(FEATURE_SETS))
ap.add_argument("--trials", type=int, default=30)
A = ap.parse_args()
FSET, FEATS = A.set, FEATURE_SETS[A.set]

tr, va = load("inner_train", FSET), load("valid", FSET)
assert tr["started_at"].max() < va["started_at"].min()
sub = tr.sample(n=min(1_000_000, len(tr)), random_state=SEED)
dsub = lgb.Dataset(sub[FEATS], sub[TARGET], free_raw_data=False)
dval = lgb.Dataset(va[FEATS], va[TARGET], reference=dsub, free_raw_data=False)
BASE = dict(objective="binary", metric="binary_logloss", verbosity=-1, seed=SEED, num_threads=0,
            bagging_freq=1, deterministic=True, force_row_wise=True)


def objective(trial):
    p = dict(BASE, learning_rate=0.1,
             num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
             min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 5000, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
             lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
             lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
             min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
             max_bin=trial.suggest_categorical("max_bin", [127, 255, 511]))
    m = lgb.train(p, dsub, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    trial.set_user_attr("best_iter", m.best_iteration)
    return m.best_score["valid_0"]["binary_logloss"]


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=A.trials)
study.trials_dataframe().to_csv(OUT / "tables" / f"Table7_tuning_{FSET}.csv", index=False)
best = dict(BASE, learning_rate=0.05, **study.best_params)
print(f"[{FSET}] search done in {(time.time() - t0) / 60:.1f} min; best valid logloss {study.best_value:.5f}")

# full inner_train at lr 0.05 -> number of trees
dtr = lgb.Dataset(tr[FEATS], tr[TARGET], free_raw_data=False)
dval = lgb.Dataset(va[FEATS], va[TARGET], reference=dtr, free_raw_data=False)
m_in = lgb.train(best, dtr, num_boost_round=10000, valid_sets=[dval], callbacks=[lgb.early_stopping(200, verbose=False)])
n_trees = m_in.best_iteration
p_val = m_in.predict(va[FEATS], num_iteration=n_trees)
val_m = metrics(va[TARGET].values, p_val)

# Decision threshold chosen on validation (maximises balanced accuracy = Youden's J)
grid = np.linspace(0.2, 0.8, 121)
yv = va[TARGET].values
bal = [((p_val >= t) & (yv == 1)).sum() / (yv == 1).sum() + ((p_val < t) & (yv == 0)).sum() / (yv == 0).sum()
       for t in grid]
thr = float(grid[int(np.argmax(bal))])
print(f"[{FSET}] inner model: trees {n_trees}, valid AUC {val_m['roc_auc']:.4f}, logloss {val_m['log_loss']:.4f}, "
      f"threshold {thr:.3f}")

# refit on final_train with fixed settings
ft = load("final_train", FSET)
m_final = lgb.train(best, lgb.Dataset(ft[FEATS], ft[TARGET]), num_boost_round=n_trees)
m_final.save_model(str(OUT / "models" / f"lgbm_{FSET}.txt"))

# baseline logistic regression on final_train subsample
binary = [f for f in FEATS if set(pd.unique(ft[f].dropna())) <= {0, 1}]
contin = [f for f in FEATS if f not in binary]
fsub = ft.sample(n=500_000, random_state=SEED)
lr = Pipeline([("pre", ColumnTransformer([
    ("cont", Pipeline([("imp", SimpleImputer(strategy="median")), ("spl", SplineTransformer(n_knots=5, degree=3)),
                       ("sc", StandardScaler())]), contin),
    ("bin", SimpleImputer(strategy="most_frequent"), binary)])),
    ("lr", LogisticRegression(C=1.0, max_iter=2000))]).fit(fsub[FEATS], fsub[TARGET].values)
joblib.dump(lr, OUT / "models" / f"logreg_{FSET}.joblib")

(OUT / "reports" / f"07_best_params_{FSET}.json").write_text(json.dumps({
    "feature_set": FSET, "features": FEATS, "lgbm_params": best, "n_trees": n_trees, "threshold": thr,
    "search_trials": A.trials, "search_best_valid_logloss": study.best_value,
    "inner_model_valid_metrics": val_m, "final_train_rows": len(ft),
    "logreg_final_train_subsample": 500_000}, indent=2, default=float))
print(f"[{FSET}] saved final LightGBM ({n_trees} trees) and logistic regression; total {(time.time() - t0) / 60:.1f} min")
