"""Step 10: post-test DESCRIPTIVE refits (never replace the locked models in outputs/models).

All refits reuse the hyperparameters and number of trees fixed in step 07 (chosen on 2025 validation only).
They are post hoc analyses, reported as such; the locked test result (step 08) is not changed by them.

Parts (--parts, default all):
  seeds    explanation stability: 5 refits of the main model on final_train with different seeds;
           SHAP importance on the same 2026 trips; rank agreement with the locked model.
  twin     effect drift: twin models with identical settings trained on Feb-Oct 2025 and Feb-Oct 2026
           (equal training size, 3 replicates each = different seed and 80% subsample).
           Every 5th calendar day is held out of training in both years. Outputs: 2x2 cross-year transfer
           matrix on held-out days; SHAP importance and ALE of both twins on the SAME held-out reference trips.
  rolling  cost of drift: for each 2026 month m, (i) the locked frozen model, (ii) the frozen model with its
           intercept re-estimated on the 28 days that end 7 days before m, (iii) a refit on all dock-to-dock
           trips from 5 Feb 2025 up to 7 days before m. Each is evaluated on month m only.

Run:  python src/step_10_xai_refits.py [--parts seeds,twin,rolling]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, SEED, TARGET, load, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="seeds,twin,rolling")
A = ap.parse_args()
PARTS = A.parts.split(",")
XAI, TAB = OUT / "xai", OUT / "tables"
F = FEATURE_SETS["main"]
BP = json.loads((OUT / "reports" / "07_best_params_main.json").read_text())
PARAMS, N_TREES = BP["lgbm_params"], BP["n_trees"]
EMBARGO = pd.Timedelta(days=7)
locked = lgb.Booster(model_file=str(OUT / "models" / "lgbm_main.txt"))


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit(X, y, seed):
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    return lgb.train(p, lgb.Dataset(X, y), num_boost_round=N_TREES)


def mean_abs_shap(b, X):
    c = np.vstack([b.predict(X.iloc[i:i + 20000], pred_contrib=True) for i in range(0, len(X), 20000)])
    return np.abs(c[:, :-1]).mean(0), c


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


smp = pd.read_parquet(XAI / "sample.parquet")

# ------------------------------------------------------------------ seeds
if "seeds" in PARTS:
    ft = load("final_train", "main")
    ref = smp[smp.year == 2026].groupby("month", group_keys=False).sample(n=2000, random_state=SEED)
    imp = {"locked": mean_abs_shap(locked, ref[F])[0]}
    preds = {"locked": locked.predict(ref[F])}
    for s in range(1, 6):
        b = fit(ft[F], ft[TARGET], 1000 + s)
        imp[f"seed_{s}"], _ = mean_abs_shap(b, ref[F])
        preds[f"seed_{s}"] = b.predict(ref[F])
        tic(f"seed refit {s} done")
    del ft
    I = pd.DataFrame(imp, index=F)
    I.to_csv(XAI / "seed_importance.csv")
    R = I.rank(ascending=False)
    top = lambda c: set(I[c].nlargest(10).index)
    rows = []
    for c in I.columns[1:]:
        rows.append(dict(model=c, spearman_vs_locked=spearmanr(I["locked"], I[c])[0],
                         top10_overlap=len(top("locked") & top(c)) / 10,
                         max_rank_change_top10=int((R.loc[list(top("locked")), c] - R.loc[list(top("locked")), "locked"])
                                                   .abs().max()),
                         pred_corr=np.corrcoef(preds["locked"], preds[c])[0, 1],
                         mean_abs_pred_diff=np.abs(preds["locked"] - preds[c]).mean()))
    st = pd.DataFrame(rows)
    st.to_csv(TAB / "Table14a_seed_stability.csv", index=False)
    tic("seed stability saved")
    print(st.round(4).to_string(index=False))

# ------------------------------------------------------------------ twin models (effect drift)
if "twin" in PARTS:
    def year_data(split, year):
        d = load(split, "main", extra=("started_at", "is_member"))
        d = d[(d.started_at.dt.year == year) & d.started_at.dt.month.between(2, 10)]
        d = d[d.started_at >= pd.Timestamp("2025-02-05")]
        d["held_out"] = (d.started_at.dt.dayofyear % 5 == 0)
        return d.reset_index(drop=True)

    D = {2025: year_data("final_train", 2025), 2026: year_data("test", 2026)}
    n_train = int(0.8 * min((~D[y].held_out).sum() for y in D))
    ho = {y: D[y][D[y].held_out].reset_index(drop=True) for y in D}
    ref = pd.concat([ho[y].assign(year=y, month=ho[y].started_at.dt.month)
                     .groupby("month", group_keys=False).sample(n=2000, random_state=SEED) for y in ho],
                    ignore_index=True)
    ref.to_parquet(XAI / "twin_reference.parquet", index=False)
    transfer, imps, ale_rows = [], [], []
    top = (pd.read_csv(TAB / "Table10_shap_importance.csv")
           .query("model == 'main' and year == 2026 and kind == 'feature'").sort_values("rank")["name"])
    ale_feats = [f for f in top if ref[f].nunique() > 2][:8]
    edges = {f: (np.arange(-0.5, 24.0, 1.0) if f == "hour" else
                 np.unique(np.nanquantile(ref[f].astype(float), np.linspace(0, 1, 21)))) for f in ale_feats}

    def ale(b, X, f, mask):
        e = edges[f]
        x = X[f].values.astype(float)
        k = np.clip(np.searchsorted(e, x, side="left") - 1, 0, len(e) - 2)
        ok = mask & ~np.isnan(x)
        lo, hi = X.copy(), X.copy()
        lo[f], hi[f] = e[k], e[k + 1]
        d = b.predict(hi, raw_score=True) - b.predict(lo, raw_score=True)
        nb = len(e) - 1
        s = np.bincount(k[ok], weights=d[ok], minlength=nb)
        c = np.bincount(k[ok], minlength=nb).astype(float)
        a = np.concatenate([[0], np.cumsum(np.where(c > 0, s / np.maximum(c, 1), 0))])
        return a - (((a[1:] + a[:-1]) / 2) * c).sum() / c.sum()

    for year in (2025, 2026):
        tr = D[year][~D[year].held_out]
        for r in range(3):
            sub = tr.sample(n=n_train, random_state=SEED + r)
            b = fit(sub[F], sub[TARGET], 2000 + 10 * year + r)
            for ey in (2025, 2026):
                m = metrics(ho[ey][TARGET].values, b.predict(ho[ey][F]))
                transfer.append(dict(train_year=year, replicate=r, eval_year=ey, **m))
            mi, _ = mean_abs_shap(b, ref[F])
            imps += [dict(train_year=year, replicate=r, feature=f, mean_abs_shap=v) for f, v in zip(F, mi)]
            for f in ale_feats:
                for sname, msk in (("all", np.ones(len(ref), bool)), ("member", ref.is_member.values == 1),
                                   ("casual", ref.is_member.values == 0)):
                    a = ale(b, ref[F], f, msk)
                    ale_rows += [dict(train_year=year, replicate=r, feature=f, subset=sname, x=x, ale=v)
                                 for x, v in zip(edges[f], a)]
            ale_rows += [dict(train_year=year, replicate=r, feature="__level__", subset="all", x=0,
                              ale=float(np.mean(b.predict(ref[F], raw_score=True))))]
            tic(f"twin {year} replicate {r} done (n_train {n_train:,})")
    tt = pd.DataFrame(transfer)
    tt.to_csv(TAB / "Table14b_twin_transfer.csv", index=False)
    pd.DataFrame(imps).to_csv(XAI / "twin_importance.csv", index=False)
    pd.DataFrame(ale_rows).to_csv(XAI / "twin_ale.csv", index=False)
    print(tt.groupby(["train_year", "eval_year"])[["roc_auc", "log_loss", "cal_intercept", "mean_pred",
                                                   "observed_rate"]].mean().round(4))

# ------------------------------------------------------------------ rolling refit
if "rolling" in PARTS:
    ft = load("final_train", "main", extra=("started_at",))
    te = load("test", "main", extra=("started_at",))
    allx = pd.concat([ft, te], ignore_index=True)
    del ft
    rows = []
    for mo in range(1, 11):
        start = pd.Timestamp(f"2026-{mo:02d}-01")
        end = start + pd.offsets.MonthBegin(1)
        cut = start - EMBARGO
        ev = te[(te.started_at >= start) & (te.started_at < end)]
        y = ev[TARGET].values
        raw_f = locked.predict(ev[F], raw_score=True)
        # (ii) intercept-only recalibration on [cut - 28 days, cut)
        cal = allx[(allx.started_at >= cut - pd.Timedelta(days=28)) & (allx.started_at < cut)]
        rc = locked.predict(cal[F], raw_score=True)
        target = cal[TARGET].mean()
        delta = brentq(lambda d: np.mean(1 / (1 + np.exp(-(rc + d)))) - target, -3, 3)
        # (iii) refit on everything before the cut
        tr = allx[allx.started_at < cut]
        assert cal.started_at.max() < cut and len(cal) > 1000
        assert tr.started_at.max() + EMBARGO <= ev.started_at.min()
        b = fit(tr[F], tr[TARGET], SEED)
        for name, p in (("frozen", 1 / (1 + np.exp(-raw_f))), ("intercept_update", 1 / (1 + np.exp(-(raw_f + delta)))),
                        ("monthly_refit", b.predict(ev[F]))):
            rows.append(dict(month=f"2026-{mo:02d}", variant=name, train_rows=len(tr) if name == "monthly_refit" else None,
                             intercept_shift=delta if name == "intercept_update" else None, **metrics(y, p)))
        tic(f"rolling {start.date()}: train {len(tr):,} rows to {tr.started_at.max().date()}, shift {delta:+.3f}")
    rl = pd.DataFrame(rows)
    rl.to_csv(TAB / "Table15_rolling_refit.csv", index=False)
    w = rl.assign(w=rl["n"]).groupby("variant").apply(
        lambda g: pd.Series({k: np.average(g[k], weights=g["w"]) for k in ("roc_auc", "log_loss", "brier",
                                                                           "cal_intercept")}))
    print(w.round(4))
tic("done")
