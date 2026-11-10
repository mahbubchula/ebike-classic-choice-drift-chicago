"""Step 09: explain the LOCKED models (no refitting, no change to any model file).

Parts (--parts, default all):
  sample   month-stratified samples: 2025 = final_train Feb-Oct, 2026 = test Jan-Oct (N_PER_MONTH per month)
  shap     TreeSHAP contributions (LightGBM pred_contrib, exact and additive) for main and main_plus_supply
  global   mean |SHAP| per feature and per feature group, day-block bootstrap 95% CI
  decomp   season-matched drift decomposition 2025 -> 2026 (Feb-Oct, months weighted equally):
           change in mean log-odds = sum over features of change in mean SHAP (exact, model fixed);
           unexplained = observed change in log-odds of the share - change in log-odds of the mean prediction
  ale      first-order ALE (raw log-odds) for the top continuous features, by year and rider type,
           day-block bootstrap bands (local effects computed once, days resampled)
  inter    SHAP interaction values on a 3,000-trip 2026 sub-sample
  perm     permutation importance on the 2026 sample (features and groups; permuted within month)
  local    eight local explanation cases
  resid    where the 2026 under-prediction concentrates (observed - predicted, full test predictions)

The test labels are used only to DESCRIBE the locked model; nothing here feeds back into model choice.
SHAP/ALE describe model behaviour, not causal effects.

Run:  python src/step_09_xai_locked.py [--parts shap,decomp]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, ROOT, SEED, TARGET, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="sample,shap,global,decomp,ale,inter,perm,local,resid")
A = ap.parse_args()
PARTS = A.parts.split(",")

XAI, TAB = OUT / "xai", OUT / "tables"
XAI.mkdir(exist_ok=True)
N_PER_MONTH, N_BOOT, MATCHED = 10_000, 500, list(range(2, 11))
MODELS = ["main", "main_plus_supply"]
rng = np.random.default_rng(SEED)
thr = json.loads((OUT / "reports" / "07_best_params_main.json").read_text())["threshold"]


def group_of(f):
    if f in ("hour", "is_weekend", "is_holiday", "is_daylight"):
        return "Time"
    if f == "is_member":
        return "Rider"
    if f in ("dist_km", "round_trip"):
        return "Trip"
    if f.startswith("o_"):
        return "Origin area"
    if f.startswith("d_"):
        return "Destination area"
    if f.startswith("sup_"):
        return "Station supply"
    return "Weather"


def booster(s):
    return lgb.Booster(model_file=str(OUT / "models" / f"lgbm_{s}.txt"))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ sample
SAMPLE = XAI / "sample.parquet"
if "sample" in PARTS:
    extra = ("started_at", "ride_id")
    parts = []
    for split, year, months in (("final_train", 2025, MATCHED), ("test", 2026, range(1, 11))):
        d = load(split, "main_plus_supply", extra=extra)
        d = d[d["started_at"].dt.year == year]
        d["month"] = d["started_at"].dt.month
        d = d[d["month"].isin(months)]
        d = d.groupby("month", group_keys=False).sample(n=N_PER_MONTH, random_state=SEED)
        d["year"] = year
        parts.append(d)
    smp = pd.concat(parts, ignore_index=True)
    smp["day"] = smp["started_at"].dt.normalize()
    assert smp["ride_id"].is_unique
    assert smp.loc[smp.year == 2025, "started_at"].min() >= pd.Timestamp("2025-02-05")
    smp.to_parquet(SAMPLE, index=False)
    tic(f"sample: {len(smp):,} trips; e-bike share 2025 {smp[smp.year == 2025][TARGET].mean():.3f}, "
        f"2026 {smp[smp.year == 2026][TARGET].mean():.3f}")
smp = pd.read_parquet(SAMPLE)

# ------------------------------------------------------------------ shap
if "shap" in PARTS:
    for s in MODELS:
        F, b = FEATURE_SETS[s], booster(s)
        X = smp[F]
        c = np.vstack([b.predict(X.iloc[i:i + 20000], pred_contrib=True) for i in range(0, len(X), 20000)])
        raw = b.predict(X, raw_score=True)
        err = np.abs(c.sum(1) - raw).max()
        assert err < 1e-8, err                      # additivity: contributions sum to the raw score
        sh = pd.DataFrame(c[:, :-1].astype("float32"), columns=[f"shap_{f}" for f in F])
        sh["base"], sh["raw"] = c[:, -1], raw
        sh["p"] = 1 / (1 + np.exp(-raw))
        sh.to_parquet(XAI / f"shap_{s}.parquet", index=False)
        tic(f"shap {s}: {sh.shape}, additivity error {err:.1e}, base value {c[0, -1]:.4f}")


def load_shap(s):
    return pd.read_parquet(XAI / f"shap_{s}.parquet")


def day_weights(days, strata, reps):
    """Stratified day-block bootstrap weights (reps x n_days): resample days within each stratum."""
    W = np.zeros((reps, len(days)))
    for st in np.unique(strata):
        idx = np.where(strata == st)[0]
        draw = rng.choice(idx, size=(reps, len(idx)), replace=True)
        for r in range(reps):
            W[r] += np.bincount(draw[r], minlength=len(days))
    return W


# ------------------------------------------------------------------ global importance
if "global" in PARTS:
    rows = []
    for s in MODELS:
        F, sh = FEATURE_SETS[s], load_shap(s)
        for year in (2025, 2026):
            m = (smp["year"] == year).values
            A_ = np.abs(sh.loc[m, [f"shap_{f}" for f in F]].values)
            G = {g: [i for i, f in enumerate(F) if group_of(f) == g] for g in dict.fromkeys(map(group_of, F))}
            Ag = np.column_stack([np.abs(sh.loc[m, [f"shap_{F[i]}" for i in ix]].values.sum(1)) for ix in G.values()])
            day = smp.loc[m, "day"].values
            ud, inv = np.unique(day, return_inverse=True)
            Sd, Sg, Nd = (np.zeros((len(ud), A_.shape[1])), np.zeros((len(ud), Ag.shape[1])),
                          np.bincount(inv).astype(float))
            np.add.at(Sd, inv, A_)
            np.add.at(Sg, inv, Ag)
            W = day_weights(ud, pd.DatetimeIndex(ud).month.values, 300)
            bF, bG = (W @ Sd) / (W @ Nd)[:, None], (W @ Sg) / (W @ Nd)[:, None]
            for lab, est, bb, kind in ((F, A_.mean(0), bF, "feature"), (list(G), Ag.mean(0), bG, "group")):
                for i, name in enumerate(lab):
                    rows.append(dict(model=s, year=year, kind=kind, name=name,
                                     group=group_of(name) if kind == "feature" else name,
                                     mean_abs_shap=est[i], lo=np.percentile(bb[:, i], 2.5),
                                     hi=np.percentile(bb[:, i], 97.5)))
    gi = pd.DataFrame(rows)
    gi["rank"] = gi.groupby(["model", "year", "kind"])["mean_abs_shap"].rank(ascending=False).astype(int)
    gi.sort_values(["model", "kind", "year", "rank"]).to_csv(TAB / "Table10_shap_importance.csv", index=False)
    tic("global importance saved (Table10)")
    print(gi[(gi.model == "main") & (gi.year == 2026) & (gi["rank"] <= 12)]
          .sort_values(["kind", "rank"])[["kind", "name", "mean_abs_shap", "lo", "hi"]].round(4).to_string(index=False))

# ------------------------------------------------------------------ drift decomposition
if "decomp" in PARTS:
    out, monthly, summ = [], [], []
    for s in MODELS:
        F, sh = FEATURE_SETS[s], load_shap(s)
        cols = [f"shap_{f}" for f in F]
        for subset, mask in (("all", np.ones(len(smp), bool)), ("member", smp["is_member"].values == 1),
                             ("casual", smp["is_member"].values == 0)):
            m = mask & smp["month"].isin(MATCHED).values
            d = smp.loc[m, ["year", "month", "day", TARGET]].reset_index(drop=True)
            Phi = sh.loc[m, cols].values.astype(float)
            P, R = sh.loc[m, "p"].values, sh.loc[m, "raw"].values
            ud, inv = np.unique(d["day"].values, return_inverse=True)
            nD = len(ud)
            S_phi = np.zeros((nD, len(F)))
            np.add.at(S_phi, inv, Phi)
            S_n = np.bincount(inv, minlength=nD).astype(float)
            S_y = np.bincount(inv, weights=d[TARGET].values, minlength=nD)
            S_p = np.bincount(inv, weights=P, minlength=nD)
            S_r = np.bincount(inv, weights=R, minlength=nD)
            dyear = pd.DatetimeIndex(ud).year.values
            dmon = pd.DatetimeIndex(ud).month.values
            W = np.vstack([np.ones((1, nD)), day_weights(ud, dyear * 100 + dmon, N_BOOT)])  # row 0 = estimate

            def stat(Wm):
                """Average over months (equal weight) of 2026 minus 2025 means; Wm: reps x days."""
                acc = {k: 0 for k in ("phi", "obs_pp", "pred_pp", "L_obs", "L_pred", "L_raw")}
                for mo in MATCHED:
                    vals = {}
                    for yr in (2025, 2026):
                        sel = (dyear == yr) & (dmon == mo)
                        w = Wm[:, sel]
                        n = w @ S_n[sel]
                        vals[yr] = dict(phi=(w @ S_phi[sel]) / n[:, None], y=(w @ S_y[sel]) / n,
                                        p=(w @ S_p[sel]) / n, r=(w @ S_r[sel]) / n)
                    a, b = vals[2025], vals[2026]
                    acc["phi"] = acc["phi"] + (b["phi"] - a["phi"]) / len(MATCHED)
                    acc["obs_pp"] = acc["obs_pp"] + 100 * (b["y"] - a["y"]) / len(MATCHED)
                    acc["pred_pp"] = acc["pred_pp"] + 100 * (b["p"] - a["p"]) / len(MATCHED)
                    acc["L_obs"] = acc["L_obs"] + (logit(b["y"]) - logit(a["y"])) / len(MATCHED)
                    acc["L_pred"] = acc["L_pred"] + (logit(b["p"]) - logit(a["p"])) / len(MATCHED)
                    acc["L_raw"] = acc["L_raw"] + (b["r"] - a["r"]) / len(MATCHED)
                return acc

            st = stat(W)
            ci = lambda v: (np.percentile(v[1:], 2.5), np.percentile(v[1:], 97.5))
            for j, f in enumerate(F):
                lo, hi = ci(st["phi"][:, j])
                out.append(dict(model=s, subset=subset, kind="feature", name=f, group=group_of(f),
                                delta_logodds=st["phi"][0, j], lo=lo, hi=hi))
            for g in dict.fromkeys(map(group_of, F)):
                ix = [j for j, f in enumerate(F) if group_of(f) == g]
                v = st["phi"][:, ix].sum(1)
                out.append(dict(model=s, subset=subset, kind="group", name=g, group=g, delta_logodds=v[0],
                                lo=ci(v)[0], hi=ci(v)[1]))
            unexpl = st["L_obs"] - st["L_pred"]
            share = st["L_pred"] / st["L_obs"]
            rec = dict(model=s, subset=subset, n=int(m.sum()))
            for k, v in (("observed_change_pp", st["obs_pp"]), ("predicted_change_pp", st["pred_pp"]),
                         ("observed_change_logodds", st["L_obs"]), ("explained_change_logodds", st["L_pred"]),
                         ("sum_shap_change_logodds", st["L_raw"]), ("unexplained_change_logodds", unexpl),
                         ("share_explained", share)):
                rec[k], rec[k + "_lo"], rec[k + "_hi"] = v[0], *ci(v)
            summ.append(rec)
            # monthly group decomposition (point estimates)
            for mo in MATCHED:
                row = dict(model=s, subset=subset, month=mo)
                means = {}
                for yr in (2025, 2026):
                    sel = ((d["year"] == yr) & (d["month"] == mo)).values
                    means[yr] = (Phi[sel].mean(0), d.loc[sel, TARGET].mean(), P[sel].mean())
                for g in dict.fromkeys(map(group_of, F)):
                    ix = [j for j, f in enumerate(F) if group_of(f) == g]
                    row[g] = means[2026][0][ix].sum() - means[2025][0][ix].sum()
                row["observed_logodds"] = logit(means[2026][1]) - logit(means[2025][1])
                row["explained_logodds"] = logit(means[2026][2]) - logit(means[2025][2])
                row["observed_pp"] = 100 * (means[2026][1] - means[2025][1])
                monthly.append(row)
    pd.DataFrame(out).to_csv(TAB / "Table11_drift_decomposition.csv", index=False)
    pd.DataFrame(summ).to_csv(TAB / "Table11a_drift_summary.csv", index=False)
    pd.DataFrame(monthly).to_csv(XAI / "drift_decomposition_monthly.csv", index=False)
    # covariate drift: month-matched standardised mean difference of each feature
    cd = []
    for f in FEATURE_SETS["main_plus_supply"]:
        diffs, pooled = [], []
        for mo in MATCHED:
            a = smp.loc[(smp.year == 2025) & (smp.month == mo), f].astype(float)
            b = smp.loc[(smp.year == 2026) & (smp.month == mo), f].astype(float)
            diffs.append(b.mean() - a.mean())
            pooled.append(np.sqrt((a.var() + b.var()) / 2))
        cd.append(dict(feature=f, group=group_of(f), mean_2025=smp.loc[smp.year == 2025, f].mean(),
                       mean_2026_matched=smp.loc[(smp.year == 2026) & smp.month.isin(MATCHED), f].mean(),
                       smd=np.mean(diffs) / np.mean(pooled)))
    pd.DataFrame(cd).to_csv(TAB / "Table11b_covariate_drift.csv", index=False)
    tic("drift decomposition saved (Table11, 11a, 11b)")
    print(pd.DataFrame(summ)[["model", "subset", "observed_change_pp", "predicted_change_pp",
                              "observed_change_logodds", "explained_change_logodds", "share_explained",
                              "share_explained_lo", "share_explained_hi"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ ALE
if "ale" in PARTS:
    F, b = FEATURE_SETS["main"], booster("main")
    gi = pd.read_csv(TAB / "Table10_shap_importance.csv")
    gi = gi[(gi.model == "main") & (gi.year == 2026) & (gi.kind == "feature")].sort_values("rank")
    cont = [f for f in gi["name"] if smp[f].nunique() > 2][:10]
    X = smp[F].reset_index(drop=True)
    ud, inv = np.unique(smp["day"].values, return_inverse=True)
    W = np.vstack([np.ones((1, len(ud))), day_weights(ud, pd.DatetimeIndex(ud).year.values * 100 +
                                                       pd.DatetimeIndex(ud).month.values, 200)])
    subsets = {"2025": smp.year.values == 2025, "2026": smp.year.values == 2026,
               "2025 member": (smp.year.values == 2025) & (smp.is_member.values == 1),
               "2025 casual": (smp.year.values == 2025) & (smp.is_member.values == 0),
               "2026 member": (smp.year.values == 2026) & (smp.is_member.values == 1),
               "2026 casual": (smp.year.values == 2026) & (smp.is_member.values == 0)}
    rows = []
    for f in cont:
        x = X[f].values.astype(float)
        ok = ~np.isnan(x)
        edges = np.unique(np.nanquantile(x, np.linspace(0, 1, 21)))
        if f == "hour":
            edges = np.arange(-0.5, 24.0, 1.0)
        k = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)
        k[x == edges[0]] = 0
        lo_, hi_ = X.copy(), X.copy()
        lo_[f], hi_[f] = edges[k], edges[k + 1]
        delta = b.predict(hi_, raw_score=True) - b.predict(lo_, raw_score=True)   # local effect per trip
        nb = len(edges) - 1
        for name, msk in subsets.items():
            mm = msk & ok
            # per-day x per-bin sums so the day bootstrap is a matrix product
            S = np.zeros((len(ud), nb))
            C = np.zeros((len(ud), nb))
            np.add.at(S, (inv[mm], k[mm]), delta[mm])
            np.add.at(C, (inv[mm], k[mm]), 1)
            WS, WC = W @ S, W @ C
            eff = np.where(WC > 0, WS / np.maximum(WC, 1e-12), 0.0)
            ale = np.concatenate([np.zeros((W.shape[0], 1)), np.cumsum(eff, 1)], 1)       # at edges
            mid = (ale[:, 1:] + ale[:, :-1]) / 2
            ale = ale - ((mid * WC).sum(1) / WC.sum(1))[:, None]                          # centre
            for i, e in enumerate(edges):
                rows.append(dict(feature=f, subset=name, x=e, ale=ale[0, i],
                                 lo=np.percentile(ale[1:, i], 2.5), hi=np.percentile(ale[1:, i], 97.5),
                                 n_bin=C[:, min(i, nb - 1)].sum()))
        tic(f"ALE {f} done")
    pd.DataFrame(rows).to_csv(XAI / "ale_main.csv", index=False)
    # distribution of each ALE feature for the support strip
    smp[cont + ["year", "is_member"]].sample(40000, random_state=SEED).to_parquet(XAI / "ale_support.parquet")

# ------------------------------------------------------------------ interactions
if "inter" in PARTS:
    import shap
    from joblib import Parallel, delayed
    F = FEATURE_SETS["main"]
    sub = smp[smp.year == 2026].groupby("month", group_keys=False).sample(n=300, random_state=SEED)
    chunks = np.array_split(sub[F], 10)

    def iv(chunk):
        return shap.TreeExplainer(booster("main")).shap_interaction_values(chunk)

    t0 = time.time()
    I = np.concatenate(Parallel(n_jobs=5)(delayed(iv)(c) for c in chunks)).astype("float32")
    np.savez_compressed(XAI / "shap_interactions_main.npz", values=I, ride_id=sub["ride_id"].values,
                        X=sub[F].values.astype("float32"), features=np.array(F), is_member=sub["is_member"].values)
    M = np.abs(I).mean(0)
    off = M.copy()
    np.fill_diagonal(off, 0)
    pairs = [dict(f1=F[i], f2=F[j], mean_abs_interaction=2 * off[i, j],
                  share_of_total=2 * off[i, j] / M.sum()) for i in range(len(F)) for j in range(i + 1, len(F))]
    pt = pd.DataFrame(pairs).sort_values("mean_abs_interaction", ascending=False)
    pt.to_csv(TAB / "Table12a_shap_interactions.csv", index=False)
    tic(f"interactions ({len(sub)} trips) in {(time.time() - t0) / 60:.1f} min; main-effect share "
        f"{np.trace(M) / M.sum():.3f}")
    print(pt.head(12).round(4).to_string(index=False))

# ------------------------------------------------------------------ permutation importance
if "perm" in PARTS:
    F, b = FEATURE_SETS["main"], booster("main")
    te = smp[smp.year == 2026].reset_index(drop=True)
    y, X = te[TARGET].values, te[F].copy()
    p0 = b.predict(X)
    L0, A0 = log_loss(y, p0), roc_auc_score(y, p0)
    months = te["month"].values
    targets = [(f, [f], "feature") for f in F] + \
              [(g, [f for f in F if group_of(f) == g], "group") for g in dict.fromkeys(map(group_of, F))]
    rows = []
    for name, cols, kind in targets:
        dl, da = [], []
        for r in range(5):
            Xp = X.copy()
            perm = np.arange(len(X))
            for mo in np.unique(months):                         # permute within month (keeps season)
                ix = np.where(months == mo)[0]
                perm[ix] = rng.permutation(ix)
            Xp[cols] = X[cols].values[perm]
            pp = b.predict(Xp)
            dl.append(log_loss(y, pp) - L0)
            da.append(A0 - roc_auc_score(y, pp))
        rows.append(dict(kind=kind, name=name, group=group_of(name) if kind == "feature" else name,
                         d_logloss=np.mean(dl), d_logloss_sd=np.std(dl), d_auc=np.mean(da), d_auc_sd=np.std(da)))
    pi = pd.DataFrame(rows).sort_values(["kind", "d_auc"], ascending=[True, False])
    pi.to_csv(TAB / "Table12b_permutation_importance.csv", index=False)
    tic(f"permutation importance done (baseline AUC {A0:.4f}, log loss {L0:.4f})")
    print(pi[pi.kind == "group"].round(4).to_string(index=False))

# ------------------------------------------------------------------ local cases
if "local" in PARTS:
    F, sh = FEATURE_SETS["main"], load_shap("main")
    d = pd.concat([smp.reset_index(drop=True), sh], axis=1)
    d = d[d.year == 2026]
    yv, pv = d[TARGET], d["p"]
    med = pv.median()
    # each rule picks from trips not already chosen, so the eight cases are distinct trips
    long_m = (d.is_member == 1) & (d.dist_km > d.dist_km.quantile(0.9))
    short_c = (d.is_member == 0) & (d.dist_km < d.dist_km.quantile(0.2))
    rules = [
        ("Confident e-bike (correct)", yv == 1, lambda s: s["p"].idxmax()),
        ("Confident classic (correct)", yv == 0, lambda s: s["p"].idxmin()),
        ("False positive", (yv == 0) & (pv >= thr), lambda s: s["p"].idxmax()),
        ("False negative", (yv == 1) & (pv < thr), lambda s: s["p"].idxmin()),
        ("Borderline", pv.notna(), lambda s: (s["p"] - thr).abs().idxmin()),
        ("Typical trip", ~long_m & ~short_c, lambda s: (s["p"] - med).abs().idxmin()),
        ("Member, long trip", long_m, lambda s: (s["p"] - med).abs().idxmin()),
        ("Casual, short trip", short_c, lambda s: (s["p"] - med).abs().idxmin()),
    ]
    cases, used = {}, set()
    for name, mask, pick in rules:
        i = pick(d[mask & ~d.index.isin(used)])
        cases[name] = i
        used.add(i)
    assert len(set(cases.values())) == len(cases)
    rows = []
    for lab, i in cases.items():
        r = d.loc[i]
        for f in F:
            rows.append(dict(case=lab, ride_id=r["ride_id"], y=int(r[TARGET]), p=r["p"], base=r["base"],
                             feature=f, value=r[f], shap=r[f"shap_{f}"]))
    pd.DataFrame(rows).to_csv(TAB / "Table13_local_cases.csv", index=False)
    tic("local cases saved (Table13)")

# ------------------------------------------------------------------ residual drift localisation
if "resid" in PARTS:
    pr = pd.read_parquet(OUT / "predictions" / "test_predictions_main.parquet")
    te = load("test", "main", extra=("started_at", "ride_id", "is_member"))
    assert len(pr) == len(te) and (pr["started_at"].values == te["started_at"].values).all()
    assert (pr["y"].values == te[TARGET].values).all()
    tr = pd.read_parquet(ROOT / "data" / "processed" / "trips_stations.parquet", columns=["ride_id", "o_tract"])
    te = te.merge(tr, on="ride_id", how="left", validate="one_to_one")
    te["p"], te["y"] = pr["p_lgbm"].values, pr["y"].values
    te["rider"] = np.where(te["is_member"] == 1, "member", "casual")
    te["dist_band"] = pd.cut(te["dist_km"], [0, 0.5, 1, 1.5, 2, 3, 4, 6, 50], include_lowest=True).astype(str)
    te["income_decile"] = pd.qcut(te["o_median_income"].rank(method="first"), 10, labels=False) + 1
    te["month"] = te["started_at"].dt.month

    def agg(by):
        g = te.groupby(by, observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
        g["gap_pp"] = 100 * (g["observed"] - g["predicted"])
        g["gap_logodds"] = logit(g["observed"]) - logit(g["predicted"])
        return g.reset_index()

    for name, by in (("tract", ["o_tract"]), ("dist_rider", ["dist_band", "rider"]), ("hour_rider", ["hour", "rider"]),
                     ("income_rider", ["income_decile", "rider"]), ("month_rider", ["month", "rider"])):
        g = agg(by)
        if name == "tract":
            g = g[g["n"] >= 300]
        g.to_csv(XAI / f"residual_by_{name}.csv", index=False)
    tic("residual localisation saved")
    print(agg(["rider"]).round(4).to_string(index=False))
tic("done")
