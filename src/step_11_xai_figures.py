"""Step 11: model and explanation figures, built ONLY from saved results of steps 06-10 (no model fitting).

Fig09_model_performance   validation comparison; test ROC, PR, calibration; monthly AUC and intercept; CIs; tuning

Fig10_global_importance   beeswarm; group importance (SHAP vs permutation); importance 2025 vs 2026
Fig11_ale_effects         ALE (log-odds) of the top continuous features, members vs casual, 2025 vs 2026
Fig12_interactions        SHAP interaction strength; distance x rider; temperature x rider; hour x CBD distance
Fig13_drift_decomposition season-matched 2025->2026 decomposition: monthly, by group, share explained, covariates
Fig14_drift_localization  weekly observed vs predicted 2026; gap by distance, income; tract map
Fig15_local_cases         eight local SHAP waterfalls
Fig16_robustness          seed stability; twin-model transfer and effect drift; monthly refit vs frozen (needs step 10)

Run:  python src/step_11_xai_figures.py [--only fig13,fig14]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fstudio as F  # noqa: E402
import plots as P  # noqa: E402
import plots_extra as PX  # noqa: E402

F.use_theme("atlas_white")
mpl.rcParams.update({"font.weight": "bold"})
OUT = ROOT / "outputs"
FIG, TAB, XAI = OUT / "figures", OUT / "tables", OUT / "xai"
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="fig09,fig10,fig11,fig12,fig13,fig14,fig15,fig16")
ONLY = ap.parse_args().only.split(",")
FEATS = json.loads((ROOT / "configs" / "features.json").read_text())["main"]
THR = json.loads((OUT / "reports" / "07_best_params_main.json").read_text())["threshold"]
rng = np.random.default_rng(42)

LAB = {"hour": "Hour of day", "is_weekend": "Weekend", "is_holiday": "Public holiday", "is_daylight": "Daylight",
       "is_member": "Member", "temp_c": "Temperature", "precip_mm": "Rain (hour)", "precip_prev3h_mm": "Rain (prev. 3 h)",
       "snow_cm": "Snow", "wind_kmh": "Wind speed", "humidity_pct": "Humidity", "cloud_pct": "Cloud cover",
       "dist_km": "Trip distance", "round_trip": "Round trip", "o_dist_cbd_km": "Origin: distance to Loop",
       "sup_ebike_share_7d": "Station e-bike share (7 d)", "sup_departures_7d": "Station departures (7 d)"}
CTX = {"pop_density_km2": "population density", "median_income": "median income", "pct_age_18_29": "aged 18-29",
       "pct_degree": "degree holders", "pct_hh_no_car": "car-free households", "pct_commute_transit": "transit commuters",
       "pct_commute_bicycle": "bike commuters", "pct_commute_walk": "walk commuters", "pct_commute_wfh": "work from home",
       "pct_jobs_retail_food": "retail/food jobs"}
for k, v in CTX.items():
    LAB[f"o_{k}"], LAB[f"d_{k}"] = f"Origin: {v}", f"Dest.: {v}"
UNIT = {"dist_km": "Trip distance (km)", "temp_c": "Temperature (°C)", "hour": "Hour of day",
        "o_dist_cbd_km": "Origin distance to Loop (km)", "o_median_income": "Origin median income (US$)",
        "d_median_income": "Dest. income (US$)"}
GROUP_COL = {"Trip": F.C6[0], "Origin area": F.C6[1], "Destination area": F.C6[4], "Rider": F.C6[3],
             "Time": F.C6[2], "Weather": F.C6[5], "Station supply": F.INK2}
RIDER = {"member": ("Members", F.S4[0]), "casual": ("Casual", F.S4[1])}
captions = {}


def lab(f):
    return LAB.get(f, f)


def ci_err(est, lo, hi):
    return np.vstack([np.asarray(est) - np.asarray(lo), np.asarray(hi) - np.asarray(est)])


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


smp = pd.read_parquet(XAI / "sample.parquet")
sh = pd.read_parquet(XAI / "shap_main.parquet")
gi = pd.read_csv(TAB / "Table10_shap_importance.csv")

# =================================================================== Fig09 (ML performance)
if "fig09" in ONLY:
    import ast
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
    MC = {"LightGBM": F.S4[0], "LightGBM + supply": F.S4[2], "LightGBM origin-only": F.S4[3],
          "LogReg": F.S4[1], "XGBoost": F.C6[3], "CatBoost": F.C6[5], "Dummy": F.MUTED}
    pred = {}
    for s_ in ("main", "main_plus_supply", "origin_only"):
        d_ = pd.read_parquet(OUT / "predictions" / f"test_predictions_{s_}.parquet").sample(400_000, random_state=42)
        pred[s_] = d_
    curves = {"LightGBM": (pred["main"]["y"], pred["main"]["p_lgbm"]),
              "LightGBM + supply": (pred["main_plus_supply"]["y"], pred["main_plus_supply"]["p_lgbm"]),
              "LogReg": (pred["main"]["y"], pred["main"]["p_logreg"]),
              "LightGBM origin-only": (pred["origin_only"]["y"], pred["origin_only"]["p_lgbm"])}
    t8 = pd.read_csv(TAB / "Table8_test_overall.csv")
    t9 = pd.read_csv(TAB / "Table9_test_by_group.csv")
    fig = F.new_figure(180, 205)
    gs = fig.add_gridspec(3, 3)

    # (a) validation comparison of candidate algorithms
    ax = fig.add_subplot(gs[0, 0])
    t6 = {k: pd.read_csv(TAB / f"Table6_candidate_models_valid_{k}.csv").set_index("model")
          for k in ("main", "origin_only")}
    models = ["Dummy", "LogReg", "CatBoost", "XGBoost", "LightGBM"]
    yy = np.arange(len(models))
    ax.hlines(yy, t6["origin_only"].loc[models, "roc_auc"], t6["main"].loc[models, "roc_auc"], color=F.RULE, lw=2.4)
    ax.scatter(t6["origin_only"].loc[models, "roc_auc"], yy, s=30, marker="s", color=F.BG,
               edgecolor=[MC[m] for m in models], lw=1.3, zorder=3, label="Origin only")
    ax.scatter(t6["main"].loc[models, "roc_auc"], yy, s=40, color=[MC[m] for m in models], zorder=4, label="Main")
    for yi, m in zip(yy, models):
        ax.text(t6["main"].loc[m, "roc_auc"] + 0.009, yi, f"{t6['main'].loc[m, 'roc_auc']:.3f}", va="center", fontsize=7.2)
    ax.set_yticks(yy, models)
    ax.set_xlim(0.48, 0.70)
    ax.set_xlabel("Validation AUC (2025)")
    ax.legend(fontsize=7, loc="lower right", handletextpad=0.2)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Algorithms")

    # (b) ROC on 2026 test
    ax = fig.add_subplot(gs[0, 1])
    ax.plot([0, 1], [0, 1], color=F.MUTED, lw=0.8, ls="--")
    for nm, (y_, p_) in curves.items():
        fpr, tpr, _ = roc_curve(y_, p_)
        k = np.linspace(0, len(fpr) - 1, 400).astype(int)
        ax.plot(fpr[k], tpr[k], color=MC[nm], lw=1.6, label=f"{roc_auc_score(y_, p_):.3f}")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(title="AUC", fontsize=7, title_fontsize=7, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    F.panel(fig, ax, "b", "ROC, 2026 test")

    # (c) precision-recall
    ax = fig.add_subplot(gs[0, 2])
    for nm, (y_, p_) in curves.items():
        pr_, rc_, _ = precision_recall_curve(y_, p_)
        k = np.linspace(0, len(pr_) - 1, 400).astype(int)
        ax.plot(rc_[k], pr_[k], color=MC[nm], lw=1.6, label=f"{average_precision_score(y_, p_):.3f}")
        ax.axhline(y_.mean(), color=MC[nm], lw=0.7, ls=":")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.4, 1)
    ax.legend(title="PR-AUC", fontsize=7, title_fontsize=7, loc="upper right")
    F.panel(fig, ax, "c", "Precision-recall")

    # (d) calibration with histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.plot([0, 1], [0, 1], color=F.MUTED, lw=0.8, ls="--")
    for nm in ("LightGBM", "LightGBM + supply", "LogReg"):
        y_, p_ = curves[nm]
        q = pd.qcut(p_, 10, labels=False, duplicates="drop")
        ax.plot(p_.groupby(q).mean(), y_.groupby(q).mean(), "o-", color=MC[nm], ms=3.4, lw=1.4, mec=F.BG, mew=0.5,
                label=nm.replace("LightGBM", "LGBM"))
    h_ = np.histogram(curves["LightGBM"][1], bins=40, range=(0, 1))[0]
    ax.bar(np.linspace(0.0125, 0.9875, 40), 0.18 * h_ / h_.max(), width=0.025, color=MC["LightGBM"], alpha=0.25, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed e-bike share")
    ax.legend(fontsize=6.8, loc="upper left")
    F.panel(fig, ax, "d", "Calibration")

    # (e) monthly AUC
    ax = fig.add_subplot(gs[1, 1])
    mon = t9[t9.group == "month"]
    for nm, (fs, md) in {"LightGBM": ("main", "LightGBM"), "LightGBM + supply": ("main_plus_supply", "LightGBM"),
                         "LogReg": ("main", "LogReg")}.items():
        d_ = mon[(mon.feature_set == fs) & (mon.model == md)].sort_values("value")
        ax.plot(range(len(d_)), d_["roc_auc"], "o-", color=MC[nm], ms=3.2, lw=1.4)
    ax.axvspan(2.5, 9.5, color=F.S4[1], alpha=0.06, lw=0)
    ax.axvline(2.5, color=F.INK2, lw=0.8, ls=":")
    ax.text(9.3, 0.70, "from Mar 16", fontsize=7, color=F.INK2, ha="right")
    ax.set_xticks(range(10), list("JFMAMJJASO"))
    ax.set_xlabel("2026 month")
    ax.set_ylabel("Test AUC")
    F.panel(fig, ax, "e", "AUC by month")

    # (f) monthly calibration-in-the-large
    ax = fig.add_subplot(gs[1, 2])
    w = 0.38
    for j, (nm, fs) in enumerate((("LightGBM", "main"), ("LightGBM + supply", "main_plus_supply"))):
        d_ = mon[(mon.feature_set == fs) & (mon.model == "LightGBM")].sort_values("value")
        ax.bar(np.arange(10) + (j - 0.5) * w, d_["cal_intercept"], width=w, color=MC[nm], label=nm.replace("LightGBM", "LGBM"))
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.set_xticks(range(10), list("JFMAMJJASO"))
    ax.set_xlabel("2026 month")
    ax.set_ylabel("Calibration intercept")
    ax.legend(fontsize=6.8, loc="upper left")
    F.panel(fig, ax, "f", "Drift in level")

    # (g) test AUC with day-block bootstrap CIs
    ax = fig.add_subplot(gs[2, 0])
    rows_ = [("main", "LightGBM", "LightGBM"), ("main_plus_supply", "LightGBM", "LightGBM + supply"),
             ("main", "LogReg", "LogReg"), ("origin_only", "LightGBM", "LightGBM origin-only"),
             ("main_plus_supply", "LogReg", None), ("origin_only", "LogReg", None)]
    yy = np.arange(len(rows_))[::-1]
    lbls = []
    for yi, (fs, md, nm) in zip(yy, rows_):
        r_ = t8[(t8.feature_set == fs) & (t8.model == md)].iloc[0]
        lo_, hi_ = ast.literal_eval(r_["roc_auc_ci95"])
        col = MC[nm] if nm else F.tint(MC["LogReg"], 0.45)
        ax.errorbar(r_["roc_auc"], yi, xerr=[[r_["roc_auc"] - lo_], [hi_ - r_["roc_auc"]]], fmt="o", color=col,
                    ms=5, capsize=2.5, lw=1.2)
        lbls.append({"main": "Main", "main_plus_supply": "+ Supply", "origin_only": "Origin only"}[fs] + f" · {md.replace('LightGBM', 'LGBM')}")
    ax.set_yticks(yy, lbls, fontsize=7.4)
    ax.set_xlabel("Test AUC (95% CI)")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "g", "Test AUC")

    # (h) confusion matrix, main LightGBM at the validation threshold
    ax = fig.add_subplot(gs[2, 1])
    y_, p_ = curves["LightGBM"]
    yh = (p_ >= THR).astype(int)
    cm = np.array([[((y_ == a) & (yh == b)).sum() for b in (0, 1)] for a in (0, 1)], float)
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap=F.HEAT, vmin=0, vmax=1)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, f"{100 * cmn[a, b]:.1f}%", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color=F.text_on(im.cmap(im.norm(cmn[a, b]))))
    ax.set_xticks([0, 1], ["Classic", "E-bike"])
    ax.set_yticks([0, 1], ["Classic", "E-bike"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.grid(False)
    F.panel(fig, ax, "h", f"Confusion (t = {THR:.2f})")

    # (i) Optuna search history
    ax = fig.add_subplot(gs[2, 2])
    for nm, fs in (("LightGBM", "main"), ("LightGBM + supply", "main_plus_supply"), ("LightGBM origin-only", "origin_only")):
        t7 = pd.read_csv(TAB / f"Table7_tuning_{fs}.csv").sort_values("number")
        ax.scatter(t7["number"], t7["value"], s=8, color=MC[nm], alpha=0.35, lw=0)
        ax.plot(t7["number"], t7["value"].cummin(), color=MC[nm], lw=1.6, drawstyle="steps-post")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Validation log loss")
    F.panel(fig, ax, "i", "Tuning")
    h = [Line2D([], [], color=MC[n], lw=2, label=n) for n in ("LightGBM", "LightGBM + supply", "LogReg", "LightGBM origin-only")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=8)
    F.save(fig, FIG / "Fig09_model_performance")
    captions["Fig09"] = (
        "**Figure 9. Predictive performance.** (a) Validation AUC (Sep-Dec 2025) of untuned candidate algorithms on "
        "the main (filled) and origin-only (open) feature sets; LightGBM was selected for tuning. (b-d) ROC, "
        "precision-recall and decile calibration curves of the locked models on the 2026 test set (400,000 "
        "randomly drawn trips per model; dotted lines in (c) = observed e-bike share; bars in (d) = distribution of "
        "LightGBM predictions). (e) Test AUC by month. (f) Calibration intercept by month (positive = e-bike use "
        "under-predicted). (g) Test AUC with day-block bootstrap 95% CIs (200 replicates) on all 3.6 million test "
        "trips. (h) Confusion matrix of the main LightGBM model, row-normalized, at the threshold chosen on "
        f"validation ({THR:.2f}). (i) Optuna search: each trial's validation log loss (points) and best value so far "
        "(line). The origin-only model uses all dock-start trips, so its sample and base rate differ.")

# =================================================================== Fig10
if "fig10" in ONLY:
    fig = F.new_figure(180, 150)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1])
    imp = gi.query("model == 'main' and kind == 'feature'")
    top = imp[imp.year == 2026].sort_values("rank")["name"].tolist()[:15]
    m26 = (smp.year == 2026).values
    idx = rng.choice(np.where(m26)[0], 12000, replace=False)
    ax = fig.add_subplot(outer[0])
    V = sh.loc[idx, [f"shap_{f}" for f in top]].values
    X = smp.loc[idx, top].values.astype(float)
    P.beeswarm(ax, V, X, [lab(f) for f in top], order=list(range(len(top)))[::-1])
    ax.set_xlabel("SHAP value (log-odds of e-bike)")
    F.panel(fig, ax, "a", "SHAP values, 2026")

    right = outer[1].subgridspec(2, 1, height_ratios=[1, 1.55])
    ax = fig.add_subplot(right[0])
    gg = imp_g = gi.query("model == 'main' and kind == 'group' and year == 2026").set_index("name")
    pi = pd.read_csv(TAB / "Table12b_permutation_importance.csv").query("kind == 'group'").set_index("name")
    order = gg["mean_abs_shap"].sort_values().index
    y = np.arange(len(order))
    s1 = gg.loc[order, "mean_abs_shap"] / gg["mean_abs_shap"].max()
    s2 = pi.loc[order, "d_auc"] / pi["d_auc"].max()
    ax.barh(y + 0.19, s1, height=0.36, color=[GROUP_COL[g_] for g_ in order], label="Mean |SHAP|")
    ax.barh(y - 0.19, s2, height=0.36, color=[F.tint(GROUP_COL[g_], 0.55) for g_ in order],
            edgecolor=[GROUP_COL[g_] for g_ in order], lw=0.8, label="Permutation ΔAUC")
    ax.set_yticks(y, order)
    ax.set_xlabel("Relative importance")
    ax.set_xlim(0, 1.75)
    ax.legend(handles=[Patch(color=F.INK2, label="SHAP"), Patch(fc=F.tint(F.INK2, 0.55), ec=F.INK2, label="Permutation")],
              loc="lower right", fontsize=7, handlelength=1.0)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "b", "Feature groups")

    ax = fig.add_subplot(right[1])
    t12 = top[:12][::-1]
    a = imp[imp.year == 2025].set_index("name").loc[t12]
    b = imp[imp.year == 2026].set_index("name").loc[t12]
    yy = np.arange(len(t12))
    ax.hlines(yy, a["mean_abs_shap"], b["mean_abs_shap"], color=F.RULE, lw=2.2, zorder=1)
    ax.errorbar(a["mean_abs_shap"], yy, xerr=ci_err(a["mean_abs_shap"], a["lo"], a["hi"]), fmt="o", ms=4.2,
                color=F.MUTED, mfc=F.BG, mec=F.INK2, capsize=0, lw=0.8, label="2025", zorder=2)
    ax.errorbar(b["mean_abs_shap"], yy, xerr=ci_err(b["mean_abs_shap"], b["lo"], b["hi"]), fmt="o", ms=4.2,
                color=F.G1, capsize=0, lw=0.8, label="2026", zorder=3)
    ax.set_yticks(yy, [lab(f) for f in t12])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=7.5)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "c", "2025 vs 2026")
    F.save(fig, FIG / "Fig10_global_importance")
    captions["Fig10"] = (
        "**Figure 10. Global explanation of the locked main LightGBM model.** (a) SHAP values (log-odds of choosing "
        "an e-bike) for the 15 most important features, 12,000 randomly drawn 2026 test trips; colour = within-feature "
        "percentile of the feature value. (b) Importance of feature groups, as mean absolute grouped SHAP value and as "
        "the drop in test AUC when the group is permuted within month (5 repeats), each scaled to its maximum. "
        "(c) Mean |SHAP| in season-matched 2025 (Feb-Oct, training period) and 2026 (Jan-Oct, test) samples "
        "with day-block bootstrap 95% CIs. SHAP values describe model behavior, not causal effects.")

# =================================================================== Fig11
if "fig11" in ONLY:
    ale = pd.read_csv(XAI / "ale_main.csv")
    sup = pd.read_parquet(XAI / "ale_support.parquet")
    feats = [f for f in ["dist_km", "temp_c", "hour", "o_dist_cbd_km", "o_pct_hh_no_car", "d_pct_hh_no_car",
                         "o_pct_commute_transit", "d_median_income"] if f in ale.feature.unique()]
    fig = F.new_figure(180, 150)
    gs = fig.add_gridspec(2, 4)
    for i, f in enumerate(feats):
        sub = gs[i // 4, i % 4].subgridspec(2, 1, height_ratios=[4, 0.8])
        ax, axn = fig.add_subplot(sub[0]), None
        axn = fig.add_subplot(sub[1], sharex=ax)
        x_all = sup[f].dropna()
        lo_x, hi_x = (np.nanpercentile(x_all, 1), np.nanpercentile(x_all, 99)) if f != "hour" else (-0.5, 23.5)
        dm = ale[(ale.feature == f) & (ale.subset == "2026 member")]
        dc = ale[(ale.feature == f) & (ale.subset == "2026 casual")]
        okx = (dm.x.values >= lo_x) & (dm.x.values <= hi_x)
        ax.fill_between(dm.x.values[okx], dm.ale.values[okx], dc.ale.values[okx], color=F.S4[2], alpha=0.16, lw=0,
                        zorder=0)
        for r in ("member", "casual"):
            for yr, ls in ((2025, "--"), (2026, "-")):
                d = ale[(ale.feature == f) & (ale.subset == f"{yr} {r}")]
                d = d[(d.x >= lo_x) & (d.x <= hi_x)]
                col = RIDER[r][1]
                if yr == 2026:
                    ax.fill_between(d.x, d.lo, d.hi, color=col, alpha=0.18, lw=0)
                ax.plot(d.x, d.ale, color=col, ls=ls, lw=1.6 if yr == 2026 else 1.0)
        ax.axhline(0, color=F.RULE, lw=0.8, zorder=0)
        F.ygrid(ax)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        if i % 4 == 0:
            ax.set_ylabel("ALE (log-odds)")
        bins = np.arange(-0.5, 24.5, 1) if f == "hour" else np.linspace(lo_x, hi_x, 30)
        hm_, _ = np.histogram(sup.loc[sup.is_member == 1, f].dropna().clip(lo_x, hi_x), bins=bins)
        hc_, _ = np.histogram(sup.loc[sup.is_member == 0, f].dropna().clip(lo_x, hi_x), bins=bins)
        ctr, wid = (bins[1:] + bins[:-1]) / 2, np.diff(bins) * 0.92
        axn.bar(ctr, hm_ / max(hm_.max(), 1), width=wid, color=F.tint(RIDER["member"][1], 0.25), lw=0)
        axn.bar(ctr, -hc_ / max(hc_.max(), 1), width=wid, color=F.tint(RIDER["casual"][1], 0.25), lw=0)
        axn.axhline(0, color=F.BG, lw=0.6)
        axn.set_yticks([])
        axn.spines["left"].set_visible(False)
        axn.set_xlabel(UNIT.get(f, lab(f).replace("Origin: ", "Origin ").replace("Dest.: ", "Dest. ") + " (%)"
                                 if "pct" in f else lab(f)), fontsize=8)
        if f == "hour":
            axn.set_xticks([0, 6, 12, 18, 23])
        if "income" in f:
            axn.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.set_xlim(lo_x, hi_x)
        short = {"dist_km": "Distance", "temp_c": "Temperature", "hour": "Hour", "o_dist_cbd_km": "Distance to Loop",
                 "o_pct_hh_no_car": "Car-free (origin)", "d_pct_hh_no_car": "Car-free (dest.)",
                 "o_pct_commute_transit": "Transit (origin)", "d_median_income": "Income (dest.)"}
        F.panel(fig, ax, "abcdefgh"[i], short[f])
    h = [Line2D([], [], color=RIDER[r][1], lw=1.8, label=RIDER[r][0]) for r in RIDER] + \
        [Line2D([], [], color=F.INK2, lw=1.0, ls="--", label="2025"), Line2D([], [], color=F.INK2, lw=1.6, label="2026 (95% band)"),
         Patch(color=F.tint(F.S4[2], 0.6), label="Rider gap, 2026")]
    fig.legend(handles=h, loc="outside lower center", ncol=5, fontsize=8)
    F.save(fig, FIG / "Fig11_ale_effects")
    captions["Fig11"] = (
        "**Figure 11. Accumulated local effects (ALE) of the locked main model.** Centred ALE on the log-odds scale "
        "for members (blue) and casual riders (orange), computed on season-matched 2025 (dashed) and 2026 "
        "(solid, day-block bootstrap 95% band) samples with 20 quantile bins (hourly bins for hour); the yellow area is "
        "the 2026 gap between members and casual riders. The mirrored strip under each panel shows the distribution "
        "of the feature for members (up) and casual riders (down); the x-axis spans its 1st-99th percentile. "
        "Because the model is fixed, differences between dashed and solid lines reflect where riders were in the "
        "joint feature space, not a re-estimated effect; changed effects are examined with twin models (Figure 16). "
        "Neighborhood measures take one value per census tract, so their ALE curves are step-like and partly "
        "capture place-specific differences between station areas rather than a smooth effect of the measure.")

# =================================================================== Fig12
if "fig12" in ONLY:
    z = np.load(XAI / "shap_interactions_main.npz", allow_pickle=True)
    I, Xi, fnames, mem = z["values"], z["X"], list(z["features"]), z["is_member"]
    M = np.abs(I).mean(0)
    tot = M.sum(1) + M.sum(0) - np.diag(M)
    top = list(np.argsort(-tot)[:12])
    short = lambda f: lab(f).replace("Origin: ", "O: ").replace("Dest.: ", "D: ")
    fig = F.new_figure(180, 225)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 1])
    top_gs = outer[0].subgridspec(1, 2, width_ratios=[1, 1.05])
    mid = outer[1].subgridspec(1, 3)
    bot = outer[2].subgridspec(1, 3)

    ax = fig.add_subplot(top_gs[0])
    S = 2 * M[np.ix_(top, top)]
    np.fill_diagonal(S, np.nan)
    S[np.triu_indices(len(top), 1)] = np.nan
    im = ax.imshow(S, cmap=F.HEAT, vmin=0)
    for i in range(len(top)):
        for j in range(i):
            if S[i, j] >= np.nanpercentile(S, 85):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                        color=F.text_on(im.cmap(im.norm(S[i, j]))))
    names = [short(fnames[i]) for i in top]
    ax.set_xticks(range(len(top)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top)), names, fontsize=7)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    cax = ax.inset_axes([0.5, 0.88, 0.45, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Mean |interaction|", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    F.panel(fig, ax, "a", "Pairwise interactions")

    ax = fig.add_subplot(top_gs[1])
    pt = pd.read_csv(TAB / "Table12a_shap_interactions.csv").head(10)[::-1]
    nrm = mpl.colors.Normalize(pt["share_of_total"].min() * 0.6, pt["share_of_total"].max())
    ax.barh(range(len(pt)), 100 * pt["share_of_total"], color=[F.HEAT(nrm(v)) for v in pt["share_of_total"]], height=0.66)
    for yi, (a_, b_, v) in enumerate(zip(pt.f1, pt.f2, pt.share_of_total)):
        ax.text(100 * v + 0.04, yi, f"{100 * v:.1f}%", va="center", fontsize=6.8)
        ax.scatter([-0.12, -0.05], [yi, yi], s=26, color=[GROUP_COL[group_of(a_)], GROUP_COL[group_of(b_)]],
                   clip_on=False, zorder=4, ec=F.BG, lw=0.5)
    ax.set_yticks(range(len(pt)), [f"{short(a)} × {short(b)}" for a, b in zip(pt.f1, pt.f2)], fontsize=6.8)
    ax.tick_params(axis="y", pad=14)
    ax.set_xlim(0, 100 * pt["share_of_total"].max() * 1.22)
    ax.set_xlabel("Share of |SHAP| (%)")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "b", "Top pairs (dots = groups)")

    def dep(ax, f, title, letter, xlab, groups, xlim=None):
        j = fnames.index(f)
        x = Xi[:, j]
        tot_phi = I[:, j, :].sum(1)
        for (name, mask, col) in groups:
            ax.scatter(x[mask], tot_phi[mask], s=3, color=col, alpha=0.22, lw=0, rasterized=True)
            xs, ys = x[mask], tot_phi[mask]
            ok = ~np.isnan(xs)
            bins = np.unique(np.nanquantile(xs[ok], np.linspace(0, 1, 16)))
            k = np.clip(np.digitize(xs[ok], bins) - 1, 0, len(bins) - 2)
            mx = [xs[ok][k == b].mean() for b in range(len(bins) - 1) if (k == b).any()]
            my = [ys[ok][k == b].mean() for b in range(len(bins) - 1) if (k == b).any()]
            PX.glow_line(ax, mx, my, col, lw=1.7, label=name)
        ax.axhline(0, color=F.RULE, lw=0.8, zorder=0)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"SHAP, {lab(f).lower()}", fontsize=8)
        F.ygrid(ax)
        ax.legend(fontsize=6.8, loc="best", handlelength=1.2)
        F.panel(fig, ax, letter, title)

    rider_groups = [("Members", mem == 1, RIDER["member"][1]), ("Casual", mem == 0, RIDER["casual"][1])]
    dep(fig.add_subplot(mid[0]), "dist_km", "Distance × rider", "c", "Trip distance (km)", rider_groups,
        (0, np.nanpercentile(Xi[:, fnames.index("dist_km")], 99)))
    dep(fig.add_subplot(mid[1]), "temp_c", "Temperature × rider", "d", "Temperature (°C)", rider_groups)
    cbd = Xi[:, fnames.index("o_dist_cbd_km")]
    q = np.nanquantile(cbd, [1 / 3, 2 / 3])
    dep(fig.add_subplot(mid[2]), "hour", "Hour × location", "e", "Hour of day",
        [("Near Loop", cbd <= q[0], F.ORD[3]), ("Middle", (cbd > q[0]) & (cbd <= q[1]), F.ORD[2]),
         ("Far", cbd > q[1], F.S4[0])], (0, 23))

    s26 = smp.assign(p=sh["p"].values)
    s26 = s26[s26.year == 2026]
    DB = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 50]
    DL = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-6", ">6"]
    surf = []
    for m in (1, 0):
        t_ = s26[s26.is_member == m]
        g = t_.groupby([pd.cut(t_["dist_km"], DB, include_lowest=True), "hour"], observed=False)["p"].agg(["mean", "size"])
        surf.append(np.where(g["size"].unstack().values >= 25, g["mean"].unstack().values, np.nan))
    vmin, vmax = np.nanpercentile(np.r_[surf[0].ravel(), surf[1].ravel()], [2, 98])
    axs = []
    for k, (m, M_) in enumerate(zip((1, 0), surf)):
        ax = fig.add_subplot(bot[k])
        im = ax.imshow(M_, aspect="auto", origin="lower", cmap=F.SEQ, vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_yticks(range(len(DL)), DL if k == 0 else [""] * len(DL), fontsize=7)
        ax.set_xlabel("Start hour")
        if k == 0:
            ax.set_ylabel("Distance (km)")
        ax.tick_params(length=0)
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        axs.append(ax)
        F.panel(fig, ax, "fg"[k], f"{RIDER['member' if m else 'casual'][0]}: model surface")
    cb = fig.colorbar(im, ax=axs, location="bottom", shrink=0.6, aspect=35, pad=0.02,
                      format=mpl.ticker.PercentFormatter(1, 0))
    cb.set_label("Mean predicted e-bike probability, 2026", fontsize=7.2)
    cb.outline.set_visible(False)
    ax = fig.add_subplot(bot[2])
    TB = [-25, 0, 5, 10, 15, 20, 25, 40]
    TL = ["<0", "0-5", "5-10", "10-15", "15-20", "20-25", ">25"]
    g = s26.groupby([pd.cut(s26["temp_c"], TB), pd.cut(s26["dist_km"], DB, include_lowest=True)], observed=False)["p"].agg(["mean", "size"])
    M_ = np.where(g["size"].unstack().values >= 25, g["mean"].unstack().values, np.nan)
    im = ax.imshow(M_, aspect="auto", origin="lower", cmap=F.SEQ, vmin=vmin, vmax=vmax)
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]):
            if not np.isnan(M_[i, j]):
                ax.text(j, i, f"{100 * M_[i, j]:.0f}", ha="center", va="center", fontsize=5.8,
                        color=F.text_on(im.cmap(im.norm(M_[i, j]))))
    ax.set_xticks(range(len(DL)), DL, rotation=45, fontsize=6.6)
    ax.set_yticks(range(len(TL)), TL, fontsize=7)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Temperature (°C)")
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    F.panel(fig, ax, "h", "Temperature × distance")
    F.save(fig, FIG / "Fig12_interactions")
    main_share = np.trace(M) / M.sum()
    captions["Fig12"] = (
        f"**Figure 12. Interactions in the locked main model.** (a) Mean absolute SHAP interaction value (both halves "
        f"summed) among the 12 features with the largest total attribution, 3,000 2026 test trips (300 per month); "
        f"the strongest pairs are labelled. (b) The ten largest pairwise interactions as a share of total absolute "
        f"attribution; dots show the feature group of each partner. Main effects account for "
        f"{100 * main_share:.0f}% of the total, so non-additive structure is substantial. (c-e) SHAP value of a "
        f"feature against its value, coloured by a second variable; lines are binned means. (f, g) Mean predicted "
        f"e-bike probability by distance band and start hour for members and casual riders, and (h) by temperature "
        f"and distance (all riders, %), on the 2026 explanation sample (95,000 trips; cells with fewer than 25 trips "
        f"blank). These are conditional means of model output, not causal surfaces.")

# =================================================================== Fig13
if "fig13" in ONLY:
    dec = pd.read_csv(TAB / "Table11_drift_decomposition.csv")
    summ = pd.read_csv(TAB / "Table11a_drift_summary.csv")
    mon = pd.read_csv(XAI / "drift_decomposition_monthly.csv")
    cov = pd.read_csv(TAB / "Table11b_covariate_drift.csv")
    GR = ["Trip", "Rider", "Time", "Weather", "Origin area", "Destination area"]
    UNEX = F.G1
    fig = F.new_figure(180, 222)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])

    ax = fig.add_subplot(gs[0, 0])
    a = mon.query("model == 'main' and subset == 'all'").set_index("month")
    x = np.arange(len(a))
    pos, neg = np.zeros(len(a)), np.zeros(len(a))
    for g_ in GR:
        v = a[g_].values
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=GROUP_COL[g_], width=0.7, lw=0)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=GROUP_COL[g_], width=0.7, lw=0)
        pos += np.where(v > 0, v, 0)
        neg += np.where(v < 0, v, 0)
    un = (a["observed_logodds"] - a["explained_logodds"]).values
    ax.bar(x, un, bottom=pos, color=F.tint(UNEX, 0.45), edgecolor=UNEX, hatch="////", lw=0.6, width=0.7)
    ax.scatter(x, a["observed_logodds"], marker="D", s=26, color=F.INK, zorder=4)
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.set_xticks(x, ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"], fontsize=7.4)
    ax.set_ylabel("Δ log-odds, 2025→2026")
    ax.legend(handles=[Patch(color=GROUP_COL[g_], label=g_) for g_ in GR] +
              [Patch(fc=F.tint(UNEX, 0.45), ec=UNEX, hatch="////", label="Not explained"),
               Line2D([], [], marker="D", ls="", color=F.INK, label="Observed")],
              fontsize=6.3, ncol=2, loc="upper left", columnspacing=0.6, handlelength=1.1)
    ax.set_ylim(-0.12, 0.95)
    F.ygrid(ax)
    F.panel(fig, ax, "a", "Month by month")

    ax = fig.add_subplot(gs[0, 1])
    s0 = summ.query("model == 'main' and subset == 'all'").iloc[0]
    g = dec.query("model == 'main' and subset == 'all' and kind == 'group'").set_index("name")
    steps = [(g_, g.loc[g_, "delta_logodds"], GROUP_COL[g_]) for g_ in GR] + \
            [("Not explained", s0["unexplained_change_logodds"], UNEX)]
    run = 0
    for i, (nm, v, col) in enumerate(steps):
        ax.bar(i, v, bottom=run, color=col, width=0.66, hatch="////" if nm == "Not explained" else None,
               edgecolor=UNEX if nm == "Not explained" else col, lw=0.6)
        ax.plot([i + 0.33, i + 0.67], [run + v, run + v], color=F.MUTED, lw=0.7)
        ax.text(i, max(run, run + v) + 0.01, F.minus(f"{v:+.3f}"), ha="center", va="bottom", fontsize=6.2,
                rotation=90 if abs(v) < 0.1 else 0)
        run += v
    ax.bar(len(steps), s0["observed_change_logodds"], color=F.INK, width=0.66)
    ax.text(len(steps), s0["observed_change_logodds"] + 0.012, f"{s0['observed_change_logodds']:+.3f}", ha="center",
            va="bottom", fontsize=6.8)
    ax.set_xticks(range(len(steps) + 1), [n.replace(" area", "").replace("Destination", "Dest.") for n, _, _ in steps] +
                  ["Observed"], rotation=40, ha="right", fontsize=7.2)
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.set_ylabel("Cumulative Δ log-odds")
    ax.set_ylim(-0.03, 0.52)
    F.ygrid(ax)
    F.panel(fig, ax, "b", "From 2025 to 2026")

    ax = fig.add_subplot(gs[1, 0])
    rows = [(m, sb) for m in ("main", "main_plus_supply") for sb in ("all", "member", "casual")]
    yy = np.arange(len(rows))[::-1]
    for yi, (m, sb) in zip(yy, rows):
        r = summ.query("model == @m and subset == @sb").iloc[0]
        col = {"all": F.INK2, "member": RIDER["member"][1], "casual": RIDER["casual"][1]}[sb]
        mk = "o" if m == "main" else "s"
        ax.barh(yi, 100 * r["share_explained"], color=F.tint(col, 0.65), height=0.6)
        ax.errorbar(100 * r["share_explained"], yi, xerr=[[100 * (r["share_explained"] - r["share_explained_lo"])],
                                                          [100 * (r["share_explained_hi"] - r["share_explained"])]],
                    fmt=mk, color=col, ms=5, capsize=2.5, lw=1)
        ax.text(100 * r["share_explained_hi"] + 2, yi, f"{100 * r['share_explained']:.0f}%", va="center", fontsize=7.3)
    ax.axhline(2.5, color=F.RULE, lw=0.8, ls="--")
    ax.set_yticks(yy, [("Main: " if m == "main" else "+ Supply: ") + {"all": "all", "member": "members",
                                                                    "casual": "casual"}[sb] for m, sb in rows])
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_xlim(-10, 72)
    ax.set_xlabel("Share of observed change explained (%)")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "c", "Share explained")

    ax = fig.add_subplot(gs[1, 1])
    yy = np.arange(len(GR) + 1)[::-1]
    for k, (sb, off) in enumerate((("member", 0.17), ("casual", -0.17))):
        gg = dec.query("model == 'main' and subset == @sb and kind == 'group'").set_index("name")
        ss = summ.query("model == 'main' and subset == @sb").iloc[0]
        vals = [gg.loc[g_, "delta_logodds"] for g_ in GR] + [ss["unexplained_change_logodds"]]
        los = [gg.loc[g_, "lo"] for g_ in GR] + [ss["unexplained_change_logodds_lo"]]
        his = [gg.loc[g_, "hi"] for g_ in GR] + [ss["unexplained_change_logodds_hi"]]
        ax.hlines(yy + off, 0, vals, color=F.tint(RIDER[sb][1], 0.4), lw=2.4)
        ax.errorbar(vals, yy + off, xerr=ci_err(vals, los, his), fmt="o", color=RIDER[sb][1], ms=4.5, capsize=1.8,
                    lw=0.9, label=RIDER[sb][0])
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_yticks(yy, GR + ["Not explained"])
    ax.set_xlabel("Contribution (log-odds)")
    ax.legend(fontsize=7, loc="center right")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "d", "Members vs casual")

    ax = fig.add_subplot(gs[2, 0])
    fdec = dec.query("model == 'main' and subset == 'all' and kind == 'feature'")
    fdec = fdec.reindex(fdec["delta_logodds"].abs().sort_values(ascending=False).index).head(10)[::-1]
    yy = np.arange(len(fdec))
    cols = [GROUP_COL[g_] for g_ in fdec["group"]]
    ax.hlines(yy, 0, fdec["delta_logodds"], color=cols, lw=2.2)
    ax.errorbar(fdec["delta_logodds"], yy, xerr=ci_err(fdec["delta_logodds"], fdec["lo"], fdec["hi"]), fmt="none",
                ecolor=F.INK2, lw=0.8, capsize=1.8)
    ax.scatter(fdec["delta_logodds"], yy, color=cols, s=30, zorder=3, ec=F.BG, lw=0.6)
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_yticks(yy, [lab(f) for f in fdec["name"]], fontsize=7.2)
    ax.set_xlabel("Contribution (log-odds)")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "e", "Top features")

    ax = fig.add_subplot(gs[2, 1])
    c = cov.reindex(cov["smd"].abs().sort_values(ascending=False).index).head(10)[::-1]
    yy = np.arange(len(c))
    ax.axvspan(-0.1, 0.1, color=F.MUTED, alpha=0.12, lw=0)
    ax.hlines(yy, 0, c["smd"], color=[GROUP_COL[g_] for g_ in c["group"]], lw=2.2)
    ax.scatter(c["smd"], yy, color=[GROUP_COL[g_] for g_ in c["group"]], s=30, zorder=3, ec=F.BG, lw=0.6)
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_yticks(yy, [lab(f) for f in c["feature"]], fontsize=7.2)
    ax.set_xlabel("SMD, 2026 vs 2025")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "f", "Covariate shift")
    F.save(fig, FIG / "Fig13_drift_decomposition")
    sa = summ.set_index(["model", "subset"])
    captions["Fig13"] = (
        "**Figure 13. Why did the e-bike share rise? Season-matched decomposition, 2025 to 2026.** Same months "
        "(February to October), each month weighted equally; 2025 trips are training data and 2026 trips test data, "
        "10,000 trips per month and year. Because the model is fixed, the change in mean log-odds equals the sum of "
        "the changes in mean SHAP value of each feature. (a) Monthly contributions of feature groups (stacked), the "
        "part not explained (hatched) and the observed change (diamonds). (b) Cumulative contributions from 2025 to "
        "2026 for all months together. (c) Share of the observed change explained by the main model and by the model "
        f"with the 7-day station supply proxy ({100 * sa.loc[('main', 'all'), 'share_explained']:.0f}% and "
        f"{100 * sa.loc[('main_plus_supply', 'all'), 'share_explained']:.0f}% for all riders). (d) Group contributions "
        "and the unexplained change for members and casual riders. (e) The ten features with the largest "
        "contributions. (f) Standardized mean differences of features between the two years. Error bars: day-block "
        "bootstrap 95% CIs (500 replicates); grey band in (f) = |SMD| < 0.1. The supply proxy is the lagged e-bike share of departures at the station "
        "and carries both fleet availability and earlier rider choices; its share is an upper bound for availability.")

# =================================================================== Fig14
if "fig14" in ONLY:
    import geopandas as gpd
    pr = pd.read_parquet(OUT / "predictions" / "test_predictions_main.parquet")
    pr["week"] = pr["started_at"].dt.to_period("W-SUN").dt.start_time
    fig = F.new_figure(180, 215)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1, 1], width_ratios=[1, 1, 1.1])
    ax = fig.add_subplot(gs[0, :])
    for r, m in (("member", 1), ("casual", 0)):
        w = pr[pr.is_member == m].groupby("week").agg(y=("y", "mean"), p=("p_lgbm", "mean"), n=("y", "size"))
        w = w[w.n >= 500]
        col = RIDER[r][1]
        ax.fill_between(w.index, w.p, w.y, color=col, alpha=0.14, lw=0)
        PX.glow_line(ax, w.index, w.y, col, lw=1.7, label=f"{RIDER[r][0]}, observed")
        ax.plot(w.index, w.p, color=col, lw=1.2, ls="--", label=f"{RIDER[r][0]}, predicted")
    ev = pd.Timestamp("2026-03-16")
    ax.axvspan(ev, pr["started_at"].max(), color=F.S4[2], alpha=0.06, lw=0)
    ax.axvline(ev, color=F.INK2, lw=0.9, ls=":")
    ax.text(ev, 0.75, " Price and fleet\n change (Mar 16)", fontsize=7.3, va="top", color=F.INK)
    F.pct_axis(ax)
    ax.set_ylim(0.35, 0.76)
    ax.set_ylabel("E-bike share")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
    ax.legend(ncol=4, fontsize=7, loc="lower center", columnspacing=1)
    F.ygrid(ax)
    F.panel(fig, ax, "a", "Weekly, 2026 test")

    def gap_lines(ax, df, xcol, xt=None):
        for r in ("member", "casual"):
            d_ = df[df.rider == r].sort_values(xcol)
            x = np.arange(len(d_)) if xt is not None else d_[xcol].values
            ax.fill_between(x, 0, d_["gap_pp"], color=RIDER[r][1], alpha=0.07, lw=0)
            ax.plot(x, d_["gap_pp"], "o-", color=RIDER[r][1], ms=3.2, lw=1.5, mec=F.BG, mew=0.5, label=RIDER[r][0])
        ax.set_ylim(0, 16)
        F.ygrid(ax)

    ax = fig.add_subplot(gs[1, 0])
    dr = pd.read_csv(XAI / "residual_by_dist_rider.csv")
    order_b = list(dict.fromkeys(dr["dist_band"]))
    dr["k"] = dr["dist_band"].map({b: i for i, b in enumerate(order_b)})
    gap_lines(ax, dr, "k")
    ax.set_xticks(range(len(order_b)), ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-6", ">6"], rotation=45,
                  fontsize=7)
    ax.set_xlabel("Trip distance (km)")
    ax.set_ylabel("Obs. − pred. (pp)")
    ax.legend(fontsize=7, loc="lower left")
    F.panel(fig, ax, "b", "By distance")

    ax = fig.add_subplot(gs[1, 1])
    gap_lines(ax, pd.read_csv(XAI / "residual_by_income_rider.csv"), "income_decile")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Origin income decile")
    F.panel(fig, ax, "c", "By income")

    ax = fig.add_subplot(gs[2, 0])
    gap_lines(ax, pd.read_csv(XAI / "residual_by_hour_rider.csv"), "hour")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Start hour")
    ax.set_ylabel("Obs. − pred. (pp)")
    F.panel(fig, ax, "e", "By hour")

    ax = fig.add_subplot(gs[2, 1])
    mr = pd.read_csv(XAI / "residual_by_month_rider.csv")
    w = 0.38
    for j, r in enumerate(("member", "casual")):
        d_ = mr[mr.rider == r].sort_values("month")
        ax.bar(d_["month"] + (j - 0.5) * w, d_["gap_pp"], width=w, color=RIDER[r][1])
    ax.axvline(3.5, color=F.INK2, lw=0.8, ls=":")
    ax.set_xticks(range(1, 11), list("JFMAMJJASO"))
    ax.set_xlabel("2026 month")
    ax.set_ylim(0, 22)
    F.ygrid(ax)
    F.panel(fig, ax, "f", "By month")

    tr = gpd.read_file(ROOT / "data" / "external" / "tiger" / "tracts_chicago.gpkg").to_crs(3857)
    tr = tr[tr["ALAND"] > 0.2 * (tr["ALAND"] + tr["AWATER"])]
    rt = pd.read_csv(XAI / "residual_by_tract.csv", dtype={"o_tract": str})
    g = tr.merge(rt, left_on="GEOID", right_on="o_tract", how="left")
    ax = fig.add_subplot(gs[1:, 2])
    tr.plot(ax=ax, color=F.tint(F.MUTED, 0.78), edgecolor="white", lw=0.08)
    lim = 20
    has = g[g["gap_pp"].notna()]
    has.plot(ax=ax, column="gap_pp", cmap=F.DIV, vmin=-lim, vmax=lim, edgecolor="white", lw=0.1)
    xmin, ymin, xmax, ymax = has.total_bounds
    ax.set_xlim(xmin - 1500, xmax + 1500)
    ax.set_ylim(ymin - 1500, ymax + 1500)
    ax.set_axis_off()
    ax.set_aspect("equal")
    cax = ax.inset_axes([0.05, 0.06, 0.5, 0.025])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-lim, lim), cmap=F.DIV), cax=cax,
                      orientation="horizontal")
    cb.set_label("Obs. − pred. (pp)", fontsize=6.8, labelpad=1)
    cb.ax.tick_params(labelsize=6.3, length=0, pad=1)
    cb.outline.set_visible(False)
    ax.text(0.97, 0.99, f"{(has['gap_pp'] > 0).mean():.0%} of tracts\nunder-predicted", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, bbox=dict(fc="white", ec=F.RULE, boxstyle="round,pad=0.3", lw=0.6))
    F.panel(fig, ax, "d", "By origin tract")
    F.save(fig, FIG / "Fig14_drift_localization")
    captions["Fig14"] = (
        "**Figure 14. Where the 2026 under-prediction occurs.** All 3.6 million 2026 test trips with predictions "
        "from the locked main model. (a) Weekly observed (solid) and predicted (dashed) e-bike share for members and "
        "casual riders (weeks with at least 500 trips); the dotted line marks the March 16, 2026, price and fleet "
        "change. (b, c, e, f) Observed minus predicted share (percentage points) by trip distance, origin-tract "
        "median-income decile, start hour and month. (d) The same gap by origin census tract (at least 300 trips; "
        "colour range clipped at +/-20 pp; grey = fewer trips). A broad, near-uniform positive gap points to a shift "
        "in the level of e-bike choice rather than a change confined to particular trips or places.")

# =================================================================== Fig15
if "fig15" in ONLY:
    lc = pd.read_csv(TAB / "Table13_local_cases.csv")
    gtop = gi.query("model == 'main' and year == 2026 and kind == 'feature'").sort_values("rank")["name"].tolist()
    K = 12
    feats_dp = gtop[:K][::-1]
    rest = [f for f in FEATS if f not in feats_dp]
    short = lambda f: lab(f).replace("Origin: ", "O: ").replace("Dest.: ", "D: ")
    CASE_SHORT = {"Confident e-bike (correct)": "Confident e-bike", "Confident classic (correct)": "Confident classic",
                  "False positive": "False positive", "False negative": "False negative", "Borderline": "Borderline",
                  "Typical trip": "Typical trip", "Member, long trip": "Member, long trip",
                  "Casual, short trip": "Casual, short trip"}
    cases = list(dict.fromkeys(lc["case"]))
    fig = F.new_figure(180, 225)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.1])
    bot = outer[1].subgridspec(2, 2)

    ax = fig.add_subplot(top[0])
    m26 = np.where(smp.year.values == 2026)[0]
    idx = rng.choice(m26, 300, replace=False)
    base = sh["base"].iloc[0]
    rest_sum = sh.loc[idx, [f"shap_{f}" for f in rest]].sum(1).values
    cum = np.column_stack([np.full(len(idx), base), base + rest_sum] +
                          [np.zeros(len(idx))] * len(feats_dp))
    for k, f in enumerate(feats_dp):
        cum[:, k + 2] = cum[:, k + 1] + sh.loc[idx, f"shap_{f}"].values
    nrm = mpl.colors.Normalize(-1.5, 1.5)
    yv = np.arange(cum.shape[1])
    for row in cum[np.argsort(np.abs(cum[:, -1]))]:
        ax.plot(row, yv, color=F.DIV(nrm(row[-1])), lw=0.5, alpha=0.45)
    lcp = lc.pivot_table(index="case", columns="feature", values="shap", sort=False)
    ends, lines_ = [], []
    for n_, cname in enumerate(cases):
        r = lcp.loc[cname]
        b0 = lc[lc.case == cname]["base"].iloc[0]
        line = [b0, b0 + r[rest].sum()]
        for f in feats_dp:
            line.append(line[-1] + r[f])
        ax.plot(line, yv, color=F.INK, lw=1.6, zorder=4)
        ends.append((line[-1], n_ + 1))
    placed = []
    for xe, num in sorted(ends):
        lvl = 0
        while any(abs(xe - xp) < 0.45 and lp == lvl for xp, lp in placed):
            lvl += 1
        placed.append((xe, lvl))
        ax.text(xe, yv[-1] + 0.35 + 0.62 * lvl, str(num), ha="center", va="bottom", fontsize=6.2, color="white",
                fontweight="bold", zorder=5, bbox=dict(boxstyle="circle,pad=0.12", fc=F.INK, ec="none"))
    top_lvl = max(l_ for _, l_ in placed)
    ax.axvline(base, color=F.INK2, lw=0.8, ls="--")
    ax.set_yticks(yv, ["Base value", f"{len(rest)} other features"] + [short(f) for f in feats_dp], fontsize=6.8)
    ax.set_ylim(-0.4, yv[-1] + 1.2 + 0.62 * top_lvl)
    ax.set_xlabel("Model output (log-odds)")
    cax = ax.inset_axes([0.62, 0.05, 0.34, 0.025])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=F.DIV), cax=cax, orientation="horizontal")
    cb.set_label("Final log-odds", fontsize=6.4, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=0, pad=1)
    cb.outline.set_visible(False)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Decision paths")

    ax = fig.add_subplot(top[1])
    cols_h = gtop[:10]
    H = lcp.loc[cases, cols_h].values
    lim = np.abs(H).max()
    im = ax.imshow(H, cmap=F.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, F.minus(f"{H[i, j]:+.2f}"), ha="center", va="center", fontsize=5.6,
                    color=F.text_on(im.cmap(im.norm(H[i, j]))))
    pinfo = lc.groupby("case", sort=False).agg(p=("p", "first"), y=("y", "first")).loc[cases]
    ax.set_yticks(range(len(cases)), [f"{k + 1}  {CASE_SHORT[c]}" for k, c in enumerate(cases)], fontsize=6.8)
    ax.set_xticks(range(len(cols_h)), [short(f) for f in cols_h], rotation=60, ha="right", fontsize=6.4)
    for i, (pp, yy_) in enumerate(zip(pinfo["p"], pinfo["y"])):
        ax.text(len(cols_h) - 0.35, i, f"p={pp:.2f} {'E' if yy_ == 1 else 'C'}", va="center", ha="left", fontsize=6.2)
    ax.set_xlim(-0.5, len(cols_h) + 1.6)
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    F.panel(fig, ax, "b", "Case × feature SHAP")

    fmtv = {"dist_km": "{:.1f} km", "temp_c": "{:.0f} °C", "hour": "{:.0f} h", "o_dist_cbd_km": "{:.1f} km",
            "is_member": "{:.0f}", "is_weekend": "{:.0f}", "is_daylight": "{:.0f}",
            "o_pop_density_km2": "{:,.0f}/km²", "d_pop_density_km2": "{:,.0f}/km²",
            "o_median_income": "${:,.0f}", "d_median_income": "${:,.0f}"}
    for k, cname in enumerate(["False positive", "False negative", "Member, long trip", "Casual, short trip"]):
        ax = fig.add_subplot(bot[k // 2, k % 2])
        d_ = lc[lc.case == cname]
        d_ = d_.reindex(d_["shap"].abs().sort_values(ascending=False).index)
        topd, restd = d_.head(6), d_.iloc[6:]
        labels = []
        for _, r in topd.iterrows():
            v = r["value"]
            sv = (fmtv.get(r["feature"], "{:.1f}%" if "pct" in r["feature"] else "{:.2g}").format(v)
                  if pd.notna(v) else "NA")
            labels.append(f"{short(r['feature'])} = {sv}")
        labels.append(f"{len(restd)} other features")
        P.waterfall(ax, labels, list(topd["shap"]) + [restd["shap"].sum()], base=d_["base"].iloc[0])
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 - 0.12 * (x1 - x0), x1 + 0.08 * (x1 - x0))
        ax.tick_params(axis="y", labelsize=6.6)
        r0 = d_.iloc[0]
        ax.set_xlabel(f"log-odds; p = {r0['p']:.2f}, observed: {'e-bike' if r0['y'] == 1 else 'classic'}", fontsize=7.2)
        F.panel(fig, ax, "cdef"[k], f"{cases.index(cname) + 1}  {CASE_SHORT[cname]}")
    F.save(fig, FIG / "Fig15_local_cases")
    captions["Fig15"] = (
        "**Figure 15. Local explanations for 2026 test trips.** (a) Decision paths: each line follows one trip from the "
        "model's base value (dashed) through the summed contribution of the less important features and then the 12 "
        "most important features (bottom to top) to its final log-odds; 300 random 2026 trips are coloured by their "
        "final output, and the eight selected cases are drawn in black and numbered. (b) SHAP values (log-odds) of "
        "the ten most important features for the eight cases, with the predicted probability and the observed bike "
        "(E = e-bike, C = classic). Cases: (1, 2) most confident correct e-bike and classic predictions, (3, 4) most "
        f"confident false positive and false negative at the validation threshold ({THR:.2f}), (5) closest to the "
        "threshold, (6) median prediction, (7) member on a long trip (top distance decile) and (8) casual rider on "
        "a short trip (bottom quintile), all distinct trips. (c-f) Waterfalls for cases 3, 4, 7 and 8 with the six "
        "largest contributions; the rest are summed.")

# =================================================================== Fig16
if "fig16" in ONLY and (TAB / "Table15_rolling_refit.csv").exists():
    seed = pd.read_csv(XAI / "seed_importance.csv", index_col=0)
    tt = pd.read_csv(TAB / "Table14b_twin_transfer.csv")
    ta = pd.read_csv(XAI / "twin_ale.csv")
    rl = pd.read_csv(TAB / "Table15_rolling_refit.csv")
    fig = F.new_figure(180, 150)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    top = seed["locked"].sort_values(ascending=False).index[:10][::-1]
    yy = np.arange(len(top))
    for c in seed.columns[1:]:
        ax.scatter(seed.loc[top, c], yy, s=10, color=F.MUTED, zorder=2)
    ax.scatter(seed.loc[top, "locked"], yy, s=22, marker="D", color=F.G1, zorder=3, label="Locked model")
    ax.scatter([], [], s=10, color=F.MUTED, label="5 seed refits")
    ax.set_yticks(yy, [lab(f).replace("Origin: ", "O: ").replace("Dest.: ", "D: ") for f in top], fontsize=7.2)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, loc="lower right")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Seed stability")

    ax = fig.add_subplot(gs[0, 1])
    M = tt.groupby(["train_year", "eval_year"])["roc_auc"].mean().unstack()
    im = ax.imshow(M.values, cmap=F.HEAT, vmin=M.values.min() - 0.01, vmax=M.values.max() + 0.005)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold",
                    color=F.text_on(im.cmap(im.norm(M.values[i, j]))))
    ax.set_xticks([0, 1], ["2025", "2026"])
    ax.set_yticks([0, 1], ["2025", "2026"])
    ax.set_xlabel("Evaluated on (held-out days)")
    ax.set_ylabel("Twin trained on")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    F.panel(fig, ax, "b", "Cross-year AUC")

    ax = fig.add_subplot(gs[0, 2])
    lv = ta[ta.feature == "__level__"].groupby("train_year")["ale"].agg(["mean", "min", "max"])
    ax.bar([0, 1], 1 / (1 + np.exp(-lv["mean"])), color=[F.MUTED, F.G1], width=0.55)
    for i, (yr, r) in enumerate(lv.iterrows()):
        ax.text(i, 1 / (1 + np.exp(-r["mean"])) + 0.01, f"{1 / (1 + np.exp(-r['mean'])):.1%}", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["2025 twin", "2026 twin"])
    F.pct_axis(ax)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Mean prediction, same trips")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Level shift")

    for k, f in enumerate(["dist_km", "temp_c"]):
        ax = fig.add_subplot(gs[1, k])
        for yr, col in ((2025, F.MUTED), (2026, F.G1)):
            for r, ls in (("member", "-"), ("casual", "--")):
                d = ta[(ta.feature == f) & (ta.train_year == yr) & (ta.subset == r)]
                g = d.groupby("x")["ale"].agg(["mean", "min", "max"]).reset_index()
                if f == "dist_km":
                    g = g[g.x <= 8]
                ax.fill_between(g.x, g["min"], g["max"], color=col, alpha=0.2, lw=0)
                ax.plot(g.x, g["mean"], color=col, ls=ls, lw=1.5)
        ax.axhline(0, color=F.RULE, lw=0.8)
        ax.set_xlabel(UNIT[f])
        ax.set_ylabel("ALE (log-odds)")
        F.ygrid(ax)
        F.panel(fig, ax, "de"[k], ["Distance effect", "Temperature effect"][k])
    h = [Line2D([], [], color=F.MUTED, lw=1.6, label="2025 twin"), Line2D([], [], color=F.G1, lw=1.6, label="2026 twin"),
         Line2D([], [], color=F.INK2, lw=1.2, label="Members"), Line2D([], [], color=F.INK2, lw=1.2, ls="--", label="Casual")]
    ax.legend(handles=h, fontsize=6.6, loc="upper right", ncol=1, handlelength=1.6)

    ax = fig.add_subplot(gs[1, 2])
    VAR = {"frozen": ("Frozen", F.INK2, "o"), "intercept_update": ("Intercept update", F.G0, "s"),
           "monthly_refit": ("Monthly refit", F.G1, "D")}
    for v, (name, col, mk) in VAR.items():
        d = rl[rl.variant == v]
        ax.plot(range(len(d)), d["log_loss"], marker=mk, color=col, ms=3.2, lw=1.2, label=name)
    ax.set_xticks(range(10), list("JFMAMJJASO"))
    ax.set_xlabel("2026 month")
    ax.set_ylabel("Test log loss")
    ax.legend(fontsize=7, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "f", "Cost of drift")
    F.save(fig, FIG / "Fig16_robustness")
    captions["Fig16"] = (
        "**Figure 16. Stability, effect drift and the cost of drift (post hoc analyses).** (a) Mean |SHAP| of the top "
        "10 features for the locked model and five refits with different seeds (same 20,000 2026 trips). (b) AUC of "
        "twin models with identical settings trained on February-October 2025 or 2026 (every 5th day held out; equal "
        "training size; mean of 3 replicates with different seeds and 80% subsamples), evaluated on held-out days "
        "of each year. (c) Mean predicted e-bike share of each twin on the same held-out reference trips. (d, e) ALE "
        "of distance and temperature from each twin on the same reference trips; bands span the 3 replicates. "
        "(f) Monthly 2026 test log loss of the frozen locked model, the frozen model with its intercept re-estimated "
        "on the 28 days ending 7 days before the month, and a full refit on all trips up to 7 days before the month.")

# ------------------------------------------------------------------ captions
cap_file = FIG / "figure_captions_xai.md"
old = {}
if cap_file.exists():
    for block in cap_file.read_text().split("\n\n")[1:]:
        if block.startswith("**Figure"):
            old[f"Fig{block.split('Figure ')[1].split('.')[0].zfill(2)}"] = block.strip()
old.update(captions)
cap_file.write_text("# Explanation figure captions (Chicago)\n\n" +
                    "\n\n".join(old[k] for k in sorted(old)) + "\n")
print("figures done:", ", ".join(captions))
