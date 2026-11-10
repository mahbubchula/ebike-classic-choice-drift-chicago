"""Step 13: LaTeX tables for the manuscript, built only from saved results and the modelling sample.

Writes outputs/manuscript_tables/tab_*.tex (booktabs; numbers rounded for print).

Run:  python src/step_13_manuscript_tables.py
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_SETS, OUT, TARGET, load  # noqa: E402

MS = OUT / "manuscript_tables"
MS.mkdir(parents=True, exist_ok=True)
TAB = OUT / "tables"


def write(name, body, caption, label, cols, note=None, size="\\small", wide=False):
    tab = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *body, "\\bottomrule", "\\end{tabular}"]
    if wide:
        tab = ["\\resizebox{\\textwidth}{!}{%", *tab, "}"]
    tex = ["\\begin{table}[!htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}", *tab]
    if note:
        tex.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\footnotesize {note}}}")
    tex.append("\\end{table}")
    (MS / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print("wrote", name)


def esc(s):
    return str(s).replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


# ------------------------------------------------------------------ Table 1: sample by year and bike type
VARS = [("dist_km", "Trip distance (km)", 2), ("temp_c", "Temperature (\\textdegree C)", 1),
        ("precip_mm", "Precipitation (mm/h)", 2), ("wind_kmh", "Wind speed (km/h)", 1), ("hour", "Start hour", 1),
        ("o_dist_cbd_km", "Origin distance to the Loop (km)", 2), ("o_median_income", "Origin median income (US\\$000)", 1),
        ("o_pct_hh_no_car", "Origin car-free households (\\%)", 1),
        ("o_pct_commute_transit", "Origin transit commuters (\\%)", 1)]
BINV = [("is_member", "Annual member (\\%)"), ("is_weekend", "Weekend (\\%)"), ("is_daylight", "Daylight (\\%)"),
        ("round_trip", "Round trip (\\%)")]
cols = list(dict.fromkeys([v for v, *_ in VARS] + [v for v, _ in BINV] + [TARGET, "started_at"]))
parts = []
for split, yr in (("final_train", 2025), ("test", 2026)):
    d = pd.read_parquet(OUT.parent / "data" / "processed" / "splits" / f"{split}.parquet", columns=cols + ["end_dock"])
    d = d[d["end_dock"] == 1]
    d["year"] = yr
    parts.append(d)
d = pd.concat(parts, ignore_index=True)
d["o_median_income"] = d["o_median_income"] / 1000
groups = [(2025, 1), (2025, 0), (2026, 1), (2026, 0)]
body = ["& \\multicolumn{2}{c}{2025 (training)} & \\multicolumn{2}{c}{2026 (test)} & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Variable & E-bike & Classic & E-bike & Classic & SMD \\\\", "\\midrule"]
n = [((d.year == y) & (d[TARGET] == e)).sum() for y, e in groups]
body.append("Trips & " + " & ".join(f"{v:,}" for v in n) + " & \\\\")
for v, lab, dp in VARS:
    cells = []
    for y, e in groups:
        s = d.loc[(d.year == y) & (d[TARGET] == e), v]
        cells.append(f"{s.mean():.{dp}f} ({s.std():.{dp}f})")
    a, b = d.loc[d[TARGET] == 1, v], d.loc[d[TARGET] == 0, v]
    smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    body.append(f"{lab} & " + " & ".join(cells) + f" & {smd:+.2f} \\\\".replace("+-", "-"))
for v, lab in BINV:
    cells = [f"{100 * d.loc[(d.year == y) & (d[TARGET] == e), v].mean():.1f}" for y, e in groups]
    a, b = d.loc[d[TARGET] == 1, v], d.loc[d[TARGET] == 0, v]
    smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    body.append(f"{lab} & " + " & ".join(cells) + f" & {smd:+.2f} \\\\")
share = [100 * d.loc[d.year == y, TARGET].mean() for y in (2025, 2026)]
write("tab_sample", body,
      "Modeling sample: classic dock-to-dock trips by year and bike type. Mean (SD) for continuous variables, "
      "percentage for binary variables; SMD = standardized mean difference between e-bike and classic trips "
      "(both years pooled).", "tab:sample", "lrrrrr", wide=True,
      note=f"E-bike share: {share[0]:.1f}\\% in 2025 (5 Feb to 31 Dec) and {share[1]:.1f}\\% in 2026 (1 Jan to 31 Oct). "
           "Distances are straight-line distances between canonical station locations.")
del d

# ------------------------------------------------------------------ Table 2: temporal design
t5 = pd.read_csv(TAB / "Table5_data_split.csv").set_index("set")
use = {"excluded_pre": "Robustness check only (before 5 Feb 2025 price change)", "inner_train": "Fit candidate models",
       "embargo": "Discarded (7-day gap)", "valid": "Early stopping, tuning, threshold",
       "final_train": "Refit of the selected configuration", "test": "Single evaluation (locked)"}
nm = {"excluded_pre": "Excluded", "inner_train": "Inner training", "embargo": "Embargo", "valid": "Validation",
      "final_train": "Final training", "test": "Test"}
body = ["Set & Period & Trips & E-bike (\\%) & Members (\\%) & Use \\\\", "\\midrule"]
for k in ["excluded_pre", "inner_train", "embargo", "valid", "final_train", "test"]:
    r = t5.loc[k]
    body.append(f"{nm[k]} & {r['start']} to {r['end']} & {int(r['trips_dock_to_dock']):,} & "
                f"{r['ebike_share_dock_to_dock_pct']:.1f} & {r['member_share_dock_to_dock_pct']:.1f} & {use[k]} \\\\")
write("tab_split", body, "Temporal design (classic dock-to-dock trips).", "tab:split", "llrrrl")

# ------------------------------------------------------------------ Table 3: predictive performance
t6 = pd.read_csv(TAB / "Table6_candidate_models_valid_main.csv").set_index("model")
t8 = pd.read_csv(TAB / "Table8_test_overall.csv")
vp = {s: json.loads((OUT / "reports" / f"07_best_params_{s}.json").read_text()) for s in
      ("main", "main_plus_supply", "origin_only")}
body = ["& \\multicolumn{3}{c}{Validation, Sep to Dec 2025} & \\multicolumn{6}{c}{Test, Jan to Oct 2026} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-10}",
        "Model & AUC & Log loss & Brier & AUC [95\\% CI] & PR-AUC & Log loss & Brier & Bal. acc. & Cal. int. \\\\",
        "\\midrule", "\\multicolumn{10}{l}{\\textit{Untuned candidates, main feature set}} \\\\"]
for m in ["Dummy", "LogReg", "XGBoost", "CatBoost", "LightGBM"]:
    r = t6.loc[m]
    body.append(f"{m} & {r['roc_auc']:.3f} & {r['log_loss']:.3f} & {r['brier']:.3f} & & & & & & \\\\")
body.append("\\midrule")
body.append("\\multicolumn{10}{l}{\\textit{Tuned and locked models}} \\\\")
lab = {("main", "LightGBM"): "LightGBM, main", ("main", "LogReg"): "Spline LR, main",
       ("main_plus_supply", "LightGBM"): "LightGBM, + supply", ("main_plus_supply", "LogReg"): "Spline LR, + supply",
       ("origin_only", "LightGBM"): "LightGBM, origin only", ("origin_only", "LogReg"): "Spline LR, origin only"}
for (fs, md), name in lab.items():
    r = t8[(t8.feature_set == fs) & (t8.model == md)].iloc[0]
    lo, hi = ast.literal_eval(r["roc_auc_ci95"])
    if md == "LightGBM":
        v = vp[fs]["inner_model_valid_metrics"]
        val = f"{v['roc_auc']:.3f} & {v['log_loss']:.3f} & {v['brier']:.3f}"
    else:
        val = "& &"
    body.append(f"{name} & {val} & {r['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}] & {r['pr_auc']:.3f} & {r['log_loss']:.3f} & "
                f"{r['brier']:.3f} & {r['balanced_acc']:.3f} & {r['cal_intercept']:+.2f} \\\\")
write("tab_performance", body,
      "Predictive performance. Validation metrics guided every modeling choice; the test set was evaluated once.",
      "tab:performance", "lrrrrrrrrr",
      note="CI: day-block bootstrap (200 replicates). Bal. acc.: balanced accuracy at the threshold chosen on validation "
           "(0.450 for the main model). Cal. int.: calibration intercept (positive = e-bike use under-predicted). "
           "The origin-only models use all trips starting at a classic dock, so their base rate differs.",
      size="\\scriptsize", wide=True)

# ------------------------------------------------------------------ Table 4: drift decomposition
s = pd.read_csv(TAB / "Table11a_drift_summary.csv")
dec = pd.read_csv(TAB / "Table11_drift_decomposition.csv")
body = ["& \\multicolumn{3}{c}{Main model} & \\multicolumn{3}{c}{Main + supply proxy} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Quantity & All & Members & Casual & All & Members & Casual \\\\",
        "\\midrule"]


def cell(m, sb, k, pct=False, dp=3):
    r = s[(s.model == m) & (s.subset == sb)].iloc[0]
    f = 100 if pct else 1
    return f"{f * r[k]:.{dp}f} [{f * r[k + '_lo']:.{dp}f}, {f * r[k + '_hi']:.{dp}f}]"


for k, labl, pct, dp in (("observed_change_pp", "Observed change (pp)", False, 1),
                         ("predicted_change_pp", "Predicted change (pp)", False, 1),
                         ("observed_change_logodds", "Observed change (log-odds)", False, 3),
                         ("explained_change_logodds", "Explained change (log-odds)", False, 3),
                         ("share_explained", "Share explained (\\%)", True, 0)):
    body.append(f"{labl} & " + " & ".join(cell(m, sb, k, pct, dp) for m in ("main", "main_plus_supply")
                                           for sb in ("all", "member", "casual")) + " \\\\")
body.append("\\midrule")
body.append("\\multicolumn{7}{l}{\\textit{Contribution of feature groups (log-odds), all riders}} \\\\")
for g in ["Trip", "Rider", "Time", "Weather", "Origin area", "Destination area", "Station supply"]:
    cells = []
    for m in ("main", "main_plus_supply"):
        r = dec[(dec.model == m) & (dec.subset == "all") & (dec.kind == "group") & (dec.name == g)]
        cells.append(f"{r['delta_logodds'].iloc[0]:+.3f} [{r['lo'].iloc[0]:+.3f}, {r['hi'].iloc[0]:+.3f}]"
                     if len(r) else "")
    body.append(f"{g} & \\multicolumn{{3}}{{c}}{{{cells[0]}}} & \\multicolumn{{3}}{{c}}{{{cells[1]}}} \\\\")
write("tab_drift", body,
      "Season-matched decomposition of the change in e-bike choice from 2025 to 2026 (February to October, months "
      "weighted equally), with day-block bootstrap 95\\% CIs (500 replicates).", "tab:drift", "lcccccc",
      note="Explained change: change in the log-odds of the mean predicted probability. Share explained: explained over "
           "observed change. The supply proxy is the station's e-bike share of departures over the previous 7 days.",
      size="\\scriptsize", wide=True)

# ------------------------------------------------------------------ Table 5: robustness and post hoc analyses
rb = pd.read_csv(TAB / "Table16_robustness.csv")
sd = pd.read_csv(TAB / "Table14a_seed_stability.csv")
tw = pd.read_csv(TAB / "Table14b_twin_transfer.csv").groupby(["train_year", "eval_year"])["roc_auc"].agg(["mean", "std"])
rl = pd.read_csv(TAB / "Table15_rolling_refit.csv")
rj = json.loads((OUT / "reports" / "12_robustness.json").read_text())
body = ["Analysis & Result \\\\", "\\midrule"]
for _, r in rb.iterrows():
    body.append(f"{esc(r['check'])}, {esc(r['subset'])} & AUC {r['roc_auc']:.3f}; log loss {r['log_loss']:.3f}; "
                f"mean predicted {100 * r['mean_pred']:.1f}\\% vs observed {100 * r['observed_rate']:.1f}\\% \\\\")
body.append(f"R1 month as a feature: rank of month in mean $|$SHAP$|$ & {rj['R1_month_rank']} of "
            f"{len(FEATURE_SETS['main_plus_month'])}; share of 2025 to 2026 change explained "
            f"{100 * rj['R1_share_explained_main_plus_month']:.0f}\\% \\\\")
body.append(f"Seed stability (5 refits) & Spearman $\\rho$ of importance with locked model "
            f"{sd['spearman_vs_locked'].min():.2f} to {sd['spearman_vs_locked'].max():.2f}; top-10 overlap "
            f"{sd['top10_overlap'].min():.1f} to {sd['top10_overlap'].max():.1f} \\\\")
body.append(f"Twin models, AUC on held-out days & 2025 model: {tw.loc[(2025, 2025), 'mean']:.3f} (2025), "
            f"{tw.loc[(2025, 2026), 'mean']:.3f} (2026); 2026 model: {tw.loc[(2026, 2026), 'mean']:.3f} (2026), "
            f"{tw.loc[(2026, 2025), 'mean']:.3f} (2025) \\\\")
w = rl.groupby("variant").apply(lambda g: pd.Series({k: np.average(g[k], weights=g["n"])
                                                     for k in ("roc_auc", "log_loss", "cal_intercept")}))
for v, nmv in (("frozen", "Frozen model"), ("intercept_update", "Monthly intercept update"),
               ("monthly_refit", "Monthly refit")):
    body.append(f"{nmv}, 2026 (trip-weighted) & AUC {w.loc[v, 'roc_auc']:.3f}; log loss {w.loc[v, 'log_loss']:.3f}; "
                f"calibration intercept {w.loc[v, 'cal_intercept']:+.2f} \\\\")
write("tab_robust", body, "Robustness checks and post hoc analyses. None of them changes the locked models or the "
      "locked test result.", "tab:robust", "p{0.36\\linewidth}p{0.58\\linewidth}", size="\\scriptsize")
