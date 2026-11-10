"""Step 04b: Fig08 feature correlation and Table 3 (Figs 4-7 moved to step_04_descriptive_figures.py).

Fig04_time_heatmaps    e-bike share by day x hour, members vs. casual, 2025 vs. 2026 and change
Fig05_trip_profile     distance and speed rainclouds; e-bike share by distance and by month, rider x year
Fig06_weather_context  temperature x rain heat map; wind; temperature by rider; context dumbbell
Fig07_neighbourhood    e-bike share across origin-tract deciles of six neighbourhood measures, 2025 vs. 2026
Fig08_feature_correlation  Spearman correlation of candidate model features (pre-modelling check)
Tables: Table3_high_correlation_pairs.csv (|rho| >= 0.7)

Descriptive only: trip duration and speed are shown here but are NOT model features.

Run:  python src/step_04b_feature_correlation.py
"""
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fstudio as F  # noqa: E402
import plots as P  # noqa: E402

F.use_theme("atlas_white")
mpl.rcParams.update({"font.weight": "bold"})
OUT = ROOT / "outputs"
FIG, TAB = OUT / "figures", OUT / "tables"
SEED = 42
rng = np.random.default_rng(SEED)

COLS = ["e_bike", "is_member", "year", "month", "hour", "day_of_week", "is_weekend", "is_holiday",
        "dist_km", "duration_min", "round_trip", "same_tract", "o_dist_cbd_km", "d_dist_cbd_km",
        "temp_c", "feels_like_c", "precip_mm", "precip_prev3h_mm", "snow_cm", "wind_kmh", "humidity_pct",
        "cloud_pct", "is_daylight", "o_pop_density_km2", "o_job_density_km2", "o_median_income",
        "o_median_age", "o_pct_age_18_29", "o_pct_degree", "o_pct_hh_no_car", "o_pct_commute_transit",
        "o_pct_commute_bicycle", "o_pct_commute_walk", "o_pct_commute_wfh", "o_pct_jobs_retail_food"]
d = pd.read_parquet(ROOT / "data" / "processed" / "chicago_model_data.parquet", columns=COLS)
print(f"Loaded {len(d):,} trips")

RIDER = {1: ("Members", F.S4[0]), 0: ("Casual", F.S4[1])}
YEAR_LS = {2025: "--", 2026: "-"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
captions = []


def rider_year_legend(ax, loc="lower right"):
    h = [Line2D([], [], color=RIDER[m][1], lw=1.8, label=RIDER[m][0]) for m in (1, 0)]
    h += [Line2D([], [], color=F.INK2, lw=1.4, ls=YEAR_LS[y], label=str(y)) for y in (2025, 2026)]
    ax.legend(handles=h, loc=loc, ncol=2, fontsize=7.8, columnspacing=1.0)


def share_lines(ax, key, x_of=None, min_n=300):
    """E-bike share by `key` for every rider x year, with Wilson 95% bands."""
    g = d.groupby(["is_member", "year", key], observed=True)["e_bike"].agg(["sum", "size"])
    for m in (1, 0):
        for y in (2025, 2026):
            s = g.loc[(m, y)]
            s = s[s["size"] >= min_n]
            x = np.arange(len(s)) if x_of is None else x_of(s.index)
            p, lo, hi = F.wilson(s["sum"].values, s["size"].values)
            ax.fill_between(x, lo, hi, color=RIDER[m][1], alpha=0.15, lw=0)
            ax.plot(x, p, color=RIDER[m][1], ls=YEAR_LS[y], marker="o", ms=2.6, lw=1.5)
    return g


# =================================================================== Fig04 heat maps
fig = F.new_figure(180, 118)
gs = fig.add_gridspec(2, 3)
hm = d.groupby(["is_member", "year", "day_of_week", "hour"])["e_bike"].mean()
vmin, vmax = np.nanpercentile(hm.values, [2, 98])
chg_all = []
for r, m in enumerate((1, 0)):
    for c, y in enumerate((2025, 2026)):
        chg_all.append(hm.loc[(m, y)].unstack())
lim = np.nanpercentile(np.abs(np.concatenate([(chg_all[1] - chg_all[0]).values.ravel(),
                                              (chg_all[3] - chg_all[2]).values.ravel()])), 98)
axes_share, axes_chg = [], []
letters = iter("abcdef")
for r, m in enumerate((1, 0)):
    mats = [hm.loc[(m, y)].unstack() for y in (2025, 2026)]
    for c in range(3):
        ax = fig.add_subplot(gs[r, c])
        if c < 2:
            im1 = ax.imshow(mats[c].values, aspect="auto", cmap=F.SEQ, vmin=vmin, vmax=vmax)
            axes_share.append(ax)
            title = f"{RIDER[m][0]}, {2025 + c}"
        else:
            im2 = ax.imshow(100 * (mats[1] - mats[0]).values, aspect="auto", cmap=F.DIV,
                            vmin=-100 * lim, vmax=100 * lim)
            axes_chg.append(ax)
            title = f"{RIDER[m][0]}, change"
        ax.set_yticks(range(7), DAYS if c == 0 else [""] * 7)
        ax.set_xticks(range(0, 24, 6))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 1:
            ax.set_xlabel("Start hour")
        F.panel(fig, ax, next(letters), title)
fig.colorbar(im1, ax=axes_share, location="right", shrink=0.75, aspect=30, pad=0.01,
             label="E-bike share", format=mpl.ticker.PercentFormatter(1, 0))
fig.colorbar(im2, ax=axes_chg, location="right", shrink=0.75, aspect=30, pad=0.02,
             label="Change, 2025 to 2026 (pp)")
plt.close(fig)  # Fig04_time_heatmaps now drawn by step_04_descriptive_figures.py
captions.append("**Figure 4. When riders choose e-bikes.** E-bike share of trips by day of week and start hour "
                "for annual members (a–c) and casual riders (d–f). Columns show 2025, 2026 and the change "
                "from 2025 to 2026 in percentage points (pp). Share panels use a common colour scale.")

# =================================================================== Fig05 trip profile
fig = F.new_figure(180, 140)
gs = fig.add_gridspec(2, 2)
samp = d.groupby("e_bike", group_keys=False).sample(n=3000, random_state=SEED)
samp = samp.assign(speed=samp["dist_km"] / (samp["duration_min"] / 60))
ax = fig.add_subplot(gs[0, 0])
cap = samp["dist_km"].quantile(0.99)
P.raincloud(ax, [samp.loc[samp.e_bike == 0, "dist_km"].clip(upper=cap),
                 samp.loc[samp.e_bike == 1, "dist_km"].clip(upper=cap)],
            ["Classic", "E-bike"], colors=[F.G0, F.G1], unit="km")
ax.set_xlabel("Straight-line trip distance (km)")
F.panel(fig, ax, "a", "Trip distance")
ax = fig.add_subplot(gs[0, 1])
sp = samp[samp["round_trip"] == 0]
cap = sp["speed"].quantile(0.99)
P.raincloud(ax, [sp.loc[sp.e_bike == 0, "speed"].clip(upper=cap), sp.loc[sp.e_bike == 1, "speed"].clip(upper=cap)],
            ["Classic", "E-bike"], colors=[F.G0, F.G1], unit="km/h")
ax.set_xlabel("Straight-line speed (km/h)")
F.panel(fig, ax, "b", "Trip speed")

ax = fig.add_subplot(gs[1, 0])
DB = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
d["dist_bin"] = pd.cut(d["dist_km"], DB)
share_lines(ax, "dist_bin", x_of=lambda idx: np.array([iv.mid for iv in idx]))
ax.set_xlabel("Trip distance (km, bin midpoint)")
ax.set_ylabel("E-bike share")
F.pct_axis(ax)
F.ygrid(ax)
rider_year_legend(ax)
F.panel(fig, ax, "c", "Share by distance")

ax = fig.add_subplot(gs[1, 1])
share_lines(ax, "month", x_of=lambda idx: np.asarray(idx))
ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
ax.set_xlabel("Calendar month")
F.pct_axis(ax)
F.ygrid(ax)
F.panel(fig, ax, "d", "Share by month")
plt.close(fig)  # Fig05_trip_profile now drawn by step_04_descriptive_figures.py
captions.append("**Figure 5. Trip profile.** (a) Straight-line distance and (b) straight-line speed of a random "
                "sample of 3,000 trips per bike type (values above the 99th percentile capped; round trips "
                "excluded from b). Boxes show the interquartile range, the coloured bar the median and the "
                "whisker line the 2.5th to 97.5th percentiles; SMD is the standardised mean difference. Speed and "
                "duration are descriptive only and are not model features. (c) E-bike share by trip distance and "
                "(d) by calendar month for members and casual riders in 2025 (dashed) and 2026 (solid); bands are "
                "Wilson 95% confidence intervals. October is the last month observed in 2026.")

# =================================================================== Fig06 weather and context
fig = F.new_figure(180, 140)
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
TB = [-25, -5, 0, 5, 10, 15, 20, 25, 30, 40]
TL = ["<−5", "−5–0", "0–5", "5–10", "10–15", "15–20", "20–25", "25–30", ">30"]
RB = [-0.01, 0.0, 0.5, 2, 100]
RL = ["Dry", "<0.5", "0.5–2", ">2"]
g = d.groupby([pd.cut(d["temp_c"], TB), pd.cut(d["precip_mm"], RB)], observed=False)["e_bike"].agg(["mean", "size"])
M = g["mean"].unstack().values
N = g["size"].unstack().values
M = np.where(N >= 300, M, np.nan)
im = ax.imshow(M, aspect="auto", cmap=F.SEQ, origin="lower")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7.2,
                    color=F.text_on(F.SEQ(im.norm(M[i, j]))))
ax.set_xticks(range(len(RL)), RL)
ax.set_yticks(range(len(TL)), TL)
ax.set_xlabel("Precipitation (mm/h)")
ax.set_ylabel("Temperature (°C)")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.85, aspect=25, pad=0.02, format=mpl.ticker.PercentFormatter(1, 0),
             label="E-bike share")
F.panel(fig, ax, "a", "Temperature × rain")

ax = fig.add_subplot(gs[0, 1])
WB = [0, 5, 10, 15, 20, 25, 30, 40, 60]
d["wind_bin"] = pd.cut(d["wind_kmh"], WB)
share_lines(ax, "wind_bin", x_of=lambda idx: np.array([iv.mid for iv in idx]))
ax.set_xlabel("Wind speed (km/h, bin midpoint)")
ax.set_ylabel("E-bike share")
F.pct_axis(ax)
F.ygrid(ax)
rider_year_legend(ax, loc="upper left")
F.panel(fig, ax, "b", "Wind")

ax = fig.add_subplot(gs[1, 0])
d["temp_bin"] = pd.cut(d["temp_c"], np.arange(-20, 41, 5))
share_lines(ax, "temp_bin", x_of=lambda idx: np.array([iv.mid for iv in idx]))
ax.set_xlabel("Temperature (°C, bin midpoint)")
ax.set_ylabel("E-bike share")
F.pct_axis(ax)
F.ygrid(ax)
F.panel(fig, ax, "c", "Temperature by rider")

ax = fig.add_subplot(gs[1, 1])
ctx = [("Dry", d["precip_mm"] <= 0.1), ("Raining", d["precip_mm"] > 0.1),
       ("Daylight", d["is_daylight"] == 1), ("Dark", d["is_daylight"] == 0),
       ("Weekday", d["is_weekend"] == 0), ("Weekend", d["is_weekend"] == 1),
       ("One-way", d["round_trip"] == 0), ("Round trip", d["round_trip"] == 1)]
mem = [d.loc[mask & (d.is_member == 1), "e_bike"].mean() for _, mask in ctx]
cas = [d.loc[mask & (d.is_member == 0), "e_bike"].mean() for _, mask in ctx]
# dumbbell drawn here so rider colours match every other panel (members blue, casual orange)
n_ctx = len(ctx)
for row, (a_, b_) in enumerate(zip(mem, cas)):
    y = n_ctx - 1 - row
    ax.plot([a_, b_], [y, y], color=F.tint(F.INK, 0.85), lw=6, solid_capstyle="butt", zorder=1)
    ax.scatter(a_, y, s=46, color=RIDER[1][1], ec=F.BG, lw=1, zorder=3)
    ax.scatter(b_, y, s=46, color=RIDER[0][1], ec=F.BG, lw=1, zorder=3)
    ax.text(1.0, y, F.minus(f"{100 * (b_ - a_):+.1f}"), transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8, color=F.INK)
ax.set_yticks(range(n_ctx), [c[0] for c in ctx][::-1])
F.clean_left(ax)
ax.set_ylim(-0.7, n_ctx - 0.3)
F.pct_axis(ax, "x")
F.ygrid(ax, "x")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=RIDER[m][1], mec=F.BG, ms=6, label=RIDER[m][0])
                   for m in (1, 0)], loc="upper left", fontsize=7.8)
lo_, hi_ = min(mem + cas), max(mem + cas)
ax.set_xlim(lo_ - 0.03, hi_ + 0.07)
ax.set_xlabel("E-bike share (right: casual − member, pp)")
F.panel(fig, ax, "d", "Context by rider")
plt.close(fig)  # Fig06_weather_context now drawn by step_04_descriptive_figures.py
captions.append("**Figure 6. Weather and trip context.** (a) E-bike share (%) by temperature and precipitation; "
                "cells with fewer than 300 trips are blank. (b, c) E-bike share by wind speed and temperature for "
                "members and casual riders in 2025 (dashed) and 2026 (solid) with Wilson 95% bands. (d) E-bike "
                "share under different trip conditions for members (blue) and casual riders (orange); numbers on the "
                "right give the casual minus member difference in percentage points.")

# =================================================================== Fig07 neighbourhood
NB = [("o_median_income", "Median income (USD 000)", 1e-3), ("o_pct_hh_no_car", "Zero-car households (%)", 1),
      ("o_pct_degree", "Bachelor's degree or more (%)", 1), ("o_job_density_km2", "Job density (000/km²)", 1e-3),
      ("o_pop_density_km2", "Population density (000/km²)", 1e-3),
      ("o_pct_commute_transit", "Transit commuters (%)", 1)]
fig = F.new_figure(180, 125)
gs = fig.add_gridspec(2, 3)
ymins, ymaxs, axs = [], [], []
for k, (var, lab, scale) in enumerate(NB):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    sub = d[["e_bike", "year", var]].dropna()
    sub["q"] = pd.qcut(sub[var].rank(method="first"), 10, labels=False)
    g = sub.groupby(["year", "q"]).agg(k=("e_bike", "sum"), n=("e_bike", "size"), x=(var, "median"))
    for y, col in ((2025, F.G0), (2026, F.G1)):
        s = g.loc[y]
        p, lo, hi = F.wilson(s["k"].values, s["n"].values)
        x = s["x"].values * scale
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, p, color=col, marker="o", ms=2.8, label=str(y))
        ymins.append(lo.min()); ymaxs.append(hi.max())
    if var == "o_job_density_km2":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(lab)
    if k % 3 == 0:
        ax.set_ylabel("E-bike share")
    F.ygrid(ax)
    axs.append(ax)
    F.panel(fig, ax, "abcdef"[k], ["Income", "Car ownership", "Education", "Jobs", "Population", "Transit use"][k])
for ax in axs:
    ax.set_ylim(min(ymins) - 0.01, max(ymaxs) + 0.01)
    F.pct_axis(ax)
axs[0].legend(loc="lower right")
plt.close(fig)  # Fig07_neighbourhood now drawn by step_04_descriptive_figures.py
captions.append("**Figure 7. Neighbourhood gradients.** E-bike share of trips across deciles of six "
                "characteristics of the origin census tract (ACS 2023 5-year; LEHD LODES 2022 for jobs), "
                "in 2025 (blue) and 2026 (red). Points are placed at the decile median; bands are Wilson 95% "
                "confidence intervals. All panels share the same y-axis. Job density uses a log scale.")

# =================================================================== Fig08 correlation
FEAT = {"dist_km": "Distance", "o_dist_cbd_km": "Origin–CBD", "d_dist_cbd_km": "Dest.–CBD", "hour": "Hour",
        "is_weekend": "Weekend", "is_member": "Member", "temp_c": "Temperature", "feels_like_c": "Feels-like",
        "precip_mm": "Rain", "precip_prev3h_mm": "Rain prev. 3 h", "wind_kmh": "Wind", "humidity_pct": "Humidity",
        "cloud_pct": "Cloud", "is_daylight": "Daylight", "o_pop_density_km2": "Pop. density",
        "o_job_density_km2": "Job density", "o_median_income": "Income", "o_median_age": "Median age",
        "o_pct_age_18_29": "Age 18–29", "o_pct_degree": "Degree", "o_pct_hh_no_car": "Zero-car",
        "o_pct_commute_transit": "Transit", "o_pct_commute_bicycle": "Bike commute",
        "o_pct_commute_walk": "Walk commute", "o_pct_commute_wfh": "Work from home",
        "o_pct_jobs_retail_food": "Retail/food jobs"}
cs = d[list(FEAT)].dropna().sample(n=200_000, random_state=SEED)
fig = F.new_figure(180, 170)
ax = fig.add_subplot(111)
mpl.rcParams["font.size"] = 8
P.corr_heatmap(ax, cs.values, list(FEAT.values()), method="spearman", cluster=True, triangle=True, annotate=True)
F.save(fig, FIG / "Fig08_feature_correlation")
C = cs.corr(method="spearman")
pairs = [(FEAT[a], FEAT[b], round(C.loc[a, b], 3)) for i, a in enumerate(FEAT) for b in list(FEAT)[i + 1:]
         if abs(C.loc[a, b]) >= 0.7]
pd.DataFrame(pairs, columns=["feature_1", "feature_2", "spearman_rho"]).to_csv(TAB / "Table3_high_correlation_pairs.csv", index=False)
print("High correlation pairs (|rho| >= 0.7):", pairs)
captions.append("**Figure 8. Correlation among candidate model features.** Spearman correlations on a random "
                "sample of 200,000 trips, ordered by hierarchical clustering of 1 − |ρ|. Pairs with |ρ| ≥ 0.7 are "
                "listed in Table 3 and guide feature reduction before modelling.")

cap_file = FIG / "figure_captions.md"
blocks = cap_file.read_text().split("\n\n") if cap_file.exists() else ["# Descriptive figure captions"]
blocks = [b for b in blocks if not b.startswith("**Figure 8.")] + [c for c in captions if c.startswith("**Figure 8.")]
cap_file.write_text("\n\n".join(b.strip() for b in blocks) + "\n")
print("Done.")
