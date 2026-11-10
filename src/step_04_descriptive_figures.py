"""Step 04: rich descriptive figures on the MODELLING sample (dock-to-dock trips from classic docks).

Sample: data/processed/trips_stations.parquet with end_dock == 1 (both ends at a classic dock, both bike
types available), January 2025 to October 2026. January 2025 (before the 5 Feb 2025 price change) is shown but
shaded, because it is excluded from model training.

Fig01_temporal          daily trips and e-bike share with policy events; calendar of daily share; hour profile;
                        year-on-year change by month
Fig02_composition       rider -> bike flows; distance ECDF; temperature dose-response; distance x hour; contexts
Fig03_spatial           tract share 2026; change; station bubbles with Loop zoom; bivariate casual x e-bike map
Fig04_time_heatmaps     day x hour e-bike share, members / casual, 2025 / 2026 / change
Fig05_trip_profile      rainclouds of distance and speed; distance x speed share surface; share by distance, month
Fig06_weather_context   temperature x rain; temperature x hour; daily temperature scatter; wind; rider; contexts
Fig07_neighborhood     decile heat tables (2026 and change); tract scatter plots
Speed and duration are descriptive only and are never model features.

Run:  python src/step_04_descriptive_figures.py [--only fig01,fig03]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fstudio as F  # noqa: E402
import plots as P  # noqa: E402
import plots_extra as PX  # noqa: E402

F.use_theme("atlas_white")
mpl.rcParams.update({"font.weight": "bold"})
OUT = ROOT / "outputs"
FIG = OUT / "figures"
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="fig01,fig02,fig03,fig04,fig05,fig06,fig07")
ONLY = ap.parse_args().only.split(",")
SEED = 42
rng = np.random.default_rng(SEED)

RIDER = {1: ("Members", F.S4[0]), 0: ("Casual", F.S4[1])}
VEH = {1: ("E-bike", F.S4[2]), 0: ("Classic", F.S4[3])}
YEAR_C = {2025: F.MUTED, 2026: F.C6[3]}
YEAR_LS = {2025: "--", 2026: "-"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
EVENTS = [(pd.Timestamp("2025-02-05"), "1"), (pd.Timestamp("2025-07-03"), "2"), (pd.Timestamp("2026-03-16"), "3")]
captions = {}

# ------------------------------------------------------------------ data
COLS = ["ride_id", "e_bike", "started_at", "is_member", "hour", "day_of_week", "is_weekend", "month", "year",
        "temp_c", "precip_mm", "wind_kmh", "is_daylight", "end_dock", "o_tract", "o_dist_cbd_km", "dist_km",
        "round_trip", "o_median_income", "o_pct_hh_no_car", "o_pct_degree", "o_job_density_km2",
        "o_pop_density_km2", "o_pct_commute_transit"]
d = pd.read_parquet(ROOT / "data" / "processed" / "trips_stations.parquet", columns=COLS)
d = d[(d["end_dock"] == 1) & (d["started_at"] < "2026-11-01")].drop(columns="end_dock").reset_index(drop=True)
extra = pd.read_parquet(ROOT / "data" / "interim" / "divvy_trips_2025_2026.parquet",
                        columns=["ride_id", "start_station_name", "start_lat", "start_lng", "duration_min"])
d = d.merge(extra.drop_duplicates("ride_id"), on="ride_id", how="left", validate="one_to_one")
del extra
d["day"] = d["started_at"].dt.normalize()
print(f"Sample: {len(d):,} dock-to-dock trips, {d['day'].min().date()} to {d['day'].max().date()}, "
      f"e-bike share {d['e_bike'].mean():.3f}")


def share(g):
    return g["e_bike"].agg(["sum", "size"])


def events(ax, ymax_frac=0.97, labels=True):
    for t, lab in EVENTS:
        ax.axvline(t, color=F.INK2, lw=0.8, ls=":", zorder=1)
        if labels:
            ax.text(t, ymax_frac, lab, transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=6.8,
                    color="white", fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.18", fc=F.INK, ec="none"))


def shade_periods(ax):
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-05"), color=F.MUTED, alpha=0.12, lw=0, zorder=0)
    ax.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-01"), color=F.C6[3], alpha=0.06, lw=0, zorder=0)


def share_lines(ax, data, key, x_of=None, min_n=300, ms=2.6):
    g = data.groupby(["is_member", "year", key], observed=True)["e_bike"].agg(["sum", "size"])
    for m in (1, 0):
        for y in (2025, 2026):
            if (m, y) not in g.index.droplevel(2):
                continue
            s = g.loc[(m, y)]
            s = s[s["size"] >= min_n]
            x = np.arange(len(s)) if x_of is None else x_of(s.index)
            p, lo, hi = F.wilson(s["sum"].values, s["size"].values)
            ax.fill_between(x, lo, hi, color=RIDER[m][1], alpha=0.16, lw=0)
            ax.plot(x, p, color=RIDER[m][1], ls=YEAR_LS[y], marker="o", ms=ms, lw=1.5, mec=F.BG, mew=0.4)


def rider_year_legend(ax, loc="lower right", ncol=2):
    h = [Line2D([], [], color=RIDER[m][1], lw=1.8, label=RIDER[m][0]) for m in (1, 0)]
    h += [Line2D([], [], color=F.INK2, lw=1.3, ls=YEAR_LS[y], label=str(y)) for y in (2025, 2026)]
    ax.legend(handles=h, loc=loc, ncol=ncol, fontsize=7.2, columnspacing=0.9, handlelength=1.6)


def heat(ax, M, xl, yl, cmap, vmin=None, vmax=None, annot=False, fmt="{:.0f}", scale=100, fs=6.6):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, fmt.format(scale * M[i, j]), ha="center", va="center", fontsize=fs,
                            color=F.text_on(im.cmap(im.norm(M[i, j]))))
    ax.set_xticks(range(len(xl)), xl)
    ax.set_yticks(range(len(yl)), yl)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    return im


def cbar(fig, ax, im, label, fmt=None, loc="right", shrink=0.85):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, aspect=22, pad=0.02, location=loc,
                      **({"format": fmt} if fmt else {}))
    cb.set_label(label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    return cb


# =================================================================== Fig01 temporal
if "fig01" in ONLY:
    daily = d.groupby(["day", "e_bike"]).size().unstack(fill_value=0)
    fig = F.new_figure(180, 215)
    outer = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.05, 1.05])
    ax = fig.add_subplot(outer[0])
    shade_periods(ax)
    for v in (0, 1):
        r = daily[v].rolling(7, center=True, min_periods=4).mean() / 1000
        ax.fill_between(r.index, 0, r.values, color=VEH[v][1], alpha=0.18, lw=0)
        PX.glow_line(ax, r.index, r.values, VEH[v][1], lw=1.6, label=VEH[v][0])
    events(ax)
    ax.set_ylabel("Trips per day (000)")
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-11-01"))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.text(pd.Timestamp("2025-07-01"), 0.82, "Training period", transform=ax.get_xaxis_transform(), fontsize=7.3,
            color=F.INK2, ha="center")
    ax.text(pd.Timestamp("2026-06-01"), 0.82, "Test period", transform=ax.get_xaxis_transform(), fontsize=7.3,
            color=F.C6[3], ha="center")
    ax.legend(loc="upper left", fontsize=7.3, bbox_to_anchor=(0.06, 0.98))
    F.ygrid(ax)
    F.panel(fig, ax, "a", "Daily trips by bike type")

    ax = fig.add_subplot(outer[1], sharex=ax)
    shade_periods(ax)
    for m in (1, 0):
        g = d[d.is_member == m].groupby("day")["e_bike"].agg(["sum", "size"])
        s = g["sum"].rolling(7, center=True, min_periods=4).sum() / g["size"].rolling(7, center=True, min_periods=4).sum()
        ax.scatter(g.index, g["sum"] / g["size"], s=2.5, color=RIDER[m][1], alpha=0.25, lw=0, rasterized=True)
        PX.glow_line(ax, s.index, s.values, RIDER[m][1], lw=1.7, label=RIDER[m][0])
    events(ax)
    F.pct_axis(ax)
    ax.set_ylabel("E-bike share")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b\n%Y"))
    ax.set_ylim(0.3, 0.8)
    ax.legend(loc="lower right", fontsize=7.3, ncol=2)
    F.ygrid(ax)
    F.panel(fig, ax, "b", "Daily e-bike share")

    ax = fig.add_subplot(outer[2])
    ds = d.groupby("day")["e_bike"].mean()
    vmin, vmax = np.nanpercentile(ds, [2, 98])
    norm = mpl.colors.Normalize(vmin, vmax)
    for k, yr in enumerate((2025, 2026)):
        y0 = 9.6 * (1 - k)
        days = pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
        first = days[0].weekday()
        for dd in days:
            wk, wd = (dd.dayofyear - 1 + first) // 7, dd.weekday()
            v = ds.get(dd, np.nan)
            fc = F.tint(F.INK, 0.94) if np.isnan(v) else F.SEQ(norm(v))
            ax.add_patch(Rectangle((wk, y0 + 6 - wd), 0.9, 0.9, fc=fc, lw=0))
        for t, lab in EVENTS:
            if t.year == yr:
                wk = (t.dayofyear - 1 + first) // 7
                ax.add_patch(Rectangle((wk - 0.1, y0 - 0.15), 1.1, 7.2, fill=False, ec=F.INK, lw=1.1))
                ax.text(wk + 0.45, y0 + 7.75, lab, ha="center", va="center", fontsize=6.3, color="white",
                        bbox=dict(boxstyle="circle,pad=0.15", fc=F.INK, ec="none"))
        ax.text(-1.2, y0 + 3.45, str(yr), ha="right", va="center", fontsize=8.5, fontweight="bold")
        for wd_, nm in ((0, "Mon"), (3, "Thu"), (6, "Sun")):
            ax.text(54.6, y0 + 6 - wd_ + 0.45, nm, ha="left", va="center", fontsize=6.3, color=F.INK2)
    for mth in range(1, 13):
        t0 = pd.Timestamp(f"2026-{mth:02d}-01")
        ax.text((t0.dayofyear - 1 + t0.replace(month=1, day=1).weekday()) // 7, -1.0, t0.strftime("%b"),
                fontsize=6.8, color=F.INK2)
    ax.set_xlim(-6, 58)
    ax.set_ylim(-1.6, 17.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    F.panel(fig, ax, "c", "Calendar of daily e-bike share")
    cax = ax.inset_axes([0.72, 1.0, 0.22, 0.05])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=F.SEQ), cax=cax, orientation="horizontal",
                      format=mpl.ticker.PercentFormatter(1, 0))
    cb.ax.tick_params(labelsize=6.5, length=0, pad=1)
    cb.outline.set_visible(False)

    bot = outer[3].subgridspec(1, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(bot[0])
    g = d.groupby(["year", "is_weekend", "hour"])["e_bike"].mean()
    for wk, mk in ((0, "o"), (1, "s")):
        for yr in (2025, 2026):
            s = g.loc[(yr, wk)]
            ax.plot(s.index, s.values, color=YEAR_C[yr], ls="-" if wk == 0 else "--", marker=mk, ms=2.4, lw=1.4)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Start hour")
    ax.set_ylabel("E-bike share")
    F.pct_axis(ax)
    F.ygrid(ax)
    ax.legend(handles=[Line2D([], [], color=YEAR_C[y], lw=1.8, label=str(y)) for y in (2025, 2026)] +
              [Line2D([], [], color=F.INK2, lw=1.2, ls="-", marker="o", ms=3, label="Weekday"),
               Line2D([], [], color=F.INK2, lw=1.2, ls="--", marker="s", ms=3, label="Weekend")],
              fontsize=6.6, ncol=4, loc="upper center", columnspacing=0.7, handlelength=1.5)
    ax.set_ylim(0.36, 0.78)
    F.panel(fig, ax, "d", "Hour profile")

    ax = fig.add_subplot(bot[1])
    g = d.groupby(["is_member", "year", "month"])["e_bike"].mean()
    w = 0.38
    for j, m in enumerate((1, 0)):
        chg = 100 * (g.loc[(m, 2026)] - g.loc[(m, 2025)]).dropna()
        x = np.arange(len(chg)) + (j - 0.5) * w
        ax.bar(x, chg.values, width=w, color=RIDER[m][1], label=RIDER[m][0])
        for xi, v in zip(x, chg.values):
            ax.text(xi, v + (0.4 if v >= 0 else -0.5), f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.2, color=F.INK2)
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.set_xticks(range(10), list("JFMAMJJASO"))
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Change (pp)")
    ax.set_ylim(-3.5, 20)
    ax.legend(fontsize=7, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "e", "Year-on-year change")
    F.save(fig, FIG / "Fig01_temporal")
    captions["Fig01"] = (
        "**Figure 1. Temporal dynamics of e-bike choice at classic docks, January 2025 to October 2026.** "
        f"Dock-to-dock trips ({len(d):,}). (a) Daily trips by bike type (7-day centered mean). (b) Daily e-bike share "
        "(points) and its 7-day mean (line) for members and casual riders. Gray shading = January 2025, before the "
        "5 February 2025 price change and excluded from model training; teal shading = 2026 test period. Numbered "
        "markers: (1) 5 Feb 2025 price increase; (2) early July 2025 $99 membership offer and e-bike cap; (3) 16 "
        "March 2026 price change and fleet expansion (see divvy_policy_timeline.csv). (c) Calendar of daily e-bike "
        "share (week columns, weekday rows); boxed weeks contain an event. (d) E-bike share by start hour on "
        "weekdays and weekends. (e) Change in e-bike share between the same months of 2025 and 2026 (percentage "
        "points, pp).")

# =================================================================== Fig02 composition and conditions
if "fig02" in ONLY:
    fig = F.new_figure(180, 205)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    sm = d.sample(60000, random_state=SEED)
    PX_order = [["2025", "2026"], ["Member", "Casual"], ["E-bike", "Classic"]]
    P.alluvial(ax, [sm["year"].astype(str).values, np.where(sm.is_member == 1, "Member", "Casual"),
                    np.where(sm.e_bike == 1, "E-bike", "Classic")], ["Year", "Rider", "Bike"],
               order=PX_order, colors={"2025": YEAR_C[2025], "2026": YEAR_C[2026], "Member": RIDER[1][1], "Casual": RIDER[0][1], "E-bike": VEH[1][1], "Classic": VEH[0][1]})
    F.panel(fig, ax, "a", "Year → rider → bike")

    ax = fig.add_subplot(gs[0, 1])
    cap = 8
    for v in (1, 0):
        for m in (1, 0):
            x = np.sort(d.loc[(d.e_bike == v) & (d.is_member == m), "dist_km"].sample(40000, random_state=SEED).clip(upper=cap))
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=VEH[v][1], ls="-" if m == 1 else "--", lw=1.6)
            ax.plot([np.median(x)] * 2, [0, 0.5], color=VEH[v][1], lw=0.7, ls=":")
    ax.axhline(0.5, color=F.RULE, lw=0.8)
    F.pct_axis(ax)
    ax.set_xlim(0, cap)
    ax.set_xlabel("Trip distance (km)")
    ax.set_ylabel("Cumulative share of trips")
    ax.legend(handles=[Line2D([], [], color=VEH[v][1], lw=2, label=VEH[v][0]) for v in (1, 0)] +
              [Line2D([], [], color=F.INK2, lw=1.3, ls=ls, label=lab) for ls, lab in (("-", "Members"), ("--", "Casual"))],
              fontsize=6.9, loc="lower right", ncol=2, columnspacing=0.8)
    F.ygrid(ax, "both")
    F.panel(fig, ax, "b", "Distance distribution")

    ax = fig.add_subplot(gs[1, 0])
    TB = np.arange(-20, 41, 4)
    for yr in (2025, 2026):
        s = d[d.year == yr]
        g = s.groupby(pd.cut(s["temp_c"], TB), observed=True)["e_bike"].agg(["sum", "size"])
        g = g[g["size"] >= 300]
        p, lo, hi = F.wilson(g["sum"].values, g["size"].values)
        x = np.array([iv.mid for iv in g.index])
        ax.fill_between(x, lo, hi, color=YEAR_C[yr], alpha=0.2, lw=0)
        ax.plot(x, p, color=YEAR_C[yr], lw=1.6, zorder=3)
        ax.scatter(x, p, s=8 + 90 * g["size"] / g["size"].max(), color=YEAR_C[yr], ec=F.BG, lw=0.8, zorder=4,
                   label=str(yr))
    F.pct_axis(ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("E-bike share")
    ax.legend(fontsize=7.2, loc="upper right", title="Marker area ∝ trips", title_fontsize=6.6)
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Temperature response")

    ax = fig.add_subplot(gs[1, 1])
    DB = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 50]
    DL = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-6", ">6"]
    s = d[d.year == 2026]
    g = s.groupby([pd.cut(s["dist_km"], DB, include_lowest=True), "hour"], observed=False)["e_bike"].agg(["mean", "size"])
    M = np.where(g["size"].unstack().values >= 200, g["mean"].unstack().values, np.nan)
    im = heat(ax, M, [str(h) if h % 6 == 0 else "" for h in range(24)], DL, F.SEQ)
    ax.set_xlabel("Start hour")
    ax.set_ylabel("Trip distance (km)")
    cbar(fig, ax, im, "E-bike share, 2026", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "d", "Distance × hour, 2026")

    ax = fig.add_subplot(gs[2, :])
    ctx = [("Dry", d.precip_mm <= 0.1), ("Rain", d.precip_mm > 0.1), ("Below 5 °C", d.temp_c < 5),
           ("5-20 °C", d.temp_c.between(5, 20)), ("Above 20 °C", d.temp_c > 20), ("Daylight", d.is_daylight == 1),
           ("Dark", d.is_daylight == 0), ("Weekday", d.is_weekend == 0), ("Weekend", d.is_weekend == 1),
           ("One-way", d.round_trip == 0), ("Round trip", d.round_trip == 1), ("Under 1 km", d.dist_km < 1),
           ("Over 3 km", d.dist_km > 3)]
    x = np.arange(len(ctx))
    for k, (lab, msk) in enumerate(ctx):
        a = d.loc[msk & (d.year == 2025), "e_bike"].mean()
        b = d.loc[msk & (d.year == 2026), "e_bike"].mean()
        ax.plot([k, k], [a, b], color=F.tint(F.C6[3], 0.55), lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(k, a, s=40, color=YEAR_C[2025], ec=F.BG, lw=1, zorder=3)
        ax.scatter(k, b, s=48, color=YEAR_C[2026], ec=F.BG, lw=1, zorder=3)
        ax.text(k, b + 0.012, f"+{100 * (b - a):.1f}", ha="center", fontsize=6.8, color=F.INK)
    ax.set_xticks(x, [c[0] for c in ctx], rotation=30, ha="right", fontsize=7.4)
    F.pct_axis(ax)
    ax.set_ylabel("E-bike share")
    ax.set_ylim(0.3, 0.75)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=YEAR_C[y], mec=F.BG, ms=6, label=str(y)) for y in (2025, 2026)],
              fontsize=7.2, loc="upper left", ncol=2)
    F.ygrid(ax)
    F.panel(fig, ax, "e", "Every condition shifted (pp change above)")
    F.save(fig, FIG / "Fig02_composition")
    captions["Fig02"] = (
        "**Figure 2. Who rides which bike, how far and in what conditions.** (a) Flows from year to rider type to "
        "bike type (random 60,000 trips; band width ∝ trips). (b) Cumulative distribution of straight-line trip "
        "distance by bike type for members (solid) and casual riders (dashed), capped at 8 km; dotted lines mark "
        "medians. (c) E-bike share by 4 °C temperature band in 2025 and 2026 with Wilson 95% bands; marker area is "
        "proportional to trips (bands with fewer than 300 trips omitted). (d) E-bike share by distance band and "
        "start hour in 2026 (cells with fewer than 200 trips blank). (e) E-bike share in 2025 and 2026 under "
        "thirteen trip conditions; numbers give the change in percentage points. The rise is present in every "
        "condition, which anticipates the decomposition in Figure 13.")

# =================================================================== Fig03 spatial
if "fig03" in ONLY:
    tr = gpd.read_file(ROOT / "data" / "external" / "tiger" / "tracts_chicago.gpkg").to_crs(3857)
    tr = tr[tr["ALAND"] > 0.2 * (tr["ALAND"] + tr["AWATER"])]
    MIN_N = 200
    a = d.groupby(["o_tract", "year"]).agg(p=("e_bike", "mean"), n=("e_bike", "size"),
                                           cas=("is_member", lambda x: 1 - x.mean())).unstack("year")
    ok = (a[("n", 2025)] >= MIN_N) & (a[("n", 2026)] >= MIN_N)
    tv = pd.DataFrame({"p26": a.loc[ok, ("p", 2026)], "chg": 100 * (a.loc[ok, ("p", 2026)] - a.loc[ok, ("p", 2025)]),
                       "cas26": a.loc[ok, ("cas", 2026)]})
    tv.index = tv.index.astype(str)
    g = tr.merge(tv, left_on="GEOID", right_index=True, how="left")
    has = g[g["p26"].notna()]
    xmin, ymin, xmax, ymax = has.total_bounds
    pad = 0.03 * (xmax - xmin)
    base_col = F.tint(F.MUTED, 0.78)

    st = d[d.year == 2026].groupby("start_station_name").agg(n=("e_bike", "size"), p=("e_bike", "mean"),
                                                              lat=("start_lat", "median"), lng=("start_lng", "median"))
    st = st[st["n"] >= 100]
    stg = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lng"], st["lat"]), crs=4326).to_crs(3857)

    def frame(ax):
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_axis_off()
        ax.set_aspect("equal")

    def scalebar(ax, km=5):
        L = km * 1000 / np.cos(np.radians(41.88))
        x0, y0 = xmin + 0.02 * (xmax - xmin), ymin + 0.015 * (ymax - ymin)
        ax.add_patch(Rectangle((x0, y0), L / 2, 300, fc=F.INK, ec=F.INK, lw=0.6))
        ax.add_patch(Rectangle((x0 + L / 2, y0), L / 2, 300, fc="white", ec=F.INK, lw=0.6))
        ax.text(x0 + L / 2, y0 + 900, f"{km} km", ha="center", fontsize=6.6, color=F.INK)
        ax.annotate("N", xy=(x0 + 0.1 * L, y0 + 0.12 * (ymax - ymin)), xytext=(x0 + 0.1 * L, y0 + 0.06 * (ymax - ymin)),
                    ha="center", fontsize=7.5, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=1))

    def inset_cbar(ax, cmap, norm, label, fmt=None, pos=(0.03, 0.30, 0.04, 0.28)):
        cax = ax.inset_axes(list(pos))
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, **({"format": fmt} if fmt else {}))
        cb.set_label(label, fontsize=6.8, labelpad=2)
        cb.ax.tick_params(labelsize=6.3, length=0, pad=1)
        cb.outline.set_visible(False)

    st25 = d[d.year == 2025].groupby("start_station_name").agg(n25=("e_bike", "size"), p25=("e_bike", "mean"))
    stc = st.join(st25, how="inner")
    stc = stc[(stc["n"] >= 1500) & (stc["n25"] >= 1500)].assign(chg=lambda x: 100 * (x["p"] - x["p25"]))

    def hcbar(ax, cmap, norm, label, fmt=None, pos=(0.05, 0.07, 0.5, 0.028)):
        cax = ax.inset_axes(list(pos))
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          **({"format": fmt} if fmt else {}))
        cb.set_label(label, fontsize=6.6, labelpad=1)
        cb.ax.tick_params(labelsize=6.2, length=0, pad=1)
        cb.outline.set_visible(False)

    fig = F.new_figure(180, 218)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1])
    top = outer[0].subgridspec(1, 3)
    bot = outer[1].subgridspec(1, 3, width_ratios=[0.62, 1.2, 1.1])

    ax = fig.add_subplot(top[0])
    tr.plot(ax=ax, color=base_col, edgecolor="white", lw=0.08)
    nrm = mpl.colors.Normalize(*np.nanpercentile(has["p26"], [2, 98]))
    has.plot(ax=ax, column="p26", cmap=F.SEQ, norm=nrm, edgecolor="white", lw=0.1)
    frame(ax)
    hcbar(ax, F.SEQ, nrm, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "a", "E-bike share, 2026")

    ax = fig.add_subplot(top[1])
    tr.plot(ax=ax, color=base_col, edgecolor="white", lw=0.08)
    lim = np.nanpercentile(np.abs(has["chg"]), 97)
    nrm = mpl.colors.TwoSlopeNorm(0, -lim, lim)
    has.plot(ax=ax, column="chg", cmap=F.DIV, norm=nrm, edgecolor="white", lw=0.1)
    frame(ax)
    hcbar(ax, F.DIV, nrm, "Change (pp)")
    ax.text(0.97, 0.99, f"{(has['chg'] > 0).mean():.0%} of\ntracts rose", transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color=F.INK, bbox=dict(fc="white", ec=F.RULE, boxstyle="round,pad=0.3", lw=0.6))
    F.panel(fig, ax, "b", "Change, 2025-26")

    ax = fig.add_subplot(top[2])
    c00, c10, c01, c11 = map(np.array, map(to_rgb, ["#F2EFEA", "#1F6F8B", "#E4572E", "#4B1D52"]))

    def bicol(i, j):
        u, v = i / 2, j / 2
        return to_hex(np.clip((1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11, 0, 1))

    qx = pd.qcut(has["cas26"], 3, labels=False)
    qy = pd.qcut(has["p26"], 3, labels=False)
    hb = has.assign(bc=[bicol(i, j) for i, j in zip(qx, qy)])
    tr.plot(ax=ax, color=base_col, edgecolor="white", lw=0.08)
    hb.plot(ax=ax, color=hb["bc"], edgecolor="white", lw=0.1)
    frame(ax)
    lg = ax.inset_axes([0.07, 0.08, 0.2, 0.2])
    for i in range(3):
        for j in range(3):
            lg.add_patch(Rectangle((i, j), 1, 1, fc=bicol(i, j), ec="white", lw=0.8))
    lg.set_xlim(0, 3)
    lg.set_ylim(0, 3)
    lg.set_aspect("equal")
    lg.set_axis_off()
    lg.text(1.5, -0.35, "Casual →", ha="center", va="top", fontsize=6.4)
    lg.text(-0.35, 1.5, "E-bike →", ha="right", va="center", rotation=90, fontsize=6.4)
    F.panel(fig, ax, "c", "Casual × e-bike")

    ax = fig.add_subplot(bot[0])
    tr.plot(ax=ax, color=F.tint(F.MUTED, 0.88), edgecolor="white", lw=0.08)
    nrm_s = mpl.colors.Normalize(*np.nanpercentile(st["p"], [3, 97]))
    order = stg.sort_values("n", ascending=False)
    smax = order["n"].max()
    ax.scatter(order.geometry.x, order.geometry.y, s=0.8 + 22 * (order["n"] / smax) ** 0.8, c=order["p"], cmap=F.SEQ,
               norm=nrm_s, ec=F.INK, lw=0.1, alpha=0.9, zorder=3)
    frame(ax)
    scalebar(ax)
    loop = gpd.GeoSeries(gpd.points_from_xy([-87.632], [41.885]), crs=4326).to_crs(3857).iloc[0]
    half = 3500 / np.cos(np.radians(41.88))
    ax.add_patch(Rectangle((loop.x - half, loop.y - half), 2 * half, 2 * half, fill=False, ec=F.INK, lw=1.0, zorder=5))
    F.panel(fig, ax, "d", "Stations")

    ax = fig.add_subplot(bot[1])
    tr.plot(ax=ax, color=F.tint(F.MUTED, 0.9), edgecolor="white", lw=0.5)
    zz = order[(abs(order.geometry.x - loop.x) < half) & (abs(order.geometry.y - loop.y) < half)]
    ax.scatter(zz.geometry.x, zz.geometry.y, s=6 + 230 * (zz["n"] / smax) ** 0.8, c=zz["p"], cmap=F.SEQ, norm=nrm_s,
               ec=F.INK, lw=0.35, alpha=0.92, zorder=3)
    ax.set_xlim(loop.x - half, loop.x + half)
    ax.set_ylim(loop.y - half, loop.y + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s_ in ax.spines.values():
        s_.set_visible(True)
        s_.set_color(F.INK)
        s_.set_linewidth(1.0)
    hcbar(ax, F.SEQ, nrm_s, "Station e-bike share, 2026", mpl.ticker.PercentFormatter(1, 0),
          pos=(0.0, -0.075, 0.42, 0.03))
    for nn in (1000, 5000, 15000):
        ax.scatter([], [], s=6 + 230 * (nn / smax) ** 0.8, color=F.MUTED, ec=F.INK, lw=0.3, label=f"{nn:,}")
    ax.legend(title="Trips, 2026", fontsize=6.4, title_fontsize=6.5, loc="upper left", bbox_to_anchor=(0.5, -0.005),
              ncol=3, frameon=False, columnspacing=0.6, handletextpad=0.2)
    F.panel(fig, ax, "e", "Loop and center (7 km)")

    ax = fig.add_subplot(bot[2])
    pick = pd.concat([stc.nlargest(8, "chg"), stc.nsmallest(5, "chg")]).sort_values("chg")
    yy = np.arange(len(pick))
    for yi, (nm, r) in zip(yy, pick.iterrows()):
        col = F.S4[1] if r["chg"] > 0 else F.S4[0]
        ax.plot([r["p25"], r["p"]], [yi, yi], color=F.tint(col, 0.55), lw=3.2, solid_capstyle="round", zorder=1)
        ax.scatter(r["p25"], yi, s=22, color=F.BG, ec=YEAR_C[2025], lw=1.3, zorder=3)
        ax.scatter(r["p"], yi, s=26, color=col, ec=F.BG, lw=0.8, zorder=3)
    names = [(n[:24] + "…") if len(n) > 25 else n for n in pick.index]
    ax.set_yticks(yy, names, fontsize=6.4)
    F.pct_axis(ax, "x")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    ax.set_xlabel("E-bike share (open = 2025, filled = 2026)", fontsize=8)
    F.panel(fig, ax, "f", "Largest station changes")
    F.save(fig, FIG / "Fig03_spatial")
    captions["Fig03"] = (
        f"**Figure 3. Spatial patterns of e-bike choice.** Origin census tracts with at least {MIN_N} dock-to-dock "
        "trips in both years are coloured; other land tracts are grey (open-water tracts removed). (a) E-bike share "
        "in 2026. (b) Change from 2025 to 2026 in percentage points (pp). (c) Bivariate map of the 2026 casual-rider "
        "share (blue axis) and e-bike share (red axis), each in tract terciles; dark purple = high on both. (d) "
        "Stations with at least 100 trips in 2026 (marker area ∝ trips, colour = e-bike share); the square marks the "
        "area enlarged in (e), a 7 km window around the Loop. (f) Stations with at least 1,500 trips in each year "
        "and the largest rises and falls in e-bike share. Projection: Web Mercator; the scale bar is corrected for "
        "latitude.")

# =================================================================== Fig04 time heat maps
if "fig04" in ONLY:
    fig = F.new_figure(180, 118)
    gs = fig.add_gridspec(2, 3)
    hm = d.groupby(["is_member", "year", "day_of_week", "hour"])["e_bike"].mean()
    vmin, vmax = np.nanpercentile(hm.values, [2, 98])
    mats = {(m, y): hm.loc[(m, y)].unstack() for m in (1, 0) for y in (2025, 2026)}
    lim = np.nanpercentile(np.abs(np.concatenate([(mats[(m, 2026)] - mats[(m, 2025)]).values.ravel() for m in (1, 0)])), 98)
    letters = iter("abcdef")
    sh, ch = [], []
    for r, m in enumerate((1, 0)):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            if c < 2:
                im1 = ax.imshow(mats[(m, 2025 + c)].values, aspect="auto", cmap=F.SEQ, vmin=vmin, vmax=vmax)
                sh.append(ax)
                title = f"{RIDER[m][0]}, {2025 + c}"
            else:
                im2 = ax.imshow(100 * (mats[(m, 2026)] - mats[(m, 2025)]).values, aspect="auto", cmap=F.DIV,
                                vmin=-100 * lim, vmax=100 * lim)
                ch.append(ax)
                title = f"{RIDER[m][0]}, change"
            ax.set_yticks(range(7), DAYS if c == 0 else [""] * 7)
            ax.set_xticks(range(0, 24, 6))
            ax.tick_params(length=0)
            ax.grid(False)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if r == 1:
                ax.set_xlabel("Start hour")
            F.panel(fig, ax, next(letters), title)
    fig.colorbar(im1, ax=sh, location="right", shrink=0.75, aspect=30, pad=0.01, label="E-bike share",
                 format=mpl.ticker.PercentFormatter(1, 0)).outline.set_visible(False)
    fig.colorbar(im2, ax=ch, location="right", shrink=0.75, aspect=30, pad=0.02,
                 label="Change, 2025 to 2026 (pp)").outline.set_visible(False)
    F.save(fig, FIG / "Fig04_time_heatmaps")
    captions["Fig04"] = (
        "**Figure 4. When riders choose e-bikes.** E-bike share of dock-to-dock trips by day of week and start hour "
        "for members (a-c) and casual riders (d-f): 2025, 2026 and the change in percentage points (pp). Share panels "
        "use a common colour scale.")

# =================================================================== Fig05 trip profile
if "fig05" in ONLY:
    s5 = d[(d.round_trip == 0) & d.duration_min.between(1, 180)].copy()
    s5["speed"] = s5["dist_km"] / (s5["duration_min"] / 60)
    s5 = s5[s5["speed"] < 40]
    fig = F.new_figure(180, 210)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])
    grp = [(1, 1), (1, 0), (0, 1), (0, 0)]
    labs = ["E-bike · member", "E-bike · casual", "Classic · member", "Classic · casual"]
    cols = [VEH[1][1], F.tint(VEH[1][1], 0.35), VEH[0][1], F.tint(VEH[0][1], 0.35)]
    smp = {k: s5[(s5.e_bike == k[0]) & (s5.is_member == k[1])].sample(2500, random_state=SEED) for k in grp}
    for j, (var, cap, unit, lab, ttl) in enumerate((("dist_km", 8, "km", "Straight-line distance (km)", "Trip distance"),
                                                     ("speed", 25, "km/h", "Straight-line speed (km/h)", "Trip speed"))):
        ax = fig.add_subplot(gs[0, j])
        P.raincloud(ax, [smp[k][var].clip(upper=cap).values for k in grp], labs, colors=cols, unit=unit)
        ax.set_xlabel(lab)
        F.panel(fig, ax, "ab"[j], ttl)

    ax = fig.add_subplot(gs[1, 0])
    DB = np.arange(0, 6.01, 0.5)
    SB = np.arange(4, 26.1, 2)
    g = s5.groupby([pd.cut(s5["speed"], SB), pd.cut(s5["dist_km"], DB)], observed=False)["e_bike"].agg(["mean", "size"])
    M = np.where(g["size"].unstack().values >= 150, g["mean"].unstack().values, np.nan)
    im = heat(ax, M, [f"{v:g}" if i % 2 == 0 else "" for i, v in enumerate(DB[:-1])],
              [f"{v:g}" if i % 2 == 0 else "" for i, v in enumerate(SB[:-1])], F.SEQ, vmin=0, vmax=1)
    ax.set_xlabel("Distance (km, bin start)")
    ax.set_ylabel("Speed (km/h, bin start)")
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "c", "Speed separates bikes")

    ax = fig.add_subplot(gs[1, 1])
    for k, lab_, col in zip(grp, labs, cols):
        x = np.sort(np.log10(smp[k]["duration_min"].values))
        xs = np.linspace(0, np.log10(120), 200)
        from scipy.stats import gaussian_kde
        y = gaussian_kde(x, 0.18)(xs)
        ax.fill_between(10 ** xs, 0, y, color=col, alpha=0.18, lw=0)
        ax.plot(10 ** xs, y, color=col, lw=1.5, ls="-" if k[1] == 1 else "--", label=lab_)
    ax.set_xscale("log")
    F.plain_log(ax, [2, 5, 10, 20, 60])
    ax.set_xlabel("Trip duration (min, log scale)")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend(fontsize=6.8, loc="upper right")
    F.panel(fig, ax, "d", "Trip duration")

    ax = fig.add_subplot(gs[2, 0])
    DBn = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
    d["dist_bin"] = pd.cut(d["dist_km"], DBn)
    share_lines(ax, d, "dist_bin", x_of=lambda idx: np.array([iv.mid for iv in idx]))
    ax.set_xlabel("Trip distance (km, bin midpoint)")
    ax.set_ylabel("E-bike share")
    F.pct_axis(ax)
    F.ygrid(ax)
    rider_year_legend(ax)
    F.panel(fig, ax, "e", "Share by distance")

    ax = fig.add_subplot(gs[2, 1])
    share_lines(ax, d, "month", x_of=lambda idx: np.asarray(idx))
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_xlabel("Calendar month")
    F.pct_axis(ax)
    F.ygrid(ax)
    F.panel(fig, ax, "f", "Share by month")
    F.save(fig, FIG / "Fig05_trip_profile")
    captions["Fig05"] = (
        "**Figure 5. Trip profile.** One-way dock-to-dock trips of 1-180 min and below 40 km/h. (a, b) Rainclouds of "
        "straight-line distance and speed for 2,500 random trips per bike × rider group (capped at 8 km and "
        "25 km/h; box = interquartile range). (c) E-bike share by distance × speed bin (cells with fewer than 150 "
        "trips blank): at a given distance, faster trips are mostly e-bike trips. (d) Duration density by group (log "
        "scale). (e, f) E-bike share by distance and calendar month for members and casual riders in 2025 (dashed) "
        "and 2026 (solid); bands are Wilson 95% intervals. Speed and duration are descriptive only and are never "
        "model features, because they are only known after the bike is chosen.")

# =================================================================== Fig06 weather and context
if "fig06" in ONLY:
    fig = F.new_figure(180, 210)
    gs = fig.add_gridspec(3, 2)
    ax = fig.add_subplot(gs[0, 0])
    TB = [-25, -5, 0, 5, 10, 15, 20, 25, 30, 40]
    TL = ["<−5", "−5–0", "0–5", "5–10", "10–15", "15–20", "20–25", "25–30", ">30"]
    RB = [-0.01, 0.0, 0.5, 2, 100]
    g = d.groupby([pd.cut(d["temp_c"], TB), pd.cut(d["precip_mm"], RB)], observed=False)["e_bike"].agg(["mean", "size"])
    M = np.where(g["size"].unstack().values >= 300, g["mean"].unstack().values, np.nan)
    im = heat(ax, M, ["Dry", "<0.5", "0.5–2", ">2"], TL, F.SEQ, annot=True)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Temperature (°C)")
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "a", "Temperature × rain")

    ax = fig.add_subplot(gs[0, 1])
    g = d.groupby([pd.cut(d["temp_c"], TB), "hour"], observed=False)["e_bike"].agg(["mean", "size"])
    M = np.where(g["size"].unstack().values >= 200, g["mean"].unstack().values, np.nan)
    im = heat(ax, M, [str(h) if h % 6 == 0 else "" for h in range(24)], TL, F.SEQ)
    ax.set_xlabel("Start hour")
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "b", "Temperature × hour")

    ax = fig.add_subplot(gs[1, 0])
    dd = d.groupby("day").agg(t=("temp_c", "mean"), p=("e_bike", "mean"), n=("e_bike", "size"), yr=("year", "first"))
    dd = dd[dd["n"] >= 200]
    for yr in (2025, 2026):
        s = dd[dd.yr == yr]
        ax.scatter(s["t"], s["p"], s=4 + 26 * s["n"] / dd["n"].max(), color=YEAR_C[yr], alpha=0.45, lw=0, label=str(yr))
        b = s.groupby(pd.cut(s["t"], np.arange(-25, 36, 3)), observed=True).agg(t=("t", "mean"), p=("p", "mean"),
                                                                                 c=("p", "size"))
        b = b[b["c"] >= 4]
        PX.glow_line(ax, b["t"], b["p"], YEAR_C[yr], lw=1.8)
    F.pct_axis(ax)
    ax.set_xlabel("Daily mean temperature (°C)")
    ax.set_ylabel("Daily e-bike share")
    ax.set_ylim(0.3, 0.72)
    ax.legend(fontsize=7.2, loc="lower right", markerscale=2)
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Days: same weather, more e-bikes")

    ax = fig.add_subplot(gs[1, 1])
    d["wind_bin"] = pd.cut(d["wind_kmh"], [0, 5, 10, 15, 20, 25, 30, 40, 60])
    share_lines(ax, d, "wind_bin", x_of=lambda idx: np.array([iv.mid for iv in idx]))
    ax.set_xlabel("Wind speed (km/h, bin midpoint)")
    ax.set_ylabel("E-bike share")
    F.pct_axis(ax)
    F.ygrid(ax)
    rider_year_legend(ax, loc="upper left")
    F.panel(fig, ax, "d", "Wind")

    ax = fig.add_subplot(gs[2, 0])
    d["temp_bin"] = pd.cut(d["temp_c"], np.arange(-20, 41, 5))
    share_lines(ax, d, "temp_bin", x_of=lambda idx: np.array([iv.mid for iv in idx]))
    ax.set_xlabel("Temperature (°C, bin midpoint)")
    ax.set_ylabel("E-bike share")
    F.pct_axis(ax)
    F.ygrid(ax)
    F.panel(fig, ax, "e", "Temperature by rider")

    ax = fig.add_subplot(gs[2, 1])
    ctx = [("Dry", d.precip_mm <= 0.1), ("Raining", d.precip_mm > 0.1), ("Daylight", d.is_daylight == 1),
           ("Dark", d.is_daylight == 0), ("Weekday", d.is_weekend == 0), ("Weekend", d.is_weekend == 1),
           ("One-way", d.round_trip == 0), ("Round trip", d.round_trip == 1)]
    n_ctx = len(ctx)
    allv = []
    for row, (lab_, msk) in enumerate(ctx):
        y = n_ctx - 1 - row
        vals = {(m, yr): d.loc[msk & (d.is_member == m) & (d.year == yr), "e_bike"].mean() for m in (1, 0) for yr in (2025, 2026)}
        for m, off in ((1, 0.15), (0, -0.15)):
            ax.annotate("", xy=(vals[(m, 2026)], y + off), xytext=(vals[(m, 2025)], y + off),
                        arrowprops=dict(arrowstyle="-|>", color=RIDER[m][1], lw=1.6, mutation_scale=8))
            ax.scatter(vals[(m, 2025)], y + off, s=16, color=F.BG, ec=RIDER[m][1], lw=1.2, zorder=3)
        allv += list(vals.values())
    ax.set_yticks(range(n_ctx), [c[0] for c in ctx][::-1])
    F.clean_left(ax)
    F.pct_axis(ax, "x")
    F.ygrid(ax, "x")
    ax.legend(handles=[Line2D([], [], color=RIDER[m][1], lw=1.8, label=RIDER[m][0]) for m in (1, 0)],
              fontsize=7, loc="lower right")
    ax.set_xlim(min(allv) - 0.02, max(allv) + 0.02)
    ax.set_ylim(-0.6, n_ctx - 0.4)
    ax.set_xlabel("E-bike share (open = 2025, arrow tip = 2026)")
    F.panel(fig, ax, "f", "Context by rider")
    F.save(fig, FIG / "Fig06_weather_context")
    captions["Fig06"] = (
        "**Figure 6. Weather and trip context.** (a) E-bike share (%) by temperature and precipitation; (b) by "
        "temperature and start hour (cells with fewer than 300 or 200 trips blank). (c) Each point is a day (area "
        "∝ trips; days with at least 200 trips); lines are means of 3 °C bands: at the same temperature, 2026 days "
        "have higher e-bike shares than 2025 days. (d, e) E-bike share by wind speed and temperature for members "
        "and casual riders in 2025 (dashed) and 2026 (solid) with Wilson 95% bands. (f) E-bike share by trip "
        "context, from 2025 (open circle) to 2026 (arrow tip), for members and casual riders.")

# =================================================================== Fig07 neighbourhood
if "fig07" in ONLY:
    NB = [("o_median_income", "Income"), ("o_pct_degree", "Degree holders"), ("o_pct_hh_no_car", "Car-free households"),
          ("o_pct_commute_transit", "Transit commuters"), ("o_job_density_km2", "Job density"),
          ("o_pop_density_km2", "Population density"), ("o_dist_cbd_km", "Distance to Loop")]
    S, C = [], []
    for var, _ in NB:
        sub = d[["e_bike", "year", var]].dropna()
        q = pd.qcut(sub[var].rank(method="first"), 10, labels=False)
        g = sub.groupby(["year", q])["e_bike"].mean()
        S.append(g.loc[2026].values)
        C.append(100 * (g.loc[2026].values - g.loc[2025].values))
    S, C = np.array(S), np.array(C)
    fig = F.new_figure(180, 190)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    for j, (M, cmap, lab_, fmt, sc) in enumerate(((S, F.SEQ, "E-bike share, 2026", "{:.0f}", 100),
                                                  (C, F.DIV, "Change (pp)", "{:+.0f}", 1))):
        ax = fig.add_subplot(gs[0, j])
        kw = {} if j == 0 else dict(vmin=-np.abs(C).max(), vmax=np.abs(C).max())
        im = heat(ax, M[::-1], [str(i) for i in range(1, 11)], [n for _, n in NB][::-1], cmap, annot=True, fmt=fmt,
                  scale=sc, fs=6.2, **kw)
        ax.set_xlabel("Origin-tract decile (1 = lowest)")
        if j == 1:
            ax.set_yticklabels([])
        cbar(fig, ax, im, lab_, mpl.ticker.PercentFormatter(1, 0) if j == 0 else None, loc="bottom", shrink=0.7)
        F.panel(fig, ax, "ab"[j], ["Share by decile, 2026", "Change by decile"][j])

    a = d.groupby(["o_tract", "year"]).agg(p=("e_bike", "mean"), n=("e_bike", "size")).unstack("year")
    tt = d.groupby("o_tract").agg(inc=("o_median_income", "first"), cbd=("o_dist_cbd_km", "first"),
                                  nocar=("o_pct_hh_no_car", "first"))
    ok = (a[("n", 2025)] >= 200) & (a[("n", 2026)] >= 200)
    t = tt[ok].assign(p25=a.loc[ok, ("p", 2025)], p26=a.loc[ok, ("p", 2026)], n=a.loc[ok, ("n", 2026)])
    ax = fig.add_subplot(gs[1, 0])
    nrm = mpl.colors.Normalize(0, np.nanpercentile(t["cbd"], 95))
    sc_ = ax.scatter(t["inc"] / 1000, t["p26"], s=6 + 80 * (t["n"] / t["n"].max()) ** 0.7, c=t["cbd"], cmap=F.HEAT.reversed(),
                     norm=nrm, ec=F.INK, lw=0.25, alpha=0.85)
    tt_ = t.dropna(subset=["inc"])
    b = tt_.groupby(pd.qcut(tt_["inc"], 8), observed=True).apply(lambda x: pd.Series({"x": x["inc"].median() / 1000,
                                                                                      "y": np.average(x["p26"], weights=x["n"])}))
    PX.glow_line(ax, b["x"], b["y"], F.S4[1], lw=1.8)
    F.pct_axis(ax)
    ax.set_xlabel("Tract median income (US$ 000)")
    ax.set_ylabel("Tract e-bike share, 2026")
    cbar(fig, ax, sc_, "Distance to Loop (km)", shrink=0.7)
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Income and centrality")

    ax = fig.add_subplot(gs[1, 1])
    lo_, hi_ = np.nanpercentile(np.r_[t["p25"], t["p26"]], [0.3, 99.9])
    lo_, hi_ = lo_ - 0.02, hi_ + 0.02
    ax.plot([lo_, hi_], [lo_, hi_], color=F.INK2, lw=0.9, ls="--")
    ax.fill_between([lo_, hi_], [lo_, hi_], [hi_, hi_], color=F.C6[3], alpha=0.07, lw=0)
    nrm = mpl.colors.Normalize(*np.nanpercentile(t["nocar"], [5, 95]))
    sc_ = ax.scatter(t["p25"], t["p26"], s=6 + 80 * (t["n"] / t["n"].max()) ** 0.7, c=t["nocar"], cmap=F.SEQ, norm=nrm,
                     ec=F.INK, lw=0.25, alpha=0.85)
    ax.text(0.04, 0.95, f"{(t['p26'] > t['p25']).mean():.0%} of tracts above the line", transform=ax.transAxes,
            fontsize=7.2, va="top", color=F.INK)
    F.pct_axis(ax)
    F.pct_axis(ax, "x")
    ax.set_xlim(lo_, hi_)
    ax.set_ylim(lo_, hi_)
    ax.set_xlabel("Tract e-bike share, 2025")
    ax.set_ylabel("Tract e-bike share, 2026")
    cbar(fig, ax, sc_, "Car-free households (%)", shrink=0.7)
    F.ygrid(ax, "both")
    F.panel(fig, ax, "d", "Tracts, 2025 vs 2026")
    F.save(fig, FIG / "Fig07_neighborhood")
    captions["Fig07"] = (
        "**Figure 7. Neighborhood gradients.** (a) E-bike share in 2026 (%) by decile of seven origin-tract "
        "characteristics (ACS 2023 5-year; LEHD LODES 2022 for jobs); (b) change from 2025 to 2026 in percentage "
        "points. (c) Tracts with at least 200 trips in both years: median income against 2026 e-bike share; marker "
        "area ∝ 2026 trips, colour = distance to the Loop; the line joins trip-weighted means of income octiles. "
        "(d) Tract e-bike share in 2025 against 2026 (dashed line = no change; shaded = increase); colour = share of "
        "car-free households.")

# ------------------------------------------------------------------ captions
cap = FIG / "figure_captions.md"
old = {}
if cap.exists():
    for block in cap.read_text().split("\n\n"):
        if block.startswith("**Figure"):
            old[f"Fig{block.split('Figure ')[1].split('.')[0].zfill(2)}"] = block.strip()
old.update(captions)
cap.write_text("# Descriptive figure captions (Chicago, dock-to-dock modelling sample)\n\n" +
               "\n\n".join(old[k] for k in sorted(old)) + "\n")
print("done:", ", ".join(captions))
