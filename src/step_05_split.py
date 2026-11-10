"""Step 05: temporal split, training-only feature reduction, leakage checks.

Writes data/processed/splits/{inner_train,valid,final_train,test,excluded_pre}.parquet
and outputs/reports/05_split_manifest.json (+ split table, feature list).
Any failed leakage check stops the script with an error.

Run:  python src/step_05_split.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "study.yaml").read_text())
SEED, TARGET = CFG["seed"], CFG["target"]
OUTD = ROOT / "data" / "processed" / "splits"
REP = ROOT / "outputs" / "reports"
TAB = ROOT / "outputs" / "tables"
for p in (OUTD, REP, TAB):
    p.mkdir(parents=True, exist_ok=True)

S = CFG["split"]
fe = CFG["features"]
candidates = [f for g in ("time", "rider", "weather", "trip", "origin", "destination") for f in fe[g]]
SUPPLY = ["sup_ebike_share_7d", "sup_departures_7d"]  # extension only (step 03c)
keep_cols = ["ride_id", "started_at", "end_dock", TARGET, *candidates, *fe["robustness_only"], *SUPPLY]
# Canonical station coordinates (step 03b) prevent coordinate-precision leakage; see step_03_station_features.py
d = pd.read_parquet(ROOT / "data" / "processed" / "trips_stations.parquet", columns=keep_cols)
day = d["started_at"].dt.normalize()


def between(a, b):
    return (day >= pd.Timestamp(a)) & (day <= pd.Timestamp(b))


masks = {
    "excluded_pre": day < pd.Timestamp(S["excluded_before"]),
    "inner_train": between(*S["inner_train"]),
    "embargo": between(*S["embargo"]),
    "valid": between(*S["valid"]),
    "final_train": between(*S["final_train"]),
    "test": between(*S["test"]),
}

# ------------------------------------------------------------------ leakage checks
forbidden = set(CFG["forbidden"])
assert not (forbidden & set(candidates)), f"Forbidden columns in features: {forbidden & set(candidates)}"
assert d["ride_id"].is_unique, "Duplicated ride_id"
ids = {k: set(d.loc[m, "ride_id"]) for k, m in masks.items()}
for a, b in [("inner_train", "valid"), ("inner_train", "test"), ("valid", "test"), ("final_train", "test"),
             ("embargo", "inner_train"), ("embargo", "valid")]:
    assert not (ids[a] & ids[b]), f"Ride IDs shared between {a} and {b}"
assert d.loc[masks["inner_train"], "started_at"].max() < d.loc[masks["valid"], "started_at"].min() - pd.Timedelta(days=7)
assert d.loc[masks["final_train"], "started_at"].max() < d.loc[masks["test"], "started_at"].min()
assert masks["inner_train"].sum() + masks["embargo"].sum() + masks["valid"].sum() == masks["final_train"].sum()
print("Leakage checks passed: no forbidden features, no shared ride IDs, strict time order, 7-day embargo.")

# ------------------------------------------------------------------ correlation filter (inner train only)
cc = CFG["correlation"]
X = d.loc[masks["inner_train"] & (d["end_dock"] == 1), candidates].sample(n=cc["sample_n"], random_state=SEED)
rho = X.corr(method="spearman")
prio = {f: i for i, f in enumerate(cc["keep_priority"])}
missing = [f for f in candidates if f not in prio]
assert not missing, f"Features missing from keep_priority: {missing}"
dropped, log = set(), []
pairs = sorted(((abs(rho.loc[a, b]), a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
                if abs(rho.loc[a, b]) >= cc["threshold"]), reverse=True)
for r, a, b in pairs:
    if a in dropped or b in dropped:
        log.append((a, b, round(r, 3), "already resolved"))
        continue
    drop = a if prio[a] > prio[b] else b
    dropped.add(drop)
    log.append((a, b, round(r, 3), f"drop {drop}"))
pd.DataFrame(log, columns=["feature_1", "feature_2", "abs_spearman_inner_train", "decision"]).to_csv(
    TAB / "Table4_correlation_filter_inner_train.csv", index=False)
features = [f for f in candidates if f not in dropped]
dest = set(fe["trip"]) | set(fe["destination"])
feature_sets = {"main": features, "origin_only": [f for f in features if f not in dest],
                "main_plus_month": features + fe["robustness_only"], "main_plus_supply": features + SUPPLY}
print(f"Correlation filter dropped {len(dropped)}: {sorted(dropped)}")
print(f"Main features: {len(features)}; origin-only: {len(feature_sets['origin_only'])}")

# ------------------------------------------------------------------ save splits
rows = []
for k in ("excluded_pre", "inner_train", "embargo", "valid", "final_train", "test"):
    part = d.loc[masks[k]]
    if k != "embargo":
        part.to_parquet(OUTD / f"{k}.parquet", index=False)
    dd = part[part["end_dock"] == 1]
    rows.append({"set": k, "start": str(part["started_at"].min().date()), "end": str(part["started_at"].max().date()),
                 "trips_dock_start": len(part), "ebike_share_dock_start_pct": round(100 * part[TARGET].mean(), 2),
                 "trips_dock_to_dock": len(dd), "ebike_share_dock_to_dock_pct": round(100 * dd[TARGET].mean(), 2),
                 "member_share_dock_to_dock_pct": round(100 * dd["is_member"].mean(), 2)})
split_tab = pd.DataFrame(rows)
split_tab.to_csv(TAB / "Table5_data_split.csv", index=False)
print(split_tab.to_string(index=False))

manifest = {
    "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    "protocol": "PROTOCOL.md",
    "seed": SEED, "split": S, "split_table": rows,
    "feature_sets": feature_sets, "dropped_by_correlation_filter": sorted(dropped),
    "samples": {"main": "dock-to-dock trips (end_dock == 1)", "main_plus_month": "dock-to-dock", "main_plus_supply": "dock-to-dock",
                "origin_only": "all dock-start trips; no destination features"},
    "checks": ["no forbidden features", "spatial features from canonical station coordinates (label-free)", "unique ride_id", "no shared ride_id across sets",
               "inner_train ends > 7 days before valid starts", "final_train ends before test starts",
               "correlation filter fitted on inner_train sample only"],
}
(REP / "05_split_manifest.json").write_text(json.dumps(manifest, indent=2))
(ROOT / "configs" / "features.json").write_text(json.dumps(feature_sets, indent=2))
print("Saved splits, manifest and feature sets.")
