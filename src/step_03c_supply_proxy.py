"""Step 03c: station supply proxy from STRICTLY EARLIER days.

For each trip starting at station s on day t:
  sup_ebike_share_7d = e-bike share of all departures from s on days t-7 ... t-1
  sup_departures_7d  = number of departures from s on days t-7 ... t-1 (station busyness)
Departures from the same day (t) and later are never used, so the feature only contains information
that existed before the trip. It is NaN when the station had fewer than 10 departures in the window.

Stations are keyed by normalised NAME (station IDs changed between 2025 and 2026).
History pool: every trip that started at a classic dock (any end), January 2025 to October 2026.
This proxy is used ONLY in the extension model "main_plus_supply"; the main model excludes it.

Output: adds the two columns to data/processed/trips_stations.parquet

Run:  python src/step_03c_supply_proxy.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRO, INT = ROOT / "data" / "processed", ROOT / "data" / "interim"
WINDOW, MIN_N = 7, 10

trips = pd.read_parquet(PRO / "trips_stations.parquet")
trips = trips.drop(columns=[c for c in ("sup_ebike_share_7d", "sup_departures_7d") if c in trips])
names = pd.read_parquet(INT / "divvy_trips_2025_2026.parquet", columns=["ride_id", "start_station_name"]
                        ).drop_duplicates("ride_id")
norm = lambda x: x.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
trips = trips.merge(names, on="ride_id", how="left")
trips["skey"] = norm(trips["start_station_name"])
trips["day"] = trips["started_at"].dt.normalize()

# daily departures and e-bike departures per station
daily = trips.groupby(["skey", "day"])["e_bike"].agg(n="size", k="sum").reset_index()
all_days = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
full = (daily.set_index(["skey", "day"])
        .reindex(pd.MultiIndex.from_product([daily["skey"].unique(), all_days], names=["skey", "day"]), fill_value=0)
        .reset_index().sort_values(["skey", "day"]))
g = full.groupby("skey", sort=False)
# rolling sum over the previous WINDOW days, then shift by one day so day t is excluded
full["n7"] = g["n"].transform(lambda s: s.rolling(WINDOW, min_periods=1).sum().shift(1))
full["k7"] = g["k"].transform(lambda s: s.rolling(WINDOW, min_periods=1).sum().shift(1))
full["sup_ebike_share_7d"] = np.where(full["n7"] >= MIN_N, full["k7"] / full["n7"].where(full["n7"] > 0), np.nan)
full["sup_departures_7d"] = full["n7"]

# ---- self-check: recompute for random station-days by brute force using only earlier days
rng = np.random.default_rng(42)
chk = full[full["n7"] >= MIN_N].sample(200, random_state=42)
for _, r in chk.iterrows():
    win = daily[(daily["skey"] == r["skey"]) & (daily["day"] < r["day"]) &
                (daily["day"] >= r["day"] - pd.Timedelta(days=WINDOW))]
    assert win["n"].sum() == r["n7"] and abs(win["k"].sum() / win["n"].sum() - r["sup_ebike_share_7d"]) < 1e-9
print("Self-check passed: proxy uses only the 7 days before the trip day (200 random checks).")

trips = trips.merge(full[["skey", "day", "sup_ebike_share_7d", "sup_departures_7d"]], on=["skey", "day"], how="left")
trips = trips.drop(columns=["start_station_name", "skey", "day"])
trips.to_parquet(PRO / "trips_stations.parquet", index=False)
print(f"Saved. Missing proxy: {trips['sup_ebike_share_7d'].isna().mean():.2%} of trips")
print("Mean proxy by bike type:", trips.groupby("e_bike")["sup_ebike_share_7d"].mean().round(3).to_dict())
