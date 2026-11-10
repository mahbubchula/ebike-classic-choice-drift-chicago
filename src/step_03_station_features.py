"""Step 03b: rebuild spatial features from canonical STATION coordinates.

Why (measurement design of mixed fleets):
  Divvy records classic bikes at fixed station coordinates, but e-bikes with their own GPS fix, and
  e-bikes parked away from a dock get end coordinates rounded to 2 decimals (~1 km grid).
  100% of classic trips end at a dock vs. 74.4% of e-bike trips, so any feature built from raw end
  coordinates (distance, destination tract, round trip) can reveal the label through measurement
  precision rather than behaviour.

Design:
  * Every station gets one canonical location: the median of ALL recorded coordinates at that
    station (both bike types pooled; the label is not used).
  * Start and end locations are replaced by the canonical location of the start and end station.
  * round_trip = same start and end station (exact, label-neutral).
  * end_dock flag kept so the main sample can be restricted to dock-to-dock trips; the origin-only
    sensitivity model uses all dock-start trips and no destination information.

Choice set:
  Divvy "lightweight" stations (public racks, e-bike corrals) only offer e-bikes: 100% of their starts
  are e-bikes. Station type is taken from Divvy's official GBFS v1.1 station_information feed
  (snapshot 2026-11-09, field station_type = classic | lightweight), matched by normalised station
  NAME because station IDs changed between 2025 and 2026. Names containing "public rack" or "corral"
  are lightweight. Unmatched names (renamed or removed stations, about 3% of trips) are excluded.
  This rule never uses the bike type of a trip.
    start_classic = trip starts at a classic dock   (sample for the origin-only model)
    end_dock      = trip ends at a classic dock     (main sample = start_classic & end_dock)

Output: data/processed/trips_stations.parquet

Run:  python src/step_03_station_features.py
"""
from pathlib import Path

import json

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT, INT, PRO = ROOT / "data" / "external", ROOT / "data" / "interim", ROOT / "data" / "processed"
CBD = (41.8819, -87.6278)
TRACT_VARS = ["pop_density_km2", "job_density_km2", "median_income", "median_age", "pct_age_18_29",
              "pct_degree", "pct_hh_no_car", "pct_commute_transit", "pct_commute_bicycle",
              "pct_commute_walk", "pct_commute_wfh", "pct_jobs_retail_food"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- canonical station locations (label-free)
raw = pd.read_parquet(INT / "divvy_trips_2025_2026.parquet",
                      columns=["ride_id", "start_station_id", "end_station_id", "start_station_name",
                               "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng"])
raw = raw.drop_duplicates("ride_id")

# ---------------------------------------------------------------- station type from GBFS (label-free)
norm = lambda x: x.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
gb = pd.DataFrame(json.loads((EXT / "gbfs" / "station_information_v11_2026-11-09.json").read_text())["data"]["stations"])
gb["key"] = norm(gb["name"])
gtype = gb.groupby("key")["station_type"].agg(lambda x: "classic" if (x == "classic").any() else "lightweight")


def station_type(names):
    k = norm(names)
    t = k.map(gtype).fillna("unmatched")
    t[k.str.contains("public rack|corral")] = "lightweight"
    t[names.isna()] = "none"
    return t


raw["start_type"] = station_type(raw["start_station_name"])
raw["end_type"] = station_type(raw["end_station_name"])
pts = pd.concat([
    raw[["start_station_id", "start_lat", "start_lng"]].set_axis(["sid", "lat", "lng"], axis=1),
    raw[["end_station_id", "end_lat", "end_lng"]].set_axis(["sid", "lat", "lng"], axis=1)]).dropna()
stations = pts.groupby("sid").agg(lat=("lat", "median"), lng=("lng", "median"), n=("lat", "size"))
print(f"Canonical locations for {len(stations):,} station IDs")

tr = gpd.read_file(EXT / "tiger" / "tracts_chicago.gpkg")[["GEOID", "geometry"]]
g = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations["lng"], stations["lat"]),
                     crs="EPSG:4326").to_crs(tr.crs)
stations["tract"] = gpd.sjoin(g, tr, how="left", predicate="within").groupby(level=0)["GEOID"].first()
stations["dist_cbd_km"] = haversine_km(stations["lat"], stations["lng"], *CBD)
tf = pd.read_parquet(EXT / "tract_features_chicago.parquet").set_index("GEOID")[TRACT_VARS]

# ---------------------------------------------------------------- rebuild spatial features
keep = ["ride_id", "e_bike", "started_at", "start_station_id", "is_member", "hour", "day_of_week",
        "is_weekend", "month", "is_holiday", "year", "temp_c", "feels_like_c", "precip_mm", "snow_cm",
        "wind_kmh", "humidity_pct", "cloud_pct", "is_daylight", "precip_prev3h_mm"]
d = pd.read_parquet(PRO / "chicago_model_data.parquet", columns=keep)
d = d.merge(raw[["ride_id", "end_station_id", "start_type", "end_type"]], on="ride_id", how="left")
print("Start station type (trips):", d["start_type"].value_counts().to_dict())
d = d[d["start_type"] == "classic"].copy()
d["end_dock"] = (d["end_type"] == "classic").astype(int)

s = stations
for side, col in (("o", "start_station_id"), ("d", "end_station_id")):
    sid = d[col]
    d[f"{side}_lat"], d[f"{side}_lng"] = sid.map(s["lat"]), sid.map(s["lng"])
    d[f"{side}_tract"] = sid.map(s["tract"])
    d[f"{side}_dist_cbd_km"] = sid.map(s["dist_cbd_km"])
    t = tf.reindex(d[f"{side}_tract"])
    for v in TRACT_VARS:
        d[f"{side}_{v}"] = t[v].values

d["dist_km"] = haversine_km(d["o_lat"], d["o_lng"], d["d_lat"], d["d_lng"])  # NaN when no end station
d["round_trip"] = (d["start_station_id"] == d["end_station_id"]).astype(int)
d["same_tract"] = (d["o_tract"] == d["d_tract"]).astype(int)
# destination information is only kept for trips ending at a classic dock
dest_cols = [c for c in d.columns if c.startswith("d_")] + ["dist_km", "same_tract", "round_trip"]
d.loc[d["end_dock"] == 0, dest_cols] = np.nan

before = len(d)
d = d[d["o_tract"].notna() & ((d["end_dock"] == 0) | d["d_tract"].notna())]
d = d.drop(columns=["start_type", "end_type"])
print(f"Dropped {before - len(d):,} trips with a station outside Cook County tracts")
d = d.drop(columns=["o_lat", "o_lng", "d_lat", "d_lng"])
d.to_parquet(PRO / "trips_stations.parquet", index=False)

print(f"Saved {len(d):,} classic-dock-start trips; classic dock-to-dock {d['end_dock'].sum():,}")
print("Share ending at a dock by type:", d.groupby("e_bike")["end_dock"].mean().round(4).to_dict())
dd = d[d["end_dock"] == 1]
print("Dock-to-dock e-bike share:", round(dd["e_bike"].mean(), 4))
print("Distance (km) by type, dock-to-dock:\n", dd.groupby("e_bike")["dist_km"].describe().round(2))
print("Round trip share by type, dock-to-dock:", dd.groupby("e_bike")["round_trip"].mean().round(4).to_dict())
