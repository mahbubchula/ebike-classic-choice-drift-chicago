"""Step 03: Clean trips and build the trip-level modelling dataset.

Target: e_bike = 1 if the trip used an electric bike, 0 if it used a classic bike.

Rules applied here
  * Only trips that START at a docking station are kept, so both bike types could have been chosen.
  * Trip duration is never a feature (it is a consequence of the vehicle chosen).
  * No statistic of the target is computed here.
  * Spatial features are rebuilt from canonical station locations in step_03_station_features.py.

Outputs
  data/processed/chicago_model_data.parquet
  outputs/tables/cleaning_flow_chicago.csv

Run:  python src/prep_prep_03_clean_and_features.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
A = CFG["audit"]
EXT, INT, PRO, TAB = (ROOT / "data" / "external", ROOT / "data" / "interim",
                      ROOT / "data" / "processed", ROOT / "outputs" / "tables")
PRO.mkdir(parents=True, exist_ok=True)

CBD = {"chicago": (41.8819, -87.6278)}  # the Loop
TRACT_VARS = ["pop_density_km2", "job_density_km2", "median_income", "median_age", "pct_age_18_29",
              "pct_degree", "pct_hh_no_car", "pct_commute_transit", "pct_commute_bicycle",
              "pct_commute_walk", "pct_commute_wfh", "pct_jobs_retail_food"]
WEATHER_VARS = {"temperature_2m": "temp_c", "apparent_temperature": "feels_like_c",
                "precipitation": "precip_mm", "snowfall": "snow_cm", "wind_speed_10m": "wind_kmh",
                "relative_humidity_2m": "humidity_pct", "cloud_cover": "cloud_pct", "is_day": "is_daylight"}
HOLIDAYS = set(USFederalHolidayCalendar().holidays("2019-01-01", "2026-12-31").date)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class Flow:
    """Keeps a CONSORT-style record of how many trips remain after each rule."""
    def __init__(self, city, df):
        self.city, self.rows = city, [("raw records", len(df), 0)]

    def step(self, label, df, mask):
        before = len(df)
        df = df[mask].copy()
        self.rows.append((label, len(df), before - len(df)))
        print(f"  {label:<55} kept {len(df):>11,}  removed {before - len(df):>10,}")
        return df

    def save(self):
        pd.DataFrame(self.rows, columns=["step", "remaining", "removed"]).to_csv(
            TAB / f"cleaning_flow_{self.city}.csv", index=False)


def add_time(df, dt):
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["is_holiday"] = dt.dt.date.isin(HOLIDAYS).astype(int)
    df["year"] = dt.dt.year
    df["datetime_hour"] = dt.dt.floor("h")
    return df


def add_weather(df, city):
    w = pd.read_parquet(EXT / "weather" / f"weather_hourly_{city}.parquet").rename(
        columns={"datetime_local": "datetime_hour", **WEATHER_VARS})
    w = w[["datetime_hour", *WEATHER_VARS.values()]]
    # precipitation in the previous 3 hours (wet surfaces)
    w = w.sort_values("datetime_hour")
    w["precip_prev3h_mm"] = w["precip_mm"].shift(1).rolling(3, min_periods=1).sum()
    return df.merge(w, on="datetime_hour", how="left")


def add_tracts(df, city, prefix, geoid_col):
    t = pd.read_parquet(EXT / f"tract_features_{city}.parquet")
    t = t[["GEOID", *TRACT_VARS]].rename(columns={v: f"{prefix}_{v}" for v in TRACT_VARS})
    return df.merge(t, left_on=geoid_col, right_on="GEOID", how="left").drop(columns="GEOID")


def points_to_tract(lat, lon, city):
    """Spatial join on unique coordinates only (fast), then map back."""
    tr = gpd.read_file(EXT / "tiger" / f"tracts_{city}.gpkg")[["GEOID", "geometry"]]
    u = pd.DataFrame({"lat": lat.round(5), "lon": lon.round(5)}).drop_duplicates()
    g = gpd.GeoDataFrame(u, geometry=gpd.points_from_xy(u["lon"], u["lat"]), crs="EPSG:4326").to_crs(tr.crs)
    j = gpd.sjoin(g, tr, how="left", predicate="within").drop_duplicates(["lat", "lon"])
    key = pd.DataFrame({"lat": lat.round(5).values, "lon": lon.round(5).values})
    return key.merge(j[["lat", "lon", "GEOID"]], on=["lat", "lon"], how="left")["GEOID"].values


# =========================================================== Chicago (Divvy)
print("--- Chicago")
dv = pd.read_parquet(INT / "divvy_trips_2025_2026.parquet",
                     columns=["ride_id", "rideable_type", "started_at", "ended_at", "start_station_id",
                              "start_lat", "start_lng", "end_lat", "end_lng", "member_casual", "duration_min"])
f = Flow("chicago", dv)
dv = f.step("drop duplicated ride_id", dv, ~dv["ride_id"].duplicated(keep="first"))
dv = f.step("start date within 2025-01-01 to 2026-10-31", dv, dv["started_at"] >= "2025-01-01")
dv = f.step(f"duration {A['min_duration_min']}-{A['max_duration_min']} min (incl. negative)", dv,
            dv["duration_min"].between(A["min_duration_min"], A["max_duration_min"]))
dv = f.step("end coordinates present", dv, dv["end_lat"].notna() & dv["end_lng"].notna())
dv = f.step("trip starts at a docking station (both types available)", dv, dv["start_station_id"].notna())
dv["dist_km"] = haversine_km(dv["start_lat"], dv["start_lng"], dv["end_lat"], dv["end_lng"])
dv = f.step("straight-line distance <= 30 km", dv, dv["dist_km"] <= 30)
dv["o_tract"] = points_to_tract(dv["start_lat"], dv["start_lng"], "chicago")
dv["d_tract"] = points_to_tract(dv["end_lat"], dv["end_lng"], "chicago")
dv = f.step("start and end inside Cook County tracts", dv, dv["o_tract"].notna() & dv["d_tract"].notna())
f.save()

dv["e_bike"] = (dv["rideable_type"] == "electric_bike").astype(int)
dv["is_member"] = (dv["member_casual"] == "member").astype(int)
dv["round_trip"] = (dv["dist_km"] < 0.1).astype(int)
dv["same_tract"] = (dv["o_tract"] == dv["d_tract"]).astype(int)
dv["o_dist_cbd_km"] = haversine_km(dv["start_lat"], dv["start_lng"], *CBD["chicago"])
dv["d_dist_cbd_km"] = haversine_km(dv["end_lat"], dv["end_lng"], *CBD["chicago"])
dv = add_time(dv, dv["started_at"])
dv = add_weather(dv, "chicago")
dv = add_tracts(dv, "chicago", "o", "o_tract")
dv = add_tracts(dv, "chicago", "d", "d_tract")
dv = dv.drop(columns=["rideable_type", "member_casual", "ended_at", "end_lat", "end_lng"])
dv.to_parquet(PRO / "chicago_model_data.parquet", index=False)
print(f"  saved {len(dv):,} trips; e-bike share {dv['e_bike'].mean():.3f}")
del dv
