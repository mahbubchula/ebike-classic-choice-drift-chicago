"""Step 01: Data audit of the Divvy (Chicago) trip histories, classic and electric bikes.

Reads raw data (never modifies it), reports structure, missing values,
time coverage, vehicle-type mix and implausible records, and saves
tables to outputs/tables and a summary report to outputs/reports.

Run from the project root:
    python src/prep_01_data_audit.py
"""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
P = {k: ROOT / v for k, v in CFG["paths"].items()}
for k in ("tables", "reports", "interim"):
    P[k].mkdir(parents=True, exist_ok=True)
A = CFG["audit"]
report = []


def log(line=""):
    print(line)
    report.append(line)


def missing_table(df, name):
    t = pd.DataFrame({"n_missing": df.isna().sum(), "pct_missing": (df.isna().mean() * 100).round(3),
                      "dtype": df.dtypes.astype(str), "n_unique": df.nunique()})
    t.to_csv(P["tables"] / f"audit_{name}_missing.csv")
    return t


# ---------------------------------------------------------------- Divvy
log("# Data audit report\n")
log("## Divvy bike share, Chicago (classic vs. electric bikes)\n")
frames, schemas = [], {}
for z in sorted(P["raw_divvy_dir"].glob("*-divvy-tripdata.zip")):
    with zipfile.ZipFile(z) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        for n in csvs:
            d = pd.read_csv(zf.open(n), dtype=str)
            schemas[z.name] = tuple(d.columns)
            d["source_file"] = z.name
            frames.append(d)
dv = pd.concat(frames, ignore_index=True)
log(f"- Files read: {len(schemas)}; distinct column layouts: {len(set(schemas.values()))}")
log(f"- Columns: {', '.join(dv.columns[:-1])}")
log(f"- Rows: {len(dv):,}")
dv["started_at"] = pd.to_datetime(dv["started_at"], errors="coerce", format="mixed")
dv["ended_at"] = pd.to_datetime(dv["ended_at"], errors="coerce", format="mixed")
for c in ("start_lat", "start_lng", "end_lat", "end_lng"):
    dv[c] = pd.to_numeric(dv[c], errors="coerce")
dv["duration_min"] = (dv["ended_at"] - dv["started_at"]).dt.total_seconds() / 60
log(f"- Date range: {dv['started_at'].min()} to {dv['started_at'].max()}")
log(f"- Duplicate ride IDs: {dv['ride_id'].duplicated().sum():,}")
mt = missing_table(dv.drop(columns=["source_file"]), "divvy")
log("\nMissing values (%):\n")
log(mt[["n_missing", "pct_missing", "n_unique"]].to_markdown())

log("\nRideable type by rider type:\n")
ct = pd.crosstab(dv["rideable_type"], dv["member_casual"], margins=True)
ct.to_csv(P["tables"] / "audit_divvy_rideable_by_member.csv")
log(ct.to_markdown())

dv["ym"] = dv["started_at"].dt.to_period("M").astype(str)
monthly = pd.crosstab(dv["ym"], dv["rideable_type"])
monthly["electric_share_pct"] = (monthly.get("electric_bike", 0) / monthly.sum(axis=1) * 100).round(2)
monthly.to_csv(P["tables"] / "audit_divvy_monthly_by_type.csv")
log("\nMonthly trips and e-bike share:\n")
log(monthly.to_markdown())

desc = dv.groupby("rideable_type")["duration_min"].describe(percentiles=[.01, .5, .99]).round(2)
desc.to_csv(P["tables"] / "audit_divvy_duration_by_type.csv")
log("\nDuration (min) by rideable type:\n")
log(desc.to_markdown())
log(f"\n- Duration < {A['min_duration_min']} min: {(dv['duration_min'] < A['min_duration_min']).mean() * 100:.2f}%; "
    f"> {A['max_duration_min']} min: {(dv['duration_min'] > A['max_duration_min']).mean() * 100:.2f}%")
log(f"- Missing start station (dockless e-bike parking): {dv['start_station_id'].isna().mean() * 100:.1f}%")

dv.to_parquet(P["interim"] / "divvy_trips_2025_2026.parquet", index=False)
log("- Saved typed copy: data/interim/divvy_trips_2025_2026.parquet")

(P["reports"] / "01_data_audit_report.md").write_text("\n".join(report))
print("\nReport saved: outputs/reports/01_data_audit_report.md")
