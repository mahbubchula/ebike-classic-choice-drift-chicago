# Data

Raw data are **not** included in this repository. Divvy trip histories are licensed under the
[Divvy Data License Agreement](https://divvybikes.com/data-license-agreement), which does not allow redistribution.
Place the files below at the paths shown, then run the pipeline from the repository root.

### Trip records

| Input | Save as | Source |
|---|---|---|
| Divvy monthly trip histories, January 2025 to October 2026 (22 files) | `data/raw/divvy/YYYYMM-divvy-tripdata.zip` | https://divvy-tripdata.s3.amazonaws.com/index.html |
| Divvy GBFS v1.1 `station_information` snapshot | `data/external/gbfs/station_information_v11_2026-11-09.json` | https://gbfs.lyft.com/gbfs/1.1/chi/en/station_information.json |

The analysis used a station feed snapshot taken on November 9, 2026. A later snapshot may list a slightly different set of stations.

### Census and weather inputs

| Input | Save as | Source |
|---|---|---|
| TIGER/Line 2023 census tracts, Illinois | `data/external/tiger/tl_2023_17_tract.zip` | https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_17_tract.zip |
| ACS 2019 to 2023 5-year, tables B01001, B01002, B01003, B08301, B15003, B19013, B25044 | `data/external/acs/acs2023_<table>.dat` (for example `acs2023_b01001.dat`) | https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/ (files `acsdt5y2023-<table>.dat`) |
| LEHD LODES 8, Workplace Area Characteristics, 2022 | `data/external/lodes/il_wac_S000_JT00_2022.csv.gz` | https://lehd.ces.census.gov/data/lodes/LODES8/il/wac/il_wac_S000_JT00_2022.csv.gz |
| Hourly weather (ERA5 reanalysis) | downloaded automatically by `prep_02_external_data.py` | Open-Meteo historical API, https://open-meteo.com/en/docs/historical-weather-api |

The Census API requires a key, so the table-based summary files are used instead. All census inputs were released before the study period they describe ends, and they are treated as time-invariant tract attributes.

### Included context file

`data/context/policy_timeline.csv` lists the operator price and fleet changes (February 5, 2025; early July 2025; March 16, 2026) with their sources. These events are never model features; they are used to interpret drift.
