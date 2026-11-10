#!/usr/bin/env bash
# Full pipeline, in order. Run from the repository root after placing the inputs listed in data/README.md.
set -euo pipefail
PY=${PYTHON:-python}

$PY src/prep_01_data_audit.py
$PY src/prep_02_external_data.py
$PY src/prep_03_clean_and_features.py
$PY src/step_03_station_features.py
$PY src/step_03c_supply_proxy.py
$PY src/step_04_descriptive_figures.py
$PY src/step_04b_feature_correlation.py
$PY src/step_05_split.py
for s in main origin_only; do $PY src/step_06_candidates.py --set "$s"; done
for s in main main_plus_supply origin_only; do $PY src/step_07_tune_and_refit.py --set "$s"; done
$PY src/step_08_test_once.py          # single test evaluation (locked models)
$PY src/step_09_xai_locked.py
$PY src/step_10_xai_refits.py
$PY src/step_11_xai_figures.py
$PY src/step_12_robustness.py
$PY src/step_13_manuscript_tables.py
