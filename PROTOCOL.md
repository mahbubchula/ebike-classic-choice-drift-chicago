# Analysis protocol

Fixed before modeling (November 9, 2026). Every later decision was recorded here before the test set was opened.

## Question

What explains a rider's choice of an e-bike over a classic bike at a Divvy dock in Chicago, and do these explanations stay stable as the e-bike share of dock-start trips grows from 2025 to 2026?

## Unit, target and choice set

- **Main sample:** trips that start and end at an official **classic** Divvy dock (6,812,658 trips). Station type comes from Divvy's GBFS v1.1 `station_information` feed (`station_type` = classic or lightweight, snapshot of November 9, 2026), matched by normalized station name because station identifiers changed between 2025 and 2026. Names containing "public rack" or "corral" are lightweight. The rule never uses the bike type of a trip.
- **Origin-only sensitivity sample:** all trips that start at a classic dock (8,382,310 trips), without any destination information.
- **Excluded:** trips from lightweight stations (public racks and e-bike corrals offer only e-bikes, so no choice exists) and trips from station names not found in the feed (about 3%, mostly renamed or removed docks).
- **Target:** `e_bike` = 1 for an electric bike, 0 for a classic bike.
- **Not observed:** the number of bikes of each type docked when a trip started (stated as a limitation).

## Temporal design (test used once)

| Set | Period | Use |
|---|---|---|
| Excluded | 2025-01-01 to 2025-02-04 | Earlier price regime; used only in a robustness check |
| Inner training | 2025-02-05 to 2025-08-24 | Fit candidate models |
| Embargo | 2025-08-25 to 2025-08-31 | Discarded (seven-day gap) |
| Validation | 2025-09-01 to 2025-12-31 | Early stopping, hyperparameters, model choice, decision threshold |
| Final training | 2025-02-05 to 2025-12-31 | Refit of the chosen configuration |
| Test | 2026-01-01 to 2026-10-31 | Evaluated once, overall and by month |

Training starts on February 5, 2025, the day non-member e-bike prices rose; training across that change would ask the model to learn a price effect it cannot observe.

## Leakage controls

1. **Post-choice variables are excluded:** trip duration, speed, end time and whether the trip ended at a dock (only e-bikes can be parked away from a dock).
2. **Canonical station locations.** Classic bikes are recorded at fixed station coordinates, whereas e-bikes carry their own GPS fix, and e-bikes parked away from a dock have end coordinates rounded to two decimals. Every station therefore receives a single location, the median of all recorded coordinates at that station with both bike types pooled. All spatial features (distance, tracts, distance to the Loop) are computed from these locations, and a round trip is a trip that starts and ends at the same station.
3. **Choice set by official station type.** Lightweight stations are removed using the GBFS station type, not the observed bike mix, and the main sample is restricted to dock-to-dock trips.
4. **Nothing is computed from the target outside the training set.** No station or tract e-bike shares enter the main model.
5. **Training-only feature reduction.** The correlation filter (absolute Spearman correlation of at least 0.7, fixed priority list) is computed on the inner training set.
6. **Pipelines.** Imputation, scaling and spline bases sit inside scikit-learn pipelines fitted on training data; gradient-boosting models use native missing-value handling.
7. **Temporal order.** No shuffling across time; `year` is never a feature; calendar `month` is not a main feature (it is confounded with price and fleet changes within one training year) and enters only a robustness model.
8. **Pre-period external data.** ACS 2019 to 2023 (released December 2024) and LODES 2022 precede the first trip. Weather is the hour of trip start; rain in the previous three hours uses past hours only.
9. **Duplicates and overlap.** Duplicated ride identifiers are removed; training, validation and test share no ride identifier (checked in code).
10. **Prior shift.** Test probabilities are not recalibrated on test data; calibration drift is reported as a result.
11. **Operator changes** (February 5, 2025; early July 2025; March 16, 2026; `data/context/policy_timeline.csv`) are not features. They are used to interpret drift.

## Feature sets

- **Main model:** time, rider type, weather, origin and destination context, and straight-line distance between canonical station locations.
- **Origin-only model:** drops every destination-based feature, because a rider on an e-bike may choose a farther destination.
- **Supply extension:** adds the e-bike share of departures from the origin station over the seven previous days (missing below ten departures).

## Models and selection

Prior-only model, spline logistic regression, LightGBM, XGBoost and CatBoost. Untuned models are compared on validation; the best boosting library is tuned with Optuna (30 trials, validation log loss) and refitted on final training. The decision threshold maximizes balanced accuracy on validation. A SHA-256 hash of every model file is written before the test evaluation, and the evaluation script refuses to run if any model changes.

## Metrics

ROC-AUC, PR-AUC, log loss, Brier score, balanced accuracy, F1 and MCC at the validation threshold, plus calibration intercept and slope. Confidence intervals come from a day-block bootstrap. Test metrics are reported overall, by month, by rider type and before and after March 16, 2026.

## Explanation plan

TreeSHAP (global importance, interactions, local cases), accumulated local effects by rider type and year, permutation importance within month, and a season-matched decomposition of the change in mean log-odds into feature-group contributions. Post hoc analyses (after the single test): seed stability, year-specific twin models and monthly updating policies. SHAP describes model behavior and is not causal evidence.

## Tuning note (recorded before the test set was opened)

During the Optuna search LightGBM warned that `max_bin` cannot change after a Dataset is constructed. Because Datasets kept their raw data, LightGBM rebuilt them with each trial's parameters. Three recorded trials were re-run on freshly built Datasets and reproduced the validation log losses to six decimals (0.662235, 0.659184, 0.656065), so the search is valid. Tuned validation results: main AUC 0.6649, log loss 0.6523 (1,065 trees, threshold 0.450); with supply proxy 0.6669, 0.6487 (1,037 trees, 0.465); origin-only 0.6029, 0.6704 (260 trees, 0.540).
