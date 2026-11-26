# COMP-385 capstone ignition review (for the solo rebuild)

Repo: https://github.com/parthk1611/COMP-385, shallow clone at commit `1b91fcf` (2026-11-23 14:24 ET), on the box at `/workspace/wildfire-solo/repo`.
Scope: ignition only. `capability_2_fire_spread/` was ignored.
Method: I read the code and the tracked audits/docs. I recomputed a few things myself, each marked **[verified by me]**. Anything that is only the team's own claim is marked as such.

---

## 1. Grid / domain
- **Cell / CRS:** 10 km squares in EPSG:3978 (NAD83 Canada Atlas Lambert), snapped to `floor(x/10000)*10000`. The projection is hand-coded rather than using pyproj (`scripts/build_cpa_daily_ignition_labels.py:28-72, 133-138`).
- **Domain:** the Ontario Fire Response Plan Area, all sectors except `OFR` ("Outside Fire Region"), unioned (`:96-130`). Season is May 1 to Oct 31 only (`:29-30, 150-158`), 2010-2019 = 1,840 days.
- **Cell universe:** a cell is kept if its centre falls in the region **OR if it contains any selected 2010-2019 NFDB ignition** (`build_grid`, `:188-200`, and `event_grid_ids` passed in at `:297`). So the 2017-2019 holdout fires helped decide which cells exist.
- **Count:** 9,673 cells and 17,798,320 cell-days (`data/processed/..._audit.json`, `data/interim/ontario_fire_region_10km_grid.csv`).
- **[verified by me]** I re-downloaded the boundary. It is a newer vintage: 31 features as a FileGDB, versus 15 features in the original GeoJSON, so the numbers are approximate. Result: **~25 cells have their centre outside the region**, meaning they exist only because a fire landed in them. Those 25 cells hold 41 events / 40 positive cell-days. **12 of them first appear because of a 2017-2019 fire, and 6 because of a 2019 (test-year) fire.** That makes those cells guaranteed positives in the test set. The effect is small (about 6 of 509 test positives) but it is real. The team's own report flags this rule but never measured it (`docs/modeling/temporal_leakage_report.md:94-98`).
- The original boundary checksum can no longer be reproduced. The grid survives only as the frozen CSV (`build_from_frozen`, `:236-275`).

## 2. Labels
- **Source:** NFDB point "current_version", file `NFDB_point_20260811.txt`, archive sha256 `9eb4773d…` (`data/raw/nfdb/manifest.json`; `scripts/download_nfdb.py:15-18`). Rows are filtered to `SRC_AGENCY == "ON"` and years 2010-2019 (`scripts/prepare_ontario_ignitions.py:67-92`). **[verified by me]** Today's download is byte-identical.
- **Date field:** the code builds the date from `YEAR/MONTH/DAY` (`prepare_ontario_ignitions.py:72-73`). **[verified by me]** For all 8,740 Ontario 2010-2019 rows, Y/M/D equals `REP_DATE` exactly, and the NFDB metadata says YEAR is derived from REP_DATE. So **the label is the report date, not the ignition date.** It has no time of day. `ATTK_DATE` is empty for all of these rows.
- **Timezone convention:** the report date is treated as a local America/Toronto date. Issue time is 00:00 local on day *t*, and the target is "≥1 recorded fire on local date *t*" (`docs/modeling/split_manifest.json` date_contract; `build_ignition_feature_panel.py:497`). So "tomorrow" in practice means **same day *t*, predicted at midnight using data up to day *t-1***.
- **Point to cell:** the WGS84 lat/lon is projected and floored to a cell (`build_cpa_daily_ignition_labels.py:174-183`). There is no tolerance for NFDB location precision. 8,740 → 8,109 events: 620 dropped as out of season and 11 as outside the region. That gives 7,785 positive cell-days, and 265 cell-days hold 2 or more fires **[verified by me]**.
- **Cause:** **all causes are pooled** (no cause filter anywhere). Region events: N 4,776, H 3,182, U 151 **[verified by me]**.
- **[verified by me] 9 prescribed burns (`FIRE_TYPE/PRESCRIBED = PB`) are included as positive ignitions.** There is no filter on `is_prescribed`.
- Also stored but not used as a target: `ignition_count` and `large_fire_200ha`.

## 3. Features (39 final predictors; `data/metadata/ignition_feature_selection_v1.json`)
| Group | Features | Source / rule |
|---|---|---|
| Calendar (5) | doy, sin/cos doy, is_weekend, day length | target date *t*, deterministic (`build_ignition_feature_panel.py:490-504`) |
| Terrain (7) | elevation, slope, aspect sin/cos, ruggedness, sample count, coverage | CDEM via geogratis profile API, **only 9 sample points per 100 km² cell** (`build_cdem_terrain_summary.py:5-7,31`). This is very coarse, and slope/aspect at this scale mean little. |
| Weather/FWI (19) | station count, nearest km, coverage, imputed/calcstatus fractions, missing flag, temp, RH, wind, precip, FFMC, DMC, DC, ISI, BUI, FWI, DSR, 7-day precip + coverage | CWFIS `cwfis_fwi2010sv3.0_ll.csv.bz2` station archive (noon obs). Takes the nearest 4 stations with inverse-distance² weighting, values dated **t-1**, and 7-day rain over t-7..t-1, left blank unless all 7 days are present. |
| Fire history (6) | same-cell and 3×3 counts over 30 and 365 days, days-since-last, censor flag | the selected NFDB events themselves, strictly `< t` (`FireHistory.advance`, cited in the leakage report :88) |
| Coordinates (2) | cell-centre easting/northing (EPSG:3978) | `grid_x_m+5000`, `grid_y_m+5000` (`build_ignition_feature_panel.py:545-546`) |

- **Lightning** (CanCPLD CG, 0.1° 3-hourly; trailing 24 h and 7 d windows ending 23:00 local on t-1) was **tested only as an experiment and not retained** (`experiments/ignition/lightning/README.md`). Its README says there is no outage flag, so **detector uptime was never verified** (a zero might mean "no data"). Only 17.2% of natural-cause positive cell-days had any lightning in the prior 24 h (`lightning_diagnosis`). That fits holdover fires and/or reporting lag.
- **Absent entirely:** fuels/land cover, human presence (roads, towns, population, rail, cottages), and cause-specific modelling.
- **Scaling:** the logistic preprocessor is fit on the training partition only (`scripts/ignition_modeling/logistic.py:79-81`). HGB uses no scaling and native NaNs. **I found no normalization fit on all years.**

## 4. Splits / evaluation
- Splits are by year (`scripts/ignition_modeling/splits.py:8,13-20`): development 2010-2016; **2017 = calibration fit; 2018 = calibration decision and threshold quantiles; 2019 = test**. Development folds expand forward and validate on 2012, 2013, 2014, 2015 and 2016, with no purge gap.
- **2017 was touched:** calibrators were fit on it (`docs/modeling/final_freeze.json`: `calibration_fit_year: 2017`, `calibration_used: false`). 2018 was used to choose against calibration and to set thresholds. 2019 was scored once for the original 40-iteration model.
- **Metrics:** mean yearly average precision (PR-AUC) as the selection metric, plus ROC-AUC, Brier, log-loss, reliability bins, and daily top-k budgets (ranking_utility).
- **Imbalance:** all negatives kept, with no resampling. `class_weight` ∈ {None, balanced} was searched and **None won** for HGB (`docs/modeling/development_comparison.csv`).
- **Results:** 2019 test AP 0.0073 (25× the 0.029% prevalence) and ROC-AUC 0.889 (`docs/modeling/final_test.json`). Prevalence swings from 0.029% to 0.070% between years (2019 vs 2018).

## 5. Model
- **Original frozen model:** HGB with 40 iterations, lr 0.1, 15 leaves, min leaf 100, L2 1.0, no class weight, fit on 2010-2016, uncalibrated (`final_freeze.json`). It beat SGD-logistic.
- **Retained / demo model:** the same settings but **80 iterations plus 2 coordinates** (39 inputs), seed 385, fit on 2010-2016 (`scripts/ignition_modeling/contract.py:26`; `experiments/ignition/README.md:51-62`). Development mean AP is 0.00697. **It was chosen after 2019 had already been seen and was never scored on an untouched year.** The demo then serves 2019 predictions from it (`scripts/ignition_demo/common.py:43`).

## 6. Ranked risks (verified in code)
1. **Holdout contamination by reuse.** Thirteen follow-up experiments (tuning, randomized, local grid, Bayesian TPE with 80 fits, XGBoost, lightning ×2, recent history, coordinates…) all selected on the same 2012-2016 folds, *after* 2019 had been viewed. 2017 and 2018 were already spent on calibration and thresholds. Today there is no clean year left in 2010-2019. Every gain in the experiments was 1-6%, which sits inside the year-to-year noise (for example, coordinates hurt 2013).
2. **Weather station lookahead (original v1 panel).** **Confirmed.** `load_station_catalog` scans the full 2010-2019 archive, `nearest_station_choices` picks the nearest 4 from that all-time list, and the `first_seen` filter runs only afterwards (`build_ignition_feature_panel.py:277-318, ~395`). So future stations displace stations that really existed. The archive is also not date-sorted (2,183 backward jumps, per the team's audit). The corrected cache (`scripts/ignition_modeling/weather.py:102-119`) admits stations by `first_seen <= day` *before* picking the top 4, and indexes dates globally. **The v1 panel CSV and the EDA still carry the buggy weather**, so anyone reading that CSV directly gets leaky values.
3. **A remaining design flaw in the corrected weather (mine, not in their audit).** `AsOfStations` only ever adds stations and has no "last seen", and it never falls back to a 5th station when one of the 4 reports nothing that day (`weather.py:110-119,129`). Dead or gappy stations keep their slots forever, which drives missingness (1.28M rows with no weather, 2.17M with no 7-day rain). This is not leakage, but it creates non-stationary missingness, and the model can learn the missingness pattern itself.
4. **Rain window.** The corrected version uses exactly t-7..t-1 with an as-of station set each day, and resets the window across season gaps (`weather.py:150-170`). **I found no bug in the corrected code.** The original 7-day rain inherited risk 2.
5. **Edge-cell universe shaped by 2010-2019 fires, including the test year.** Confirmed in code, plus my approximate count: about 25 cells, 6 of which were added by 2019 fires.
6. **Label = report date.** Fires that hold over (lightning especially) or are detected late get pinned to the wrong day. This blurs the weather→ignition link and may explain why lightning didn't help.
7. **Pooled causes and 9 prescribed burns counted as ignitions.** Human and lightning fires have different drivers.
8. **Fire-history censoring.** History counts only May-Oct events from 2010 onward, so the 365-day features are empty or short in 2010 and partially filled in 2011. That makes the history features non-stationary between early training years and later years.
9. **Missing driver families:** no fuels and no human presence. The coordinates were standing in for "where people and fuels are", which can't transfer to new areas.
10. **Archive availability unknown.** CWFIS values may be quality-controlled or revised after the fact, and the archive has no `available_at` field (team's own caveat). The CDEM release is from 2013.

## Decisions to make fresh
1. Grid universe: boundary-only and target-independent, fixed before looking at any fires.
2. Label date: REP_DATE vs something better, plus a "fire within ±1 day" sensitivity check. Drop prescribed burns (`PB`). Keep fire_type IFR only.
3. Cause: separate human and lightning models (or targets), or one pooled model with cause-aware evaluation.
4. Holdout: pick untouched years now (for example, add 2020-2024 if the sources cover them), write the choice down, and touch them once. Cap how many experiments run on the development folds.
5. Weather source: the station archive with an as-of rule, last-seen handling and fallback, versus gridded reanalysis (ERA5 / CaSR / CWFIS gridded FWI) with no station topology at all.
6. Add human-access and fuel features (roads, population/towns, FBP fuels / land cover). Decide whether coordinates are allowed at all.
7. Lightning: include it only with an uptime/coverage check, and use windows that cover holdovers (3-7+ days).
8. Fire history: decide whether to include it, and if so, seed it from pre-study years so the early years aren't censored.
9. Metric: AP plus top-k% daily capture as the main metrics, and report year-to-year spread rather than a single mean.
10. Imbalance: keep every negative (as they did; class weighting hurt), or subsample and correct the probabilities.
