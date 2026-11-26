# ontario-wildfire

## Build order

From the repository root, with `requirements.txt` installed in `.venv`, run in this order:

```sh
.venv/bin/python build_fire_weather.py
.venv/bin/python build_grid_weather.py spatial
.venv/bin/python build_grid_weather.py daily
.venv/bin/python build_labels.py
.venv/bin/python build_fire_history.py
.venv/bin/python build_static_features.py
```

Raw inputs and generated Parquet files stay out of Git; source pins and build audit JSON are tracked.

## 10 km fire-weather grid

The historical statutory fire-region footprint is the dissolved union of LIO Restricted Fire Zone polygons 1–36. It is not an active fire-ban map. See the [historical regulation](https://www.ontario.ca/laws/regulation/960207) and the [LIO catalogue](https://open.canada.ca/data/en/dataset/1f8c45b5-443b-415c-9cc1-6fab98455e98).

The pinned `data/ontario-fire-zones.geojson` snapshot came from this [exact download URL](https://ws.lioservices.lrc.gov.on.ca/arcgis2/rest/services/LIO_OPEN_DATA/LIO_Open09/MapServer/34/query?where=1%3D1&outFields=OFFICIAL_NAME&outSR=4326&f=geojson). Its raw-byte SHA-256 is `d29e12e479fb41a88f922a435bafb762284e641004858b6b52a529a2801e72a6`. The grid builder requires both the snapshot and the same hash in `data/ontario-fire-weather.parquet` metadata.

From the repository root, with the native CaSR v3.2 daily Parquet in place:

```sh
.venv/bin/python build_grid_weather.py spatial
.venv/bin/python build_grid_weather.py daily
```

`spatial` writes `data/grid_10km.parquet`, `data/grid_10km.svg` and `data/casr_to_grid_weights.parquet`. A full EPSG:3978 square on the 10,000 m origin-anchored lattice is included only if its centre lies strictly inside the unbuffered footprint. The native builder's 20 km buffer selects donors, not targets. Grid IDs are local to this snapshot; `casr_cell_id` identifies a native donor.

`spatial` fetches only the CaSR grid metadata and rotated axes from PAVICS. CaSR publishes no cell bounds; the builder uses adjacent rotated-coordinate midpoints and extrapolated endpoint half-spacings, with eight segments per side projected into EPSG:3978. Weights are whole-target-square planar overlap fractions normalized over the covered area only. Partial targets use only physically covered area; uncovered targets have no weights.

`daily` runs offline against the saved grid and weights. It writes `data/grid_fire_weather.parquet` and `data/grid_fire_weather.checks.json` for May–October 2010–2024. Each weather field and each native fire-weather index is averaged directly with its static weights. Null or nonfinite donors invalidate only the affected target field; uncovered targets remain null. Source dates are unchanged. The target noon label is 16 UTC at/east of 90°W, 17 UTC west of it; the label does not retime the source weather.

## NFDB report-date labels

Run `.venv/bin/python build_labels.py` after publishing `data/grid_10km.parquet`. The builder caches the [official Canadian National Fire Database point archive](https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_txt.zip), pinned to `NFDB_point_20260811.txt` and archive SHA-256 `9eb4773dfc60cfcfe7d76efa0f2470cdd341f4967fbec8084041f3fab87f8be3`. Cached runs are offline; a changed upstream archive is rejected. Source: [Canadian Forest Service / Natural Resources Canada, Canadian National Fire Database – Agency Fire Data](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb).

- `data/labels.parquet`: every `(cell_id, date)` from May 1 to October 31, 2010–2024, with `fire_any`, `fire_human`, `fire_lightning`, `n_fires`, and `check_fire_any_pm1`. The `check_` column is for audits only and is excluded from the model-table label allowlist. It includes April 30 and November 1 reports at the season edges.
- `data/ontario_nfdb_fires.parquet`: dated, non-prescribed, on-grid Ontario agency records (`SRC_AGENCY=ON`) for all months, 2000–2024, with human/lightning/unknown cause preserved. Exact report-date/latitude/longitude/cause matches merge under the lowest `fire_id` (numeric suffix order, e.g. `COC7` before `COC12`). `merged_fire_ids` includes the representative and every source member. Distinct positions on the same date within 100 m flag both retained fires as `near_duplicate` but do not merge them. Reused IDs alone do not trigger a merge. 2000–2009 is history input only; this build computes no features.
- `data/labels.checks.json`: annual filter counts and positive rates, records excluded for falling outside the published grid or lacking a report date, deduplication counts and label changes versus counting source rows, and three source-to-label spot checks.

Dates are NFDB report dates from `YEAR/MONTH/DAY`, not ignition estimates. Prescribed burns are excluded; unknown causes contribute to `fire_any` and `n_fires`, not the cause-specific flags. Points are projected to EPSG:3978 and joined to the full saved squares; exact grid lines go to the east/north square, with no snapping to nearby cells. Zero means no retained NFDB report for the cell-date, not proof that no fire occurred.

## Fire history features

Run `.venv/bin/python build_fire_history.py` after the grid and labels are built. `data/fire_history.parquet` has the exact keys of `data/labels.parquet`. Feature date `D` includes NFDB reports through `D` and is available for the target date `D+1`. The source is the deduplicated inventory from 2000 onward, including all off-season reports.

The 18 `fire_count_{cell|3x3}_{30d|365d|5y}_{all|human|lightning}` columns count reports in `(D − window, D]`, with five calendar years for `5y`. The neighbourhood includes the cell itself plus the eight adjacent lattice positions that are in the saved grid. Unknown causes contribute only to `all` counts. `days_since_last_fire_{human|lightning}` is same-cell recency over `(D − 1,826 days, D]`: zero on a report date, and 1,826 if no report of that cause falls in the window. It is never NaN or null.

`data/fire_history.checks.json` records all 20 yearly means, the fraction of recency values at 1,826, three report-date transitions, and the full-table comparison to an independent per-fire reference that never includes reports after `D`. Recency means include the capped values. No modelling runs in this build.

## Static grid features

Run `.venv/bin/python build_static_features.py` on the published grid. [data/static_sources.json](data/static_sources.json) pins the URL, vintage and SHA-256 of each dataset; local source files are cached in `.cache/static/`. For live services, preserve the exact cached extraction: the builder rejects a changed upstream response. Outputs are `data/static_features.parquet` (one row per saved `cell_id`), `data/static_features.checks.json`, and `data/static_features.svg`.

- **Access:** ORN physical road-centreline length clipped to each full cell, excluding virtual roads and ferry links. Access share uses the union of 1 km road and NRWN rail buffers, including lines beyond cell boundaries. Distances to current CGNDB populated community points are straight-line in the grid CRS, not travel distances.
- **Population:** 2021 Census population counts joined by DGUID to Statistics Canada's cartographic dissemination areas, queried with 5 m service generalization. Allocate each DA population uniformly across its polygon, then divide the allocated total by the full 100 km² square. `census_da_coverage_share` records the union of overlapping area; zero coverage means zero allocated residents, not an observed census count. A cell overlapping a DA with suppressed or unavailable population has density filled with zero and `pop_missing=true`.
- **Recreation:** named provincial-park points, OpenStreetMap `camp_site` and `caravan_site` points and polygon representative points (© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL), and Statistics Canada ODRSF facilities are counted separately. These inventories are not exhaustive and may overlap. Provincial-park share is the union of the regulated park boundaries, not a buffer around park points.
- **Fuels:** intersect native 250 m CFS FBP 2014b pixels with the squares by projected area. Six grouped shares sum to one over the area assigned to those groups; `fuel_classified_share` reports that area's fraction of the full square. NoData, omitted slash (S1–S3), and wetlands of unknown FBP type are outside this denominator. They are not treated as water or known non-fuel.

**Known temporal limitation:** these are fixed snapshots, not the state of the landscape in each label year. Post-2010 roads are knowingly used for early years; the same hindsight applies to other later snapshots. The 2014b fuel map uses historical burn scars and is not a pre-2010 snapshot.
