# Ontario daily fire weather, 2010–2024

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python build_fire_weather.py
```

The build publishes one `data/ontario-fire-weather.parquet` and a separate `data/ontario-fire-weather.checks.json`. The JSON reports annual row counts, full-table null fractions, and three independent rain-window checks. Completed weather blocks are cached under `.cache/fire-weather` so the build can resume after a transfer failure. Run one build at a time per cache/output path.

## Sources and conventions

- ECCC Canadian Surface Reanalysis **CaSR v3.2**, hourly 10 km native rotated grid. [Live PAVICS THREDDS catalog](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/reanalyses/catalog.xml), [hourly OPeNDAP aggregation](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/datasets/reanalyses/1hr_NAM_GovCan_CaSR_v32_196801-202412.ncml), [live variable attributes](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/datasets/reanalyses/1hr_NAM_GovCan_CaSR_v32_196801-202412.ncml.das). The older `198001-202412` hourly URL in the PAVICS notebook returns 404.
- The static footprint is Ontario's [statutory fire region](https://www.ontario.ca/laws/regulation/960207), the dissolved [Land Information Ontario zones 1–36](https://open.canada.ca/data/en/dataset/1f8c45b5-443b-415c-9cc1-6fab98455e98), not the area under temporary restrictions. Buffer by 20,000 m in NAD83 / Ontario MNR Lambert (EPSG:3161); keep native cell centres in or on the buffered polygon. The buffer is not clipped at provincial borders; no land-only mask or regridding is applied. `cell_id` is zero-based `rlat_index * 706 + rlon_index`. The Parquet metadata records both source URLs and the hash of the zone snapshot.
- `temperature`: `tas` (`P_TT_1.5m`), K → °C. `RH`: `hurs` (`P_HR_1.5m`), %. **The source's closest near-surface temperature, RH and optional dewpoint are at 1.5 m, not 2 m.** No inferred 2 m adjustment is applied. `wind`: `sfcWind` (`P_UVC_10m`), m/s → km/h.
- `noon_utc_hour` is 16 at/east of −90° longitude and 17 west of it. `rain_24h` uses the hourly CaPA analysis `pr` (`A_PR0_SFC`, kg m⁻² s⁻¹), integrated over the 24 one-hour intervals with valid timestamps D−1 13 UTC to D 12 UTC inclusive. This is total precipitation as water equivalent; it can include frozen precipitation. [ECCC documents](https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/dataset_specifics.html#champs_precip) that the underlying analysis is a 24 h field valid at 12 UTC, disaggregated to hourly flux. A missing hour makes the daily total null.
- Canadian FWI indices use [xclim 0.59.1 `cffwis_indices`](https://xclim.readthedocs.io/en/v0.59.1/apidoc/xclim.indices.fire.html), based on the Canadian Forest Service implementation of Van Wagner's FWI system. Start each April 1 update from FFMC/DMC/DC = 85/6/15. April updates are warm-up; write May 1–October 31 only. No overwintering, season autodetection, date shift, or modelling. A missing driver makes that day's indices and the rest of the cell's season null, without imputation.

## Copernicus check

Configure an [EWDS API key](https://ewds.climate.copernicus.eu/how-to-api) and accept the [dataset terms](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1?tab=download), then run:

```sh
.venv/bin/python compare_cems.py
```

Alternatively, pass predownloaded EWDS NetCDF/ZIP files with repeated `--reference PATH`. Use `cems-fire-historical-v1`, system 4.1, consolidated ERA5 reanalysis, 0.25° grid, all days May–October 2015 and 2018, the three indices, and a box covering the five sites (south 46°, north 57°, west −95°, east −79°). The CSV reports paired-day Pearson correlations and mean differences (CaSR minus CEMS) for each site/year/index using nearest reference grid centres and the same calendar date. CEMS is a comparison, not truth: its meteorology, local-noon interpolation, grid and moisture-code history can differ from this table.
