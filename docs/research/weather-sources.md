# Gridded weather / fire-weather source comparison (solo Ontario ignition project)
Researched 2026-11-26 (ET). Issue time = 00:00 America/Toronto on day t; features may use data up to the end of day t-1.
"Checked" = I tested it live from the box. Everything else comes from the cited documentation.

## NFDB coverage (checked)
- Same NFDB point file as the capstone (`NFDB_point_20260811.txt`, sha256 9eb4773d…), downloaded today.
- Ontario (`SRC_AGENCY=ON`) fire counts by year: 2020: 609, 2021: 1,200, 2022: 276, 2023: 748, 2024: 491, **2025: 649**. The latest year is 2025.
- Causes for 2020+: N 2,572, H 1,341, U 60.
- So a **2020–2024 holdout is covered by NFDB**, and 2025 exists too. 2025 may still be revised; that is unverified.
- Source: https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_txt.zip

## (a) CWFIS gridded daily FWI / weather (NRCan)
- **Method:** the weather grids are IDW interpolation (weight 1/d²) of the **nearest 12 stations**. FWI is then computed from those grids. Times are local noon. https://catalogue.cwfif.nrcan.gc.ca/geonetwork/srv/api/records/66b81ea6-3056-46d0-873f-7a773a9326bc and https://catalogue.cwfif.nrcan.gc.ca/geonetwork/srv/api/records/efec0079-273e-47d4-84d7-6940e0af3932
- **Checked (GeoServer WCS DescribeCoverage + GetCoverage):**
  - `public:fwi` time range starts **2018-01-01**, native grid 2 km in EPSG:3978.
  - `temp`, `rh`, `ws`, `precip`, `ffmc`, `dmc`, `dc` start only **2024-01-01**.
  - Both run to 2026-12-08, which means the mosaic mixes in forecasts.
  - Requests for 2010, 2016 and 2017 return errors. The archived PNG URLs for 2010–2017 return HTML, not images.
  - The catalogue says "2000–", but in practice **there is no public raster archive for 2010–2017**.
  - Endpoint: https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs (docs: https://cwfis.cfs.nrcan.gc.ca/downloads/CWFIS_DataServices_HowtoAccessDailyMaps%26DataLayers.pdf)
- **Timing:** day 0 starts as a forecast and is replaced by observations in the late afternoon (MST). The archive stores whatever the final map was.
- **Station issues:** **yes, it inherits them.** The changing station network shows up directly as changing interpolation quality. It is 2 km, but its real information content follows station spacing, which is sparse in the north.
- **Verdict:** rejected as the primary source. It can't cover 2010–2017, and it is station-derived. It could serve as a 2018+ cross-check.

## (b) ERA5 and ERA5-Land (ECMWF / Copernicus C3S)
- **ERA5:** 0.25° (~28 km N–S, ~18 km E–W at 50°N), hourly, 1940–present.
- **ERA5-Land:** 0.1° grid (native 9 km), hourly, 1950–present. It replays the land model forced by ERA5 and **assimilates no observations itself**. https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation
- **Variables for FWI:** 2 m temperature, 2 m dewpoint (→ RH), 10 m u/v wind, total precipitation.
- **Accumulation quirk:** ERA5-Land accumulations run from 00 UTC.
- **Noon local:** 12:00 EDT = 16 UTC (most of Ontario). Northwest Ontario on CDT = 17 UTC.
- **Revisions:** ERA5T and ERA5-Land-T are preliminary (about 5 days behind) and replaced by the final version after ~2–3 months. That is irrelevant for 2010–2024, which is final. https://climate.copernicus.eu/what-copernicus-climate-change-services-era5-reanalysis-dataset
- **Hindcast caveat:** this is reanalysis, not what forecasters knew at the time. ERA5's 4D-Var windows are 09–21 and 21–09 UTC, so an analysis can absorb observations a few hours after its valid time. Rule: **use noon-local values and noon-to-noon precipitation ending at noon t-1, or at least ending ≤ 04 UTC on day t.** Avoid "calendar-day" sums that end at local midnight.
- **Access:** CDS API (free account, accept the licence). Size: an Ontario subset of 4 variables at 2 hours/day over 15 seasons is about 1–2 GB.
- **Ontario caveats:** ERA5 precipitation is model-generated (no gauge assimilation); it has known drizzle and wet bias, which damps FFMC. The Great Lakes coastline is smeared at 0.25°.

## (c) Copernicus CEMS / GEFF fire danger historical (`cems-fire-historical-v1`)
- **What it is:** ready-made Canadian FWI system indices (FFMC, DMC, DC, ISI, BUI, FWI, DSR) plus others, computed from ERA5.
- **Coverage:** **1940–present, daily**, 0.25° regular grid (native N320 ~31 km). https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1?tab=overview
- **Timing:** a **local-noon snapshot**, built by weighting the two nearest timesteps (newer method). https://confluence.ecmwf.int/spaces/CEMS/pages/340779714/User+Guide+for+Fire+danger+indices+historical+data+from+the+Copernicus+Emergency+Management+Service
- **Access:** EWDS API at https://ewds.climate.copernicus.eu/api (not the CDS endpoint), `dataset_type: consolidated_dataset`, `system_version 4_1`. The volume is tiny (MBs for Ontario).
- **Caveats:**
  - It inherits ERA5's precipitation bias.
  - Each 0.25° pixel covers about 4–6 of our 10 km cells, so neighbouring cells get identical values.
  - Overwintering/start-up rules for DC are not stated on the pages I read (uncertain).
  - Exact precipitation accumulation window not verified (uncertain).
  - Same hindcast caveat as ERA5.

## (d) ECCC CaPA / RDPA precipitation, and CaSR (ECCC Canadian Surface Reanalysis)
- **RDPA (CaPA):**
  - 10 km precipitation analysis that assimilates Canadian gauges and radar.
  - 24 h totals valid at 06 and 12 UTC; 6 h totals at 00/06/12/18 UTC.
  - Online GRIB2 archive runs **2011-04-06 → 2023-09-23**, plus a reanalysis covering 2002–2012-06. The Datamart itself is rolling, not an archive.
  - So there is a gap from late 2023 onward, unless you use CaSR or the live feed.
  - https://eccc-msc.github.io/open-data/msc-data/nwp_rdpa/readme_rdpa-datamart_en/
- **CaSR v3.2:**
  - ERA5-driven regional reanalysis plus CaLDAS land assimilation plus CaPA precipitation reanalysis.
  - **10 km (0.09° rotated grid), hourly, 1968-01-01 → 2024-12-31.** https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/dataset_specifics.html
  - **Checked:** PAVICS THREDDS has both `1hr_NAM_GovCan_CaSR_v32_196801-202412.ncml` and `day_NAM_GovCan_CaSR_v32_1968-2024.ncml`, served over open OPeNDAP with no account. https://pavics-sdi.readthedocs.io/en/latest/notebooks/CaSR%5Fbasic.html
  - Daily variables include hurs/hursmin, wind, pr, and `cfia` (the CaPA confidence index). Full raw archive is about 12 TB via wget on GPSC-C. For a noon-only Ontario subset over 15 seasons, expect about 2 GB (my estimate).
  - **Timing traps:**
    1. Each file `YYYYMMDD12` covers 13 UTC day D → 12 UTC day D+1.
    2. CaPA 24 h totals are valid at **12 UTC (08:00 EDT)**.
    3. **Hourly precipitation is disaggregated from the 24 h analysis.** Even pre-midnight hourly rain values are scaled by a total that includes observations up to 08:00 local on day t.
    → **For day t, use only the 24 h analysis valid at 12 UTC on t-1** (rain from 12 UTC t-2 to 12 UTC t-1), plus the model-only `P_PR0` if you need the afternoon of t-1.
    - The PAVICS daily aggregates are probably UTC days (unverified). Don't use them for FWI; take the hourly noon-local values.
  - **Ends 2024.** That works for a 2020–2024 holdout, but 2025 would need a "CaSR-like" rebuild from RDPS + CaPA.
  - **No ready FWI:** compute it yourself (e.g. xclim's FWI implementation) from noon T, RH, wind and 24 h precipitation.
  - Caveat: the CaPA confidence index is low where gauges are sparse (far north). Use `cfia` as a quality feature or sensitivity check rather than letting the model learn it naively.

## (e) Daymet V4 R1 and others
- Daymet: 1 km daily, North America, 1980–2024, with tmax, tmin, prcp and vp. **No wind, no noon RH**, so you can't build a proper FWI from it.
- It is station-interpolated (inherits station issues) and its daily convention is not noon-based.
- Not recommended, except as a terrain-aware temperature cross-check. https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-daymet-daily-v4r1-2129-4.1

## Recommendation
- **Primary: CaSR v3.2 hourly (PAVICS OPeNDAP), compute FWI ourselves at local noon.**
  - Native 10 km, matching our cells.
  - Precipitation assimilates Canadian gauges and radar, the most important input for FFMC and ignition.
  - One consistent system for 2010–2024 with no station topology.
  - Open access.
  - Costs: we must own the FWI calculation (start-up and overwintering rules), handle the CaPA 12 UTC timing, and accept that the data stops at 2024.
- **Fallback, and cross-check: CEMS `cems-fire-historical-v1`.**
  - Ready FWI indices, 1940–present, noon-local, trivial download.
  - Coarser (0.25°) with ERA5 precipitation bias, but it removes all FWI-coding risk. Covers 2025 if we ever need it.
- **Second fallback for raw weather:** ERA5-Land hourly.
- **Rejected as primary:** CWFIS rasters (2018+ / 2024+ only, station-IDW), Daymet (no wind or RH), and standalone RDPA (archive gap after Sep 2023; already inside CaSR).
- **Rules regardless of source:**
  - Features use noon-local values from t-1 and precipitation windows ending ≤ 12 UTC on t-1 (CaSR) or at noon t-1 (ERA5).
  - Treat it as a hindcast and say so.
  - Pin dataset versions (CaSR v3.2, CEMS system_version 4_1, consolidated).
