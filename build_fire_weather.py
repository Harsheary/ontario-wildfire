"""Build the Ontario CaSR v3.2 daily fire-weather table, without shifting dates."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
import requests
from requests.adapters import HTTPAdapter
import shapely
from shapely.geometry import shape
from urllib3.util.retry import Retry
import xarray as xr
import xclim
from xclim.indices.fire import cffwis_indices

THREDDS = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds"
CATALOG = f"{THREDDS}/catalog/datasets/reanalyses/catalog.xml"
CASR = f"{THREDDS}/dodsC/datasets/reanalyses/1hr_NAM_GovCan_CaSR_v32_196801-202412.ncml"
ZONE_URL = "https://ws.lioservices.lrc.gov.on.ca/arcgis2/rest/services/LIO_OPEN_DATA/LIO_Open09/MapServer/34/query"
ZONE_PARAMS = {"where": "1=1", "outFields": "OFFICIAL_NAME", "outSR": 4326, "f": "geojson"}
REGULATION = "https://www.ontario.ca/laws/regulation/960207"
TIME_ORIGIN = "1968-01-01"
EPOCH = np.datetime64(TIME_ORIGIN, "h")
NOON_RULE = (-90, 16, 17)  # Longitude split, UTC hours at/east and west.
RAIN_WINDOW = (-11, 12)  # Inclusive hours relative to midnight on D.
RAIN_OFFSETS = np.arange(RAIN_WINDOW[0], RAIN_WINDOW[1] + 1)
WEATHER = {"hurs": ("%", "P_HR_1.5m"), "sfcWind": ("m s-1", "P_UVC_10m"), "tas": ("K", "P_TT_1.5m")}
SOURCE = {**WEATHER, "pr": ("kg m-2 s-1", "A_PR0_SFC")}
KEY_FIELDS = ("cell_id", "date", "latitude", "longitude", "noon_utc_hour")
FIELDS = ("temperature", "RH", "wind", "rain_24h", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI")
YEARS = range(2010, 2025)
DAYS_PER_YEAR = 184
MAX_MISSING_SHARE = 0.001
SPOTS = ("2010-05-01", "2015-07-01", "2018-10-31")


def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2,
             status_forcelist=[429, 500, 502, 503, 504])))
    return s


def fetch(s, url, **kwargs):
    response = s.get(url, timeout=(15, 180), **kwargs)
    response.raise_for_status()
    return response


def dap(s, projections):
    """Read THREDDS DAP2 projected numeric arrays without implicit coordinate reads."""
    names = list(projections)
    query = ",".join(projections.values())
    header, payload = fetch(s, f"{CASR}.dods?{query}").content.split(b"\nData:\n", 1)
    arrays = re.findall(rb"\b(Int32|Float32) (\w+)((?:\[\w+ = \d+\])+);", header)
    if [name.decode() for _, name, _ in arrays] != names:
        raise ValueError(f"Unexpected DAP2 projection: {header!r}")
    result, offset = {}, 0
    for dtype, name, axes in arrays:
        dims = tuple(map(int, re.findall(rb" = (\d+)", axes)))
        count = int(np.prod(dims))
        if struct.unpack_from(">II", payload, offset) != (count, count):
            raise ValueError(f"Unexpected DAP2 array size for {name!r}")
        result[name.decode()] = np.frombuffer(payload, dtype=">i4" if dtype == b"Int32" else ">f4",
                                              count=count, offset=offset + 8).reshape(dims)
        offset += 8 + 4 * count
    if offset != len(payload):
        raise ValueError("Unexpected trailing DAP2 data")
    return result


def check_metadata(s):
    das = fetch(s, f"{CASR}.das").text
    attrs = {name: dict(re.findall(r'^[ \t]+String (\w+) "([^"\n]*)";$', body, re.M))
             for name, body in re.findall(r"^    (\w+) \{(.*?)^    \}", das, re.M | re.S)}
    if attrs.get("time", {}).get("units") != f"hours since {TIME_ORIGIN}":
        raise ValueError("Source time coordinate must use hours since " + TIME_ORIGIN)
    dds = fetch(s, f"{CASR}.dds?{','.join(SOURCE)}").text
    grids = {name: body for body, name in re.findall(r"^    Grid \{(.*?)^    \} (\w+);", dds, re.M | re.S)}
    for name, (units, original) in SOURCE.items():
        methods = "time: mean (interval: 1 hour)" if name == "pr" else "time: point"
        expected = {"units": units, "original_variable": original, "cell_methods": methods}
        actual = attrs.get(name, {})
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"Source {name} {key}: expected {value!r}, got {actual.get(key)!r}")
        # DAS declares time units on the shared coordinate; DDS binds each field to it.
        array, _, maps = grids.get(name, "").partition("MAPS:")
        axis = re.search(rf"^[ \t]*Float32 {name}\[time = (\d+)\]", array, re.M)
        time = re.search(r"^[ \t]*Int32 time\[time = (\d+)\];$", maps, re.M)
        if not axis or not time or axis[1] != time[1]:
            raise ValueError(f"Source {name} must use the shared hourly time coordinate")


def grid(s, zone_path):
    if not zone_path.exists():
        zone_path.parent.mkdir(parents=True, exist_ok=True)
        zone_path.write_bytes(fetch(s, ZONE_URL, params=ZONE_PARAMS).content)
    raw = zone_path.read_bytes()
    features = json.loads(raw)["features"]
    if sorted(int(f["properties"]["OFFICIAL_NAME"]) for f in features) != list(range(1, 37)):
        raise ValueError("Ontario fire region must contain zones 1 through 36")
    to_metres = Transformer.from_crs(4326, 3161, always_xy=True).transform
    zones = [shapely.make_valid(shapely.transform(shape(f["geometry"]), to_metres, interleaved=False)) for f in features]
    region = shapely.union_all(zones)
    buffered = region.buffer(20_000, quad_segs=32)
    native = dap(s, {"lat": "lat.lat", "lon": "lon.lon"})
    lat, lon = native["lat"], (native["lon"].astype("float64") + 180) % 360 - 180
    x, y = to_metres(lon, lat)
    row, col = np.where(shapely.intersects_xy(buffered, x, y))
    boundary, east, west = NOON_RULE
    return {
        "row": row, "col": col, "id": (row * lat.shape[1] + col).astype("int32"),
        "lat": lat[row, col].astype("float32"), "lon": lon[row, col].astype("float32"),
        "noon": np.where(lon[row, col] >= boundary, east, west).astype("uint8"),
        "inside": shapely.intersects_xy(region, x[row, col], y[row, col]),
        "zone_sha256": hashlib.sha256(raw).hexdigest(),
    }


def hours(times):
    return (times.astype("datetime64[h]") - EPOCH).astype("int64")


def rain_hours(days):
    return hours(days)[:, None] + RAIN_OFFSETS


def read_block(s, times, variables, row, col):
    stride = times[1] - times[0] if len(times) > 1 else 1
    if np.any(np.diff(times) != stride):
        raise ValueError("DAP subset needs regularly spaced hours")
    yr, xr = int(row.min()), int(col.min())
    t = f"[{times[0]}:{stride}:{times[-1]}]"
    window = f"[{yr}:{row.max()}][{xr}:{col.max()}]"
    result = dap(s, {"time": f"time{t}", **{v: f"{v}.{v}{t}{window}" for v in variables}})
    if not np.array_equal(result.pop("time"), times):
        raise ValueError("Source valid times differ from requested hours")
    return {v: a[:, row - yr, col - xr] for v, a in result.items()}


def rain_total(flux):
    # All 24 intervals must exist; plain sum propagates missing hourly values.
    return (flux.sum(axis=1, dtype="float64") * 3600).astype("float32")


def daily_weather(s, days, cells):
    row, col = cells["row"], cells["col"]
    temps = np.empty((len(days), len(row), 3), dtype="float32")
    for utc in NOON_RULE[1:]:
        select = cells["noon"] == utc
        if not select.any():
            continue
        noon = read_block(s, hours(days) + utc, WEATHER, row[select], col[select])
        temps[:, select, 0] = noon["tas"] - 273.15
        temps[:, select, 1] = noon["hurs"]
        temps[:, select, 2] = noon["sfcWind"] * 3.6
    t = rain_hours(days)
    flux = read_block(s, t.ravel(), ("pr",), row, col)["pr"].reshape(len(days), len(RAIN_OFFSETS), len(row))
    return np.concatenate((temps, rain_total(flux)[..., None]), axis=-1)


def cached_weather(days, cells, cache):
    path = cache / f"{days[0]}_{days[-1]}.npy"
    if path.exists():
        return np.load(path, allow_pickle=False)
    with session() as s:
        result = daily_weather(s, days, cells)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f, result, allow_pickle=False)
    tmp.replace(path)
    return result


def fwi(days, cells, weather):
    coords = {"time": days, "cell": cells["id"]}
    t, rh, wind, rain = [xr.DataArray(weather[..., i], coords=coords, dims=("time", "cell"), attrs={"units": unit})
                         for i, unit in enumerate(("degC", "%", "km/h", "mm/day"))]
    lat = xr.DataArray(cells["lat"], coords={"cell": cells["id"]}, dims="cell", attrs={"units": "degrees_north"})
    # A missing driver invalidates the whole recurrence until the next April.
    complete = np.isfinite(weather).all(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = cffwis_indices(t, rain.clip(min=0), wind.clip(min=0), rh.clip(0, 100), lat,
                                ffmc_start=85, dmc_start=6, dc_start=15, season_method=None, overwintering=False)
    valid = np.logical_and.accumulate(complete, axis=0)
    return np.stack([np.where(valid, getattr(scores, k).transpose("time", "cell"), np.nan)
                     for k in FIELDS[4:]], axis=-1)


def table(days, cells, weather, scores, metadata):
    n = len(cells["id"])
    days_since_epoch = days.astype("datetime64[D]").astype("int32")
    columns = {"cell_id": np.tile(cells["id"], len(days)),
               "date": pa.array(np.repeat(days_since_epoch, n)).cast(pa.date32()),
               "latitude": np.tile(cells["lat"], len(days)), "longitude": np.tile(cells["lon"], len(days)),
               "noon_utc_hour": np.tile(cells["noon"], len(days))}
    for data, fields in ((weather, FIELDS[:4]), (scores, FIELDS[4:])):
        columns.update((key, pa.array(data[..., i].ravel(), mask=~np.isfinite(data[..., i].ravel())))
                       for i, key in enumerate(fields))
    return pa.table(columns, metadata=metadata)


def nearest_cell(cells, lat, lon):
    distance = (cells["lat"] - lat)**2 + ((cells["lon"] - lon) * np.cos(np.deg2rad(lat)))**2
    return int(np.argmin(np.where(cells["inside"], distance, np.inf)))


def rain_audit(s, day, cells, cell, stored):
    date = np.datetime64(day, "D")
    previous = date - np.timedelta64(1, "D")
    timestamps = [previous + np.timedelta64(h, "h") for h in range(13, 24)]
    timestamps += [date + np.timedelta64(h, "h") for h in range(13)]
    first = np.datetime64(f"{previous}T13", "h")
    last = np.datetime64(f"{date}T12", "h")
    if len(timestamps) != 24 or timestamps[0] != first or timestamps[-1] != last:
        raise ValueError(f"Invalid rain audit window for {day}: expected {first} through {last}")
    times = hours(np.array([*timestamps, date + np.timedelta64(13, "h")]))
    ref = read_block(s, times, ("pr",), cells["row"][[cell]], cells["col"][[cell]])["pr"][:, 0]
    expected = sum(float(hourly_flux) * 3600 for hourly_flux in ref[:-1])
    if not np.isfinite(expected) or not np.isclose(expected, stored, rtol=1e-6):
        raise ValueError(f"Rain spot check failed for {day}, cell {cells['id'][cell]}")
    return {"cell_id": int(cells["id"][cell]), "date": str(day), "hour_count": 24,
            "first_hour_utc": str(first), "last_hour_utc": str(last), "stored_mm": float(stored),
            "independent_sum_mm": expected, "excluded_13_utc_mm": float(ref[-1]) * 3600}


def _weather_cache_key(zone_sha256):
    inputs = (zone_sha256, CASR, tuple(SOURCE), RAIN_WINDOW, NOON_RULE)
    return "v32-" + hashlib.sha256(json.dumps(inputs).encode()).hexdigest()


def build(zone_path, output, cache, workers):
    with session() as s:
        check_metadata(s)
        cells = grid(s, zone_path)
    cache = cache / _weather_cache_key(cells["zone_sha256"])
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    report = {"cells": len(cells["id"]), "rows_by_year": {}, "rain_audits": []}
    if not report["cells"]:
        raise ValueError("No native cells in the Ontario fire-weather grid")
    annual_rows = DAYS_PER_YEAR * report["cells"]
    total = annual_rows * len(YEARS)
    missing = dict.fromkeys(FIELDS, 0)
    metadata = {"CaSR_version": "3.2", "hourly_url": CASR, "THREDDS_catalog": CATALOG,
                "hourly_variables_and_units": json.dumps(SOURCE), "zone_query": requests.Request(
                    "GET", ZONE_URL, params=ZONE_PARAMS).prepare().url,
                "zone_sha256": cells["zone_sha256"], "region_definition": REGULATION,
                "spatial_selection": "Native cell centre inside union of zones 1–36 buffered 20 km in EPSG:3161",
                "cell_id": "zero-based rlat_index * 706 + rlon_index",
                "units": "temperature: degC at 1.5 m; RH: % at 1.5 m; wind: km/h at 10 m; rain_24h: mm water equivalent",
                "timing": "date D unchanged; noon_utc_hour=16 at/east of -90 degrees, else 17; rain hours D-1 13 UTC through D 12 UTC inclusive",
                "FWI": f"xclim {xclim.__version__} cffwis_indices; FFMC/DMC/DC 85/6/15 before April 1 update; no overwintering",
                "FWI_missing": "Missing driver invalidates all indices for that cell from that day through Oct 31; no imputation"}
    print(f"{report['cells']} cells; {np.count_nonzero(cells['noon'] == 17)} use 17 UTC", flush=True)
    writer = None
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor, ExitStack() as resources:
            for year in YEARS:
                days = np.arange(f"{year}-04-01", f"{year}-11-01", dtype="datetime64[D]")
                # Bound DAP response size; cache reduced weather only, not the hourly cube.
                offsets = range(0, len(days), 14)
                futures = {executor.submit(cached_weather, days[i:i + 14], cells, cache): i for i in offsets}
                weather = np.empty((len(days), len(cells["id"]), 4), dtype="float32")
                for future in as_completed(futures):
                    i = futures[future]
                    weather[i:i + 14] = future.result()
                scores = fwi(days, cells, weather)
                current = table(days[30:], cells, weather[30:], scores[30:], metadata)
                if len(current) != annual_rows:
                    raise ValueError(f"{year}: expected {annual_rows} rows, got {len(current)}")
                for k in KEY_FIELDS:
                    if current[k].null_count:
                        raise ValueError(f"{year}: null values in key column {k}")
                for k in FIELDS:
                    missing[k] += current[k].null_count
                    # Future years cannot reduce the missing count or increase the fixed row budget.
                    if missing[k] / total > MAX_MISSING_SHARE:
                        raise ValueError(f"{year}: {k} missing share exceeds {MAX_MISSING_SHARE:.1%}")
                with session() as s:
                    for day, lat, lon in zip(SPOTS, (50, 48, 51), (-93, -81, -87), strict=True):
                        if day.startswith(str(year)):
                            cell = nearest_cell(cells, lat, lon)
                            report["rain_audits"].append(rain_audit(s, day, cells, cell, weather[days == np.datetime64(day), cell, 3].item()))
                if writer is None:
                    writer = resources.enter_context(pq.ParquetWriter(tmp, current.schema, compression="zstd"))
                writer.write_table(current)
                report["rows_by_year"][year] = len(current)
                print(f"{year}: {len(current)} rows", flush=True)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    report["missing_share"] = {k: count / total for k, count in missing.items()}
    report["missing_share"].update(dict.fromkeys(KEY_FIELDS, 0))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", type=Path, default=Path("data/ontario-fire-zones.geojson"))
    parser.add_argument("--output", type=Path, default=Path("data/ontario-fire-weather.parquet"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/fire-weather"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = build(args.zones, args.output, args.cache, args.workers)
    args.output.with_suffix(".checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
