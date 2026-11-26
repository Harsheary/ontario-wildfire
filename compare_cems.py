"""Compare five native CaSR cells against the same dates at nearest CEMS grid points."""

import argparse
from calendar import monthrange
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, is_zipfile

import cdsapi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from build_fire_weather import grid, nearest_cell, session

DATASET = "cems-fire-historical-v1"
EWDS = "https://ewds.climate.copernicus.eu/api"
YEARS = (2015, 2018)
MONTHS = range(5, 11)
INDICES = {"FWI": ("fire_weather_index", "fwinx", 260540),
           "FFMC": ("fine_fuel_moisture_code", "ffmcode", 260541),
           "DC": ("drought_code", "drtcode", 260543)}
SITES = {"Red Lake": (51.03, -93.83), "Fort Severn": (55.99, -87.63),
         "Wawa": (47.99, -84.78), "Timmins": (48.48, -81.33), "North Bay": (46.31, -79.46)}


def request_for(year, months, points):
    return {"dataset_type": "consolidated_dataset", "product_type": "reanalysis", "system_version": ["4_1"],
            "data_format": "netcdf", "grid": "0.25/0.25", "variable": [x[0] for x in INDICES.values()],
            "year": [str(year)], "month": [f"{m:02}" for m in months],
            "day": [f"{d:02}" for d in range(1, monthrange(year, months[0])[1] + 1)],
            "area": [float(np.ceil(points.latitude.max())), float(np.floor(points.longitude.min())),
                     float(np.floor(points.latitude.min())), float(np.ceil(points.longitude.max()))]}


def download(points, cache):
    cache.mkdir(parents=True, exist_ok=True)
    client = None
    for year in YEARS:
        for length in (30, 31):
            months = [m for m in MONTHS if monthrange(year, m)[1] == length]
            request = request_for(year, months, points)
            area = '_'.join(str(int(value)) for value in request['area'])
            target = cache / f"{DATASET}-4_1-{year}-{length}-{area}.zip"
            if not target.exists():
                if client is None:
                    client = cdsapi.Client(url=EWDS)
                tmp = target.with_suffix(".part")
                client.retrieve(DATASET, request, str(tmp))
                tmp.replace(target)
            yield target


def read_netcdf(path, points):
    with xr.open_dataset(path, engine="netcdf4") as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        date = pd.DatetimeIndex(np.atleast_1d(ds[time_name].values)).normalize()
        grid_lon = np.asarray(ds["longitude"])
        target_lon = points.longitude.to_numpy()
        j = np.abs((grid_lon[None, :] - target_lon[:, None] + 180) % 360 - 180).argmin(axis=1)
        i = np.abs(np.asarray(ds.latitude)[None, :] - points.latitude.to_numpy()[:, None]).argmin(axis=1)
        lat = ds.latitude.values[i]
        lon = (grid_lon[j] + 180) % 360 - 180
        if (np.abs(lat - points.latitude) > 0.126).any() or (np.abs(lon - points.longitude) > 0.126).any():
            raise ValueError(f"CEMS subset does not cover the nearest 0.25° grid cell: {path}")
        selected = ds.isel(latitude=xr.DataArray(i, dims="site"), longitude=xr.DataArray(j, dims="site"))
        frame = {"date": np.repeat(date, len(points)), "cell_id": np.tile(points.cell_id, len(date)),
                 "reference_latitude": np.tile(lat, len(date)), "reference_longitude": np.tile(lon, len(date))}
        for key, (long, short, param_id) in INDICES.items():
            for name, var in selected.data_vars.items():
                if name in (long, short, key, key.lower()) or str(var.attrs.get("GRIB_paramId")) == str(param_id):
                    if time_name not in var.dims:
                        var = var.expand_dims({time_name: date})
                    extra = set(var.dims) - {time_name, "site"}
                    if any(var.sizes[d] != 1 for d in extra):
                        raise ValueError(f"Unexpected CEMS dimensions: {var.dims}")
                    frame[key] = var.squeeze(dim=list(extra), drop=True).transpose(time_name, "site").values.ravel()
                    break
        if not any(key in frame for key in INDICES):
            raise ValueError(f"No requested indices in CEMS file {path}")
        return pd.DataFrame(frame)


def reference(paths, points):
    frames = []
    for path in paths:
        if not is_zipfile(path):
            frames.append(read_netcdf(path, points))
            continue
        with ZipFile(path) as archive, tempfile.TemporaryDirectory() as temp:
            netcdf = [m for m in archive.namelist() if m.lower().endswith((".nc", ".nc4"))]
            if not netcdf:
                raise ValueError(f"No NetCDF members in {path}")
            local = Path(temp) / "subset.nc"
            for member in netcdf:
                with archive.open(member) as src, local.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                frames.append(read_netcdf(local, points))
    combined = pd.concat(frames, ignore_index=True)
    groups = combined.groupby(["date", "cell_id"], sort=False)
    if (groups.nunique() > 1).any().any():
        raise ValueError("Conflicting duplicate CEMS dates or grid points")
    return groups.first().reset_index()


def comparison(our, ref, points):
    paired = our.merge(ref, on=["date", "cell_id"], suffixes=("_casr", "_cems"), validate="one_to_one")
    result = []
    for site, point in points.iterrows():
        for year in YEARS:
            rows = paired[(paired.cell_id == point.cell_id) & (paired.date.dt.year == year)]
            if len(rows) != 184:
                raise ValueError(f"{site} {year}: expected 184 matched calendar dates, found {len(rows)}")
            for key in INDICES:
                a, b = rows[f"{key}_casr"].to_numpy(), rows[f"{key}_cems"].to_numpy()
                mask = np.isfinite(a) & np.isfinite(b)
                a, b = a[mask], b[mask]
                r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 0 and b.std() > 0 else None
                result.append({"site": site, "cell_id": int(point.cell_id), "latitude": float(point.latitude),
                               "longitude": float(point.longitude), "cems_latitude": float(rows.reference_latitude.iloc[0]),
                               "cems_longitude": float(rows.reference_longitude.iloc[0]), "year": year, "index": key,
                               "paired_days": len(a), "pearson_r": r,
                               "mean_difference_casr_minus_cems": float((a - b).mean()) if len(a) else None})
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=Path("data/ontario-fire-weather.parquet"))
    p.add_argument("--zones", type=Path, default=Path("data/ontario-fire-zones.geojson"))
    p.add_argument("--reference", type=Path, action="append", help="EWDS NetCDF or ZIP; repeat for multiple files")
    p.add_argument("--output", type=Path, default=Path("data/cems-comparison.csv"))
    args = p.parse_args()
    with session() as s:
        cells = grid(s, args.zones)
    idx = [nearest_cell(cells, lat, lon) for lat, lon in SITES.values()]
    points = pd.DataFrame({"cell_id": cells["id"][idx], "latitude": cells["lat"][idx], "longitude": cells["lon"][idx]},
                          index=list(SITES))
    if not points.cell_id.is_unique:
        raise ValueError("Comparison sites must be distinct native cells")
    filters = [[("cell_id", "in", points.cell_id.tolist()), ("date", ">=", pd.Timestamp(year, 5, 1).date()),
                ("date", "<=", pd.Timestamp(year, 10, 31).date())] for year in YEARS]
    our = pq.read_table(args.parquet, columns=["cell_id", "date", *INDICES], filters=filters).to_pandas()
    our["date"] = pd.to_datetime(our.date)
    paths = list(args.reference or download(points, Path(".cache/cems")))
    results = comparison(our, reference(paths, points), points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    args.output.with_suffix(".source.json").write_text(json.dumps({"dataset": DATASET, "api": EWDS,
        "system_version": "4_1", "product_type": "reanalysis", "dataset_type": "consolidated_dataset",
        "grid": "0.25/0.25", "doi": "10.24381/cds.0e89c522", "files": list(map(str, paths))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
