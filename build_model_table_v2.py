"""Append causal daily weather features and training-only climate normals to v1."""

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from build_fire_weather import FIELDS, _weather_cache_key, fwi
from build_grid_weather import (
    DAILY, DATA, GRID, NATIVE, PIN, WEIGHTS, YEARS, checked_days, read_weights,
    require_pin, staged, table_hash,
)
from build_model_table import OUTPUT as V1, SAMPLE_SIZE, SEED, calendar

OUTPUT = DATA / "model_table_v2.parquet"
CHECKS = OUTPUT.with_suffix(".checks.json")
CACHE = Path(".cache/fire-weather")
BASE_FIELDS = ("rain_24h", "temperature", "RH", "FFMC", "DMC", "BUI", "FWI")
DELTA_FIELDS = ("FFMC", "DMC", "BUI")
ANOMALY_FIELDS = ("FWI", "DMC", "temperature")
LOOKBACK = 30
THRESHOLD = 2.0
RAIN_WINDOWS = (3, 7, 14)
EXTREMA_WINDOWS = (3, 7)
LAGS = (1, 3)
NORMAL_RADIUS = 7
TRAIN_YEARS = range(2010, 2020)
RECENCY = "days_since_rain_ge_2mm"
NEW_FIELDS = (
    (RECENCY, pa.uint8()),
    *((f"rain_total_{n}d", pa.float64()) for n in RAIN_WINDOWS),
    *((name, pa.float64()) for n in EXTREMA_WINDOWS for name in (f"temperature_max_{n}d", f"RH_min_{n}d")),
    *((f"{field}_change_{n}d", pa.float64()) for field in DELTA_FIELDS for n in LAGS),
    *((f"{field}_anomaly_doy7", pa.float64()) for field in ANOMALY_FIELDS),
)
NEW_NAMES = tuple(name for name, _ in NEW_FIELDS)


def warmup_inputs(ids):
    columns = ("date", "cell_id", "latitude")
    with pq.ParquetFile(NATIVE) as native:
        require_pin(native.schema_arrow.metadata, NATIVE)
        if native.schema_arrow.metadata[b"FWI"].split(b";")[0] != b"xclim 0.59.1 cffwis_indices":
            raise ValueError("Unexpected native FWI configuration")
        batch = next(native.iter_batches(columns=columns, batch_size=65_536))
    dates = batch.column("date").to_numpy(zero_copy_only=False)
    first_day = np.flatnonzero(dates != np.datetime64("2010-05-01"))
    if not first_day.size or not first_day[0]:
        raise ValueError("Cannot recover native cell order from the first complete May day")
    first = batch.slice(0, int(first_day[0]))
    donors = {"id": first.column("cell_id").to_numpy(), "lat": first.column("latitude").to_numpy()}
    if (np.diff(donors["id"]) <= 0).any() or not np.isfinite(donors["lat"]).all():
        raise ValueError("Invalid native IDs or latitudes")
    edges, donor_columns, covered, _ = read_weights(WEIGHTS, ids, donors["id"])
    if not np.array_equal(ids, np.arange(len(ids))) or not covered.all():
        raise ValueError("Warm-up requires a fully covered, contiguous target grid")
    matrix = csr_matrix((edges["weight"].to_numpy(), (edges["cell_id"].to_numpy(), donor_columns)),
                        shape=(len(ids), len(donors["id"])))
    cache = CACHE / _weather_cache_key(PIN)
    if not cache.is_dir():
        # The original downloader keyed the same reduced weather by zone hash.
        cache = CACHE / f"v32-{PIN}"
    return donors, matrix, cache


def april_weather(year, dates, donors, matrix, cache):
    # The third cached block overlaps May. Verify that the reconstructed April
    # initial states and the donor order reproduce published early-May weather.
    days = dates[:42]
    chunks = []
    for start in range(0, len(days), 14):
        path = cache / f"{days[start]}_{days[start + 13]}.npy"
        values = np.load(path, allow_pickle=False)
        if values.shape != (14, len(donors["id"]), 4) or values.dtype != np.float32:
            raise ValueError(f"{path}: unexpected weather cache shape or dtype")
        chunks.append(values)
    raw = np.concatenate(chunks)
    indices = fwi(days, donors, raw)
    fields = np.concatenate((raw, indices), axis=-1)[..., [FIELDS.index(name) for name in BASE_FIELDS]]
    donor_rows = fields.transpose(1, 0, 2).reshape(len(donors["id"]), -1)
    mapped = (matrix @ donor_rows).reshape(matrix.shape[0], len(days), len(BASE_FIELDS))
    result = mapped.transpose(1, 0, 2)
    if not np.isfinite(result).all():
        raise ValueError(f"{year}: nonfinite April or early-May weather")
    return result


def day_of_year(dates):
    return (dates - dates.astype("datetime64[Y]")).astype(int) + 1


def load_weather(ids):
    donors, matrix, cache = warmup_inputs(ids)
    schema = pq.read_schema(DAILY)
    require_pin(schema.metadata, DAILY)
    if schema.metadata.get(b"weights_sha256") != table_hash(WEIGHTS).encode():
        raise ValueError("Daily weather and April mapping use different weights")
    source = checked_days(DAILY, ("cell_id", "date", *BASE_FIELDS), len(ids))
    seasons = []
    for year in YEARS:
        dates = np.arange(f"{year}-04-01", f"{year}-11-01", dtype="datetime64[D]")
        overlap = april_weather(year, dates, donors, matrix, cache)
        values = np.empty((len(BASE_FIELDS), len(dates), len(ids)), dtype=np.float64)
        values[:, :LOOKBACK] = overlap[:LOOKBACK].transpose(2, 0, 1)
        for d in range(LOOKBACK, len(dates)):
            day, batch = next(source)
            if day != dates[d] or not np.array_equal(batch.column("cell_id").to_numpy(), ids):
                raise ValueError(f"Daily grid date or cells differ on {day}")
            for i, name in enumerate(BASE_FIELDS):
                values[i, d] = batch.column(name).to_numpy(zero_copy_only=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{year}: null or nonfinite gridded source weather")
        if not np.allclose(values[:, LOOKBACK:len(overlap)], overlap[LOOKBACK:].transpose(2, 0, 1),
                           rtol=1e-6, atol=1e-7):
            raise ValueError(f"{year}: April warm-up does not reproduce published May 1–12 weather")
        seasons.append((day_of_year(dates), dict(zip(BASE_FIELDS, values, strict=True))))
    if next(source, None) is not None:
        raise ValueError("Unexpected trailing grid weather")
    return seasons, cache


def normal_totals(seasons, cells):
    sums = {field: np.zeros((367, cells)) for field in ANOMALY_FIELDS}
    counts = np.zeros(367, dtype=np.int64)
    for year, (doys, weather) in zip(YEARS, seasons, strict=True):
        if year in TRAIN_YEARS:
            counts[doys] += 1
            for field, total in sums.items():
                total[doys[0]:doys[-1] + 1] += weather[field]
    for total in sums.values():
        np.cumsum(total, axis=0, out=total)
    return sums, counts.cumsum()


def derive(year, doys, weather, totals, counts):
    start, end = LOOKBACK, len(doys) - 1  # Feature dates: May 1 through October 30.
    current = slice(start, end)
    rain = weather["rain_24h"]
    recency = np.empty(rain.shape, dtype=np.uint8)
    recency[0] = np.where(rain[0] >= THRESHOLD, 0, LOOKBACK)
    for day in range(1, end):
        recency[day] = np.where(rain[day] >= THRESHOLD, 0, np.minimum(recency[day - 1] + 1, LOOKBACK))
    out = {RECENCY: recency[current]}
    for n in RAIN_WINDOWS:
        out[f"rain_total_{n}d"] = sliding_window_view(rain, n, axis=0)[start - n + 1:end - n + 1].sum(axis=-1)
    for n in EXTREMA_WINDOWS:
        windows = slice(start - n + 1, end - n + 1)
        out[f"temperature_max_{n}d"] = sliding_window_view(weather["temperature"], n, axis=0)[windows].max(axis=-1)
        out[f"RH_min_{n}d"] = sliding_window_view(weather["RH"], n, axis=0)[windows].min(axis=-1)
    for field in DELTA_FIELDS:
        for lag in LAGS:
            out[f"{field}_change_{lag}d"] = weather[field][current] - weather[field][start - lag:end - lag]

    lower, upper = doys[current] - NORMAL_RADIUS - 1, doys[current] + NORMAL_RADIUS
    count = counts[upper] - counts[lower]
    if year in TRAIN_YEARS:
        before = np.clip(lower - doys[0] + 1, 0, len(doys))
        through = np.clip(upper - doys[0] + 1, 0, len(doys))
        count -= through - before
    if (count <= 0).any():
        raise ValueError(f"{year}: empty training climate normal")
    for field, total in totals.items():
        window_sum = total[upper] - total[lower]
        if year in TRAIN_YEARS:
            own = np.empty((len(doys) + 1, weather[field].shape[1]))
            own[0] = 0
            np.cumsum(weather[field], axis=0, out=own[1:])
            window_sum -= own[through] - own[before]
        out[f"{field}_anomaly_doy7"] = weather[field][current] - window_sum / count[:, None]
    return out


def write_table(path, ids, dates, seasons):
    cells = len(ids)
    sums, counts = normal_totals(seasons, cells)
    with pq.ParquetFile(V1) as source:
        base = source.schema_arrow
        require_pin(base.metadata, V1)
        if base.metadata.get(b"grid_sha256") != table_hash(GRID).encode():
            raise ValueError("V1 table and weather use different grids")
        schema = pa.schema([*base, *(pa.field(name, dtype, nullable=False) for name, dtype in NEW_FIELDS)],
                           metadata={**base.metadata,
                                     b"weather_lookbacks": b"Daily windows end on D and use April warm-up",
                                     b"climate_normals": b"2010-2019, DOY +/-7 within available Apr-Oct; leave own training year out"})
        batches = source.iter_batches(batch_size=8 * cells)
        offset = 0
        features = None
        steps = dates.shape[1]
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for batch in batches:
                days, partial = divmod(len(batch), cells)
                if partial or offset + days > dates.size:
                    raise ValueError("V1 does not have complete, ordered cell-days")
                start = 0
                extra = {name: [] for name in NEW_NAMES}
                keys = batch.column("cell_id").to_numpy().reshape(days, cells)
                seen = batch.column("feature_date").to_numpy(zero_copy_only=False).reshape(days, cells)
                if not ((keys == ids).all() and (seen == dates.ravel()[offset:offset + days, None]).all()):
                    raise ValueError(f"V1 date or cell mismatch at day offset {offset}")
                while start < days:
                    i, local = divmod(offset, steps)
                    doys, weather = seasons[i]
                    if local == 0:
                        features = derive(YEARS[i], doys, weather, sums, counts)
                        print(f"{YEARS[i]}: appending {len(NEW_NAMES)} weather features", flush=True)
                    span = min(days - start, steps - local)
                    for name in BASE_FIELDS:
                        original = batch.column(name).slice(start * cells, span * cells).to_numpy(zero_copy_only=False)
                        if not np.array_equal(original, weather[name][LOOKBACK + local:LOOKBACK + local + span].ravel()):
                            raise ValueError(f"V1 source {name} differs from grid weather at day offset {offset}")
                    for name, dtype in NEW_FIELDS:
                        values = features[name][local:local + span].ravel()
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite output {name} at day offset {offset}")
                        extra[name].append(pa.array(values, type=dtype))
                    start += span
                    offset += span
                writer.write_table(pa.Table.from_arrays([*batch.columns, *(pa.chunked_array(extra[n]) for n in NEW_NAMES)],
                                                        schema=schema))
        if offset != dates.size:
            raise ValueError("V1 table ended before the expected feature calendar")
    return schema


def direct_row(year_index, day, cell, seasons):
    doys, weather = seasons[year_index]
    # Slice individual source days rather than using the production prefix sums/windows.
    expected = [next((lag for lag in range(30) if weather["rain_24h"][day - lag, cell] >= 2), 30)]
    expected += [sum(weather["rain_24h"][day - n + 1:day + 1, cell]) for n in (3, 7, 14)]
    for n in (3, 7):
        expected.extend((max(weather["temperature"][day - n + 1:day + 1, cell]),
                         min(weather["RH"][day - n + 1:day + 1, cell])))
    expected += [weather[field][day, cell] - weather[field][day - n, cell]
                 for field in ("FFMC", "DMC", "BUI") for n in (1, 3)]
    normal_values = {field: [] for field in ANOMALY_FIELDS}
    for j, (reference_doys, reference) in enumerate(seasons):
        if j == year_index or YEARS[j] not in TRAIN_YEARS:
            continue
        matched = abs(reference_doys - doys[day]) <= 7
        for field in ANOMALY_FIELDS:
            normal_values[field].extend(reference[field][matched, cell])
    expected += [weather[field][day, cell] - np.mean(normal_values[field]) for field in ANOMALY_FIELDS]
    return expected


def audit(path, schema, ids, dates, seasons):
    cells = len(ids)
    steps = dates.shape[1]
    rows = dates.size * cells
    rng = np.random.default_rng(SEED)
    # Stratify the sample so the April lookbacks are explicitly exercised.
    early_count = SAMPLE_SIZE // 10
    early = rng.choice(len(YEARS) * 7 * cells, size=early_count, replace=False)
    other = rng.choice(len(YEARS) * (steps - 7) * cells, size=SAMPLE_SIZE - early_count, replace=False)
    y1, a = np.divmod(early, 7 * cells)
    y2, b = np.divmod(other, (steps - 7) * cells)
    sample = np.sort(np.concatenate((y1 * steps * cells + a, y2 * steps * cells + 7 * cells + b)))
    offset = checked = 0
    tolerance = {"rtol": 1e-10, "atol": 1e-8}
    with pq.ParquetFile(path) as file:
        if not file.schema_arrow.equals(schema, check_metadata=True) or file.metadata.num_rows != rows:
            raise ValueError("Saved schema or row count differs")
        for index in range(file.num_row_groups):
            group = file.metadata.row_group(index)
            for name in NEW_NAMES:
                stats = group.column(schema.get_field_index(name)).statistics
                if stats is None or not stats.has_null_count or stats.null_count:
                    raise ValueError(f"Cannot verify zero nulls in {name}, row group {index}")
            end = offset + group.num_rows
            left, right = np.searchsorted(sample, (offset, end))
            if left < right:
                positions = sample[left:right]
                local = positions - offset
                saved = file.read_row_group(index, columns=["feature_date", "cell_id", *NEW_NAMES]).take(local)
                year, day = np.divmod(positions // cells, steps)
                cell = positions % cells
                if not (np.array_equal(saved["cell_id"].to_numpy(), ids[cell]) and
                        np.array_equal(saved["feature_date"].to_numpy(), dates[year, day])):
                    raise ValueError("Saved sampled weather keys differ")
                expected = np.array([direct_row(y, d + LOOKBACK, c, seasons) for y, d, c in zip(year, day, cell, strict=True)])
                actual = np.column_stack([saved[name].to_numpy() for name in NEW_NAMES])
                matches = np.isclose(actual, expected, **tolerance)
                if not matches.all():
                    r, c = np.argwhere(~matches)[0]
                    raise ValueError(f"Weather recomputation differs at row {positions[r]}, column {NEW_NAMES[c]}")
                checked += len(local)
            offset = end
    if checked != SAMPLE_SIZE or offset != rows:
        raise ValueError("Audit did not cover all sampled rows")
    return {"rows": rows, "new_column_count": len(NEW_NAMES), "new_columns": NEW_NAMES,
            "new_null_counts": dict.fromkeys(NEW_NAMES, 0),
            "spot_check": {"rows_checked": checked, "may_1_to_7_rows": early_count, "columns_checked": len(NEW_NAMES),
                           "seed": SEED, "mismatches": 0, **tolerance}}


def build():
    grid = pq.read_table(GRID, columns=["cell_id"])
    require_pin(grid.schema.metadata, GRID)
    ids = grid["cell_id"].to_numpy()
    dates = calendar().reshape(len(YEARS), -1)
    seasons, cache = load_weather(ids)
    with staged(OUTPUT) as tmp, staged(CHECKS) as checks_tmp:
        schema = write_table(tmp, ids, dates, seasons)
        report = audit(tmp, schema, ids, dates, seasons)
        report.update({"source": str(V1), "weather_source": str(DAILY), "april_cache": str(cache),
                       "april_overlap_verified": "May 1–12 for every cell and year",
                       "window_end": "D inclusive", "rain_threshold_mm": THRESHOLD,
                       "rain_recency_cap_days": LOOKBACK,
                       "normals": {"years": list(TRAIN_YEARS), "radius_doy": NORMAL_RADIUS,
                                   "weather_dates": "April 1–October 31; truncate at available dates",
                                   "leave_training_year_out": True}})
        checks_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(OUTPUT)
        checks_tmp.replace(CHECKS)
    return report


if __name__ == "__main__":
    report = build()
    print(f"Audited {report['rows']:,} rows with {report['new_column_count']} new columns; see {CHECKS}")
