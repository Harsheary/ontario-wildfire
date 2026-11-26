"""Assemble and audit a feature-date panel with next-day NFDB targets; no modelling."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_fire_history import OUTPUT as HISTORY, SCHEMA as HISTORY_SCHEMA
from build_grid_weather import (
    DAILY as WEATHER, DATA, GRID, PANEL_SCHEMA as WEATHER_SCHEMA, PIN_META, YEARS,
    checked_days, expected_days, require_pin, staged, table_hash,
)
from build_labels import LABELS, LABEL_SCHEMA
from build_static_features import OUTPUT as STATIC

OUTPUT = DATA / "model_table.parquet"
CHECKS = DATA / "model_table.checks.json"
KEYS = ("cell_id", "date")
LABEL_FIELDS = ("fire_any", "fire_human", "fire_lightning")
TARGETS = ("target_any", "target_human", "target_lightning")
SAMPLE_SIZE = 10_000
SEED = 20260811
ONE_DAY = np.timedelta64(1, "D")
DAYS_PER_YEAR = 183


def inputs():
    grid = pq.read_table(GRID, columns=["cell_id"])
    require_pin(grid.schema.metadata, GRID)
    ids = grid["cell_id"].to_numpy()
    if grid["cell_id"].type != pa.int32() or grid["cell_id"].null_count or not len(ids) or (np.diff(ids) <= 0).any():
        raise ValueError(f"{GRID}: expected unique, ordered int32 cell IDs")
    grid_hash = table_hash(GRID).encode()
    for path, contract in ((WEATHER, WEATHER_SCHEMA), (HISTORY, HISTORY_SCHEMA), (LABELS, LABEL_SCHEMA)):
        schema = pq.read_schema(path)
        require_pin(schema.metadata, path)
        if schema != contract or (path != WEATHER and schema.metadata.get(b"grid_sha256") != grid_hash):
            raise ValueError(f"{path}: input schema or grid pin differs from the build contract")
    static = pq.read_table(STATIC)
    require_pin(static.schema.metadata, STATIC)
    if (static.schema.metadata.get(b"grid_sha256") != grid_hash or len(set(static.column_names)) != static.num_columns
            or static["cell_id"].type != pa.int32() or any(c.null_count for c in static.columns)):
        raise ValueError(f"{STATIC}: grid pin differs, or duplicate fields, invalid keys or nulls present")
    static = static.sort_by("cell_id").combine_chunks()
    if not np.array_equal(static["cell_id"].to_numpy(), ids):
        raise ValueError(f"{STATIC}: missing, duplicate or unexpected cell IDs")
    return ids, static.drop(["cell_id"]), grid_hash


def calendar():
    source = expected_days().reshape(len(YEARS), -1)
    current, following = source[:, :-1], source[:, 1:]
    if (current.shape[1] != DAYS_PER_YEAR or not (following - current == ONE_DAY).all()
            or not np.array_equal(current.astype("datetime64[Y]"), following.astype("datetime64[Y]"))):
        raise ValueError("Expected consecutive same-season dates and 183 feature days per year")
    return current.ravel()


def output_schema(static, grid_hash):
    schema = pa.schema([
        ("cell_id", pa.int32()), ("feature_date", pa.date32()),
        ("year", pa.int16()), ("doy", pa.int16()), ("split", pa.string()),
        *list(WEATHER_SCHEMA)[2:], *list(HISTORY_SCHEMA)[2:], *list(static.schema),
        *((name, pa.bool_()) for name in TARGETS),
    ], metadata={**PIN_META, b"grid_sha256": grid_hash,
                 b"target_alignment": b"Targets at D+1; weather and fire history at feature_date D",
                 b"feature_season": b"May 1 through October 30, 2010-2024",
                 b"split": b"By feature year: train 2010-2019, holdout 2020-2024"})
    if len(set(schema.names)) != len(schema.names):
        raise ValueError("Input feature names collide with another output column")
    return schema


def validate_day(path, day, batch, ids):
    if any(c.null_count for c in batch.columns):
        raise ValueError(f"{path}: null values on {day}")
    if not np.array_equal(batch.column("cell_id").to_numpy(), ids):
        raise ValueError(f"{path}: missing, duplicate, unexpected or out-of-order cell IDs on {day}")


def assemble(path, schema, ids, static, days):
    weather = checked_days(WEATHER, WEATHER_SCHEMA.names, len(ids))
    history = checked_days(HISTORY, HISTORY_SCHEMA.names, len(ids))
    labels = checked_days(LABELS, [*KEYS, *LABEL_FIELDS], len(ids))
    static_arrays = [c.chunk(0) for c in static.columns]
    prev = None
    count = 0
    block = []
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for (d, w), (history_day, h), (label_day, label) in zip(weather, history, labels, strict=True):
            if d != history_day or d != label_day:
                raise ValueError(f"Input dates differ: weather {d}, history {history_day}, labels {label_day}")
            for source, batch in ((WEATHER, w), (HISTORY, h), (LABELS, label)):
                validate_day(source, d, batch, ids)
            if prev is not None:
                feature_day, w0, h0 = prev
                if feature_day.astype("datetime64[Y]") == d.astype("datetime64[Y]"):
                    if d - feature_day != ONE_DAY or count >= len(days) or days[count] != feature_day:
                        raise ValueError(f"Invalid feature/target pairing: {feature_day} and {d}")
                    feature_date = feature_day.astype(object)
                    year = feature_date.year
                    doy = feature_date.timetuple().tm_yday
                    columns = [w0.column("cell_id"), w0.column("date"),
                               pa.repeat(pa.scalar(year, type=pa.int16()), len(ids)),
                               pa.repeat(pa.scalar(doy, type=pa.int16()), len(ids)),
                               pa.repeat(pa.scalar("train" if year < 2020 else "holdout"), len(ids)),
                               *w0.columns[2:], *h0.columns[2:], *static_arrays, *label.columns[2:]]
                    block.append(pa.RecordBatch.from_arrays(columns, schema=schema))
                    count += 1
                    if len(block) == 8:
                        writer.write_table(pa.Table.from_batches(block))
                        block.clear()
            prev = d, w, h
        if block:
            writer.write_table(pa.Table.from_batches(block))
    if count != len(days):
        raise ValueError(f"Unexpected joined day count: {count}, expected {len(days)}")


def audit(path, schema, ids, days):
    n = len(ids)
    total = n * len(days)
    size = len(YEARS)
    positives = np.zeros((size, len(TARGETS)), dtype=np.int64)
    yearly_rows = np.zeros(size, dtype=np.int64)
    years = days.astype("datetime64[Y]")
    doys = (days - years).astype(np.int16) + 1
    years = years.astype(np.int16) + 1970
    sample = np.sort(np.random.default_rng(SEED).choice(total, SAMPLE_SIZE, replace=False))
    sampled_labels = {}
    offset = 0
    with pq.ParquetFile(path) as file:
        if file.schema_arrow != schema or file.metadata.num_rows != total:
            raise ValueError("Saved table schema or total row count differs from expected")
        nulls = dict.fromkeys(schema.names, 0)
        for group in range(file.num_row_groups):
            row_group = file.metadata.row_group(group)
            for i, name in enumerate(schema.names):
                stats = row_group.column(i).statistics
                if stats is None or not stats.has_null_count or stats.null_count:
                    raise ValueError(f"Cannot verify zero nulls in {name}, row group {group}")
        for batch in file.iter_batches(columns=["cell_id", "feature_date", "year", "doy", "split", *TARGETS]):
            end = offset + len(batch)
            day_index, cell_index = np.divmod(np.arange(offset, end), n)
            observed_cells = batch.column("cell_id").to_numpy()
            observed_dates = batch.column("feature_date").to_numpy(zero_copy_only=False)
            observed_year = batch.column("year").to_numpy()
            if (not np.array_equal(observed_cells, ids[cell_index]) or not np.array_equal(observed_dates, days[day_index])
                    or not np.array_equal(observed_year, years[day_index])
                    or not np.array_equal(batch.column("doy").to_numpy(), doys[day_index])
                    or not pc.all(pc.equal(batch.column("split"), pc.if_else(
                        pc.less(batch.column("year"), 2020), "train", "holdout"))).as_py()):
                raise ValueError(f"Saved keys, calendar or split disagree starting at row {offset}")
            year_index = observed_year - YEARS.start
            yearly_rows += np.bincount(year_index, minlength=size)
            targets = [batch.column(name).to_numpy(zero_copy_only=False) for name in TARGETS]
            for i, target in enumerate(targets):
                positives[:, i] += np.bincount(year_index[target], minlength=size)
            a, b = np.searchsorted(sample, (offset, end))
            for i in sample[a:b] - offset:
                target_day = observed_dates[i] + ONE_DAY
                sampled_labels.setdefault(target_day, []).append((observed_cells[i], tuple(t[i] for t in targets)))
            offset = end
    if offset != total or not (yearly_rows == n * DAYS_PER_YEAR).all():
        raise ValueError("Saved rows do not cover the expected yearly panel")

    # Join saved samples independently to the original label date and cell keys.
    checked = 0
    for day, batch in checked_days(LABELS, [*KEYS, *LABEL_FIELDS], n):
        rows = sampled_labels.get(day)
        if not rows:
            continue
        cell_ids, target_values = zip(*rows, strict=True)
        label_ids = batch.column("cell_id").to_numpy()
        index = np.searchsorted(label_ids, cell_ids)
        if (index >= len(label_ids)).any() or not np.array_equal(label_ids[index], cell_ids):
            raise ValueError(f"Sampled cell missing from labels on {day}")
        original = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False)[index] for name in LABEL_FIELDS])
        if not np.array_equal(original, target_values):
            raise ValueError(f"Saved targets differ from D+1 labels on {day}")
        checked += len(rows)
    if checked != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} matched sample rows, got {checked}")
    return {"rows": total, "cells": n, "feature_days_per_year": DAYS_PER_YEAR, "columns": schema.names,
            "null_counts": nulls,
            "label_spot_check": {"rows_checked": checked, "seed": SEED, "mismatches": 0, "day_offset": 1},
            "yearly": {year: {"rows": int(yearly_rows[i]), "target_any_positives": int(positives[i, 0]),
                              "target_any_positive_rate": float(positives[i, 0] / yearly_rows[i])}
                       for i, year in enumerate(YEARS)}}


def build():
    ids, static, grid_hash = inputs()
    days = calendar()
    schema = output_schema(static, grid_hash)
    with staged(OUTPUT) as tmp, staged(CHECKS) as checks_tmp:
        assemble(tmp, schema, ids, static, days)
        report = audit(tmp, schema, ids, days)
        checks_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(OUTPUT)
        checks_tmp.replace(CHECKS)
    return report


if __name__ == "__main__":
    report = build()
    print(f"Audited {report['rows']:,} rows and {len(report['columns'])} columns; see {CHECKS}")
