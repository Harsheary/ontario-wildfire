"""Score four wildfire baselines using only the 2010–2019 model-table rows."""

import json
from pathlib import Path
from statistics import mean

import numpy as np
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent / "data"
SOURCE = DATA / "model_table.parquet"
CHECKS = DATA / "baselines.checks.json"
YEARS = tuple(range(2010, 2020))
SEED = 42
FEATURES = ("FWI", "fire_count_3x3_5y_all")
TARGETS = ("target_any", "target_human", "target_lightning")


def training_plan(parquet):
    """Apply the year cutoff from the footer before any row reads."""
    seasons = [np.arange(f"{year}-05-01", f"{year}-10-31", dtype="datetime64[D]") for year in YEARS]
    dates = np.stack(seasons)
    flat_dates = dates.ravel()
    date_col = parquet.schema_arrow.get_field_index("feature_date")
    year_col = parquet.schema_arrow.get_field_index("year")
    if min(date_col, year_col) < 0:
        raise ValueError("Model table requires feature_date and year")

    offset = 0
    groups = []
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        years = group.column(year_col).statistics
        if years is None or not years.has_min_max or years.null_count != 0:
            raise ValueError(f"Row group {index} lacks year bounds")
        if years.min > YEARS[-1]:
            break
        bounds = group.column(date_col).statistics
        if bounds is None or not bounds.has_min_max or bounds.null_count != 0:
            raise ValueError(f"Row group {index} lacks date bounds")
        if (bounds.min != flat_dates[offset].astype(object)
                or (bounds.min.year, bounds.max.year) != (years.min, years.max)):
            raise ValueError(f"Row group {index} has unexpected dates or years")
        if not groups:
            if years.max != YEARS[0]:
                raise ValueError("Expected the first row group within the first training year")
            cells, remainder = divmod(group.num_rows, (bounds.max - bounds.min).days + 1)
            if not cells or remainder:
                raise ValueError("Expected complete days in the first row group")

        mixed = years.max > YEARS[-1]
        rows = (flat_dates.size - offset) * cells if mixed else group.num_rows
        days, partial = divmod(rows, cells)
        end = offset + days
        if (not rows or partial or end > flat_dates.size or (mixed and rows >= group.num_rows)
                or (not mixed and bounds.max != flat_dates[end - 1].astype(object))):
            raise ValueError(f"Invalid training prefix in row group {index}")
        groups.append((index, rows, group.num_rows))
        offset = end
        if offset == flat_dates.size:
            break
    if offset != flat_dates.size:
        raise ValueError("Model table does not contain the complete training calendar")
    return dates, cells, groups


def training_batches(parquet, dates, cells, groups, fields):
    flat_dates = dates.ravel()
    flat_years = np.repeat(YEARS, dates.shape[1])
    offset = 0
    for index, rows, _ in groups:
        # The model-table writer orders complete days, then cell_id. Stop at
        # the exact training prefix even when Parquet pages span the cutoff.
        columns = ("feature_date", "year", "cell_id", *fields)
        batch = next(parquet.iter_batches(batch_size=rows, row_groups=[index], columns=columns))
        if batch.num_rows != rows or any(column.null_count for column in batch.columns):
            raise ValueError(f"Unexpected training batch size or nulls in row group {index}")
        days = rows // cells
        span = slice(offset, offset + days)
        actual_dates = batch.column("feature_date").to_numpy(zero_copy_only=False).reshape(days, cells)
        actual_years = batch.column("year").to_numpy().reshape(days, cells)
        if not ((actual_dates == flat_dates[span, None]).all() and
                (actual_years == flat_years[span, None]).all()):
            raise ValueError(f"Unexpected training dates or years in row group {index}")
        keys = batch.column("cell_id").to_numpy().reshape(days, cells)
        if offset == 0:
            cell_ids = keys[0].copy()
            if not (cell_ids[1:] > cell_ids[:-1]).all():
                raise ValueError("Expected strictly ascending cell IDs")
        if not (keys == cell_ids).all():
            raise ValueError(f"Unexpected cell IDs in row group {index}")
        yield span, batch
        offset += days


def read_training(source):
    with pq.ParquetFile(source, pre_buffer=False) as parquet:
        dates, cells, groups = training_plan(parquet)
        fields = (*FEATURES, *TARGETS)
        panel = {name: np.empty((*dates.shape, cells), dtype=dtype)
                 for name, dtype in zip(fields, (np.float64, np.int32, np.bool_, np.bool_, np.bool_), strict=True)}
        flat = {name: values.reshape(-1, cells) for name, values in panel.items()}
        for span, batch in training_batches(parquet, dates, cells, groups, fields):
            for name, out in flat.items():
                out[span] = batch.column(name).to_numpy(zero_copy_only=False).reshape(-1, cells)
    if not np.isfinite(panel["FWI"]).all():
        raise ValueError("Nonfinite training FWI")
    return panel, groups


def top_cells(scores, k):
    cutoff = np.partition(scores, -k)[-k]
    higher = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k - higher.size]
    return np.concatenate((higher, tied))


def average_precision(scores, target):
    # Only thresholds containing a positive increase recall. Count all rows at
    # or above those thresholds, including equal-scored negatives.
    levels, positives = np.unique(scores[target], return_counts=True)
    ranks = np.searchsorted(levels, scores, side="right").ravel()
    bins = np.bincount(ranks, minlength=levels.size + 1)
    predicted = bins[:0:-1].cumsum()[::-1]
    true_positives = positives[::-1].cumsum()[::-1]
    return float(np.dot(positives, true_positives / predicted) / true_positives[0])


def metrics(scores, labels, totals, k):
    top = np.stack([top_cells(day, k) for day in scores])
    result = {f"top_1pct_catch_{name.removeprefix('target_')}":
              int(np.take_along_axis(target, top, axis=1).sum()) / totals[name]
              for name, target in labels.items()}
    ap = average_precision(scores, labels["target_any"])
    result["pr_auc"] = ap
    result["pr_auc_over_base_rate"] = ap / (totals["target_any"] / scores.size)
    return result


def summarize(by_year):
    summary = {}
    for metric in by_year[YEARS[0]]:
        values = {year: by_year[year][metric] for year in YEARS}
        low = min(values, key=values.__getitem__)
        high = max(values, key=values.__getitem__)
        summary[metric] = {"mean": mean(values.values()),
                           "min": {"year": low, "value": values[low]},
                           "max": {"year": high, "value": values[high]}}
    return summary


def climatology_loyo(target_any):
    positives_by_cell = target_any.sum(axis=1)
    years, days, _ = target_any.shape
    # A complete panel gives every cell the same denominator in each fold.
    return (positives_by_cell.sum(axis=0) - positives_by_cell) / ((years - 1) * days)


def build():
    panel, groups = read_training(SOURCE)
    _, days, cells = panel["target_any"].shape
    k = (cells + 99) // 100
    climatology = climatology_loyo(panel["target_any"])
    rng = np.random.default_rng(SEED)
    annual = {}
    baselines = {name: {} for name in (*FEATURES, "random", "climatology_loyo")}
    for i, year in enumerate(YEARS):
        labels = {name: panel[name][i] for name in TARGETS}
        totals = {name: int(target.sum()) for name, target in labels.items()}
        if not all(totals.values()):
            raise ValueError(f"Catch rate undefined for a target without positives in {year}")
        annual[year] = {"rows": days * cells, "positives": totals,
                        "base_rate": totals["target_any"] / (days * cells)}
        scores = {name: panel[name][i] for name in FEATURES}
        scores["random"] = rng.random((days, cells))
        scores["climatology_loyo"] = np.broadcast_to(climatology[i], (days, cells))
        for name, values in scores.items():
            baselines[name][year] = metrics(values, labels, totals, k)

    report = {
        "source": str(SOURCE.relative_to(DATA.parent)),
        "years": YEARS,
        "rows_read": sum(rows for _, rows, _ in groups),
        "row_groups_read": len(groups),
        "partial_row_groups": [{"row_group": index, "training_prefix_rows": rows, "total_rows": total}
                               for index, rows, total in groups if rows != total],
        "cells_per_day": cells,
        "top_cells_per_day": k,
        "random_seed": SEED,
        "top_k_ties": "cell_id ascending",
        "pr_auc_method": "Non-interpolated average precision for target_any over all cell-days in each year",
        "climatology_method": "Per-cell target_any rate in the other nine training years",
        "annual": annual,
        "baselines": {name: {"annual": by_year, "summary": summarize(by_year)}
                      for name, by_year in baselines.items()},
    }
    CHECKS.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = build()
    print(f"Scored {report['rows_read']:,} training rows; see {CHECKS.relative_to(DATA.parent)}")
