"""Build and audit report-date fire history on the published label keys."""

from datetime import date, timedelta
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from build_grid_weather import DATA, GRID, PIN_META, checked_days, expected_days, require_pin, staged, table_hash
from build_labels import EVENTS, LABELS, LABEL_YEARS, read_domain

OUTPUT = DATA / "fire_history.parquet"
CHECKS = DATA / "fire_history.checks.json"
KEYS = ["cell_id", "date"]
WINDOWS = ("30d", "365d", "5y")
SCOPES = ("cell", "3x3")
CAUSES = ("all", "human", "lightning")
CAUSE_CODE = {"unknown": 0, "human": 1, "lightning": 2}
COUNTS = tuple(f"fire_count_{scope}_{window}_{cause}"
               for window in WINDOWS for scope in SCOPES for cause in CAUSES)
RECENCY = tuple(f"days_since_last_fire_{cause}" for cause in CAUSES[1:])
FEATURES = (*COUNTS, *RECENCY)
SCHEMA = pa.schema([("cell_id", pa.int32()), ("date", pa.date32()),
                    *((name, pa.int32()) for name in COUNTS), *((name, pa.float32()) for name in RECENCY)])
NO_FIRE = -1_000_000
CAP = 1_826
SPOTS = ("ON-2010-COC6", "ON-2017-KEN16", "ON-2023-PAR6")


def epoch_day(day):
    return np.datetime64(day, "D").astype(np.int32).item()


def cutoff_days(days):
    def five_years_ago(day):
        if day.month == 2 and day.day == 29:
            return day.replace(year=day.year - 5, day=28)
        return day.replace(year=day.year - 5)

    return np.stack((days - 30, days - 365,
                     np.array([epoch_day(five_years_ago(day)) for day in days.astype('datetime64[D]').astype(object)],
                              dtype=np.int32)))


def read_events(ids):
    grid_hash = table_hash(GRID).encode()
    for path in (EVENTS, LABELS):
        meta = pq.read_schema(path).metadata
        require_pin(meta, path)
        if meta.get(b"grid_sha256") != grid_hash or not meta.get(b"deduplication"):
            raise ValueError(f"{path}: expected exact-deduped NFDB data on the published grid")
    table = pq.read_table(EVENTS, columns=["date", "cell_id", "cause", "latitude", "longitude", "nfdb_fire_id"])
    if not len(table) or any(c.null_count for c in table.columns) or table["date"].type != pa.date32():
        raise ValueError(f"{EVENTS}: empty inventory, null events, or invalid report date type")
    days = table["date"].cast(pa.int32()).to_numpy()
    cells = table["cell_id"].to_numpy()
    causes = np.array([CAUSE_CODE[c] for c in table["cause"].to_pylist()], dtype=np.intp)
    signatures = set(zip(days, table["latitude"].to_numpy(), table["longitude"].to_numpy(), causes, strict=True))
    if (len(signatures) != len(days) or (np.diff(days) < 0).any() or days[0] < epoch_day(date(2000, 1, 1))
            or cells.min() < 0 or cells.max() >= len(ids)):
        raise ValueError(f"{EVENTS}: duplicate events, unsorted dates, or events outside the source domain")
    return days, cells, causes, table["nfdb_fire_id"].to_pylist()


def neighbourhoods(ids, cells):
    neighbours = [None] * len(ids)
    for (x, y), cell in cells.items():
        neighbours[cell] = np.array([cells[p] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                      if (p := (x + dx, y + dy)) in cells], dtype=np.intp)
    return neighbours


def recency(last, day):
    return np.minimum(day - last, CAP).astype(np.float32)


def write_history(path, schema, ids, days, cutoffs, reports, fire_cells, causes, neighbours):
    """Sweep each window over the complete event stream, including the off-season."""
    n = len(ids)
    counts = np.zeros((len(WINDOWS), len(SCOPES), len(CAUSES), n), dtype=np.int32)
    latest = np.full((len(RECENCY), n), NO_FIRE, dtype=np.int32)
    open_at = 0
    expired = [0] * len(WINDOWS)
    out_ids = pa.array(ids)
    block = []

    def update(state, i, delta):
        cell, cause = fire_cells[i], causes[i]
        area = neighbours[cell]
        state[0, 0, cell] += delta
        state[1, 0, area] += delta
        if cause:
            state[0, cause, cell] += delta
            state[1, cause, area] += delta

    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for j, day in enumerate(days):
            stop = int(np.searchsorted(reports, day, side="right"))
            for i in range(open_at, stop):
                if causes[i]:
                    latest[causes[i] - 1, fire_cells[i]] = reports[i]
            for w in range(len(WINDOWS)):
                keep = int(np.searchsorted(reports, cutoffs[w, j], side="right"))
                for i in range(expired[w], min(keep, open_at)):
                    update(counts[w], i, -1)
                for i in range(max(open_at, keep), stop):
                    update(counts[w], i, 1)
                expired[w] = keep
            open_at = stop
            columns = [out_ids, pa.repeat(pa.scalar(int(day), type=pa.date32()), n)]
            columns.extend(pa.array(v) for v in counts.reshape(len(COUNTS), n).copy())
            columns.extend(pa.array(v) for v in recency(latest, day))
            block.append(pa.RecordBatch.from_arrays(columns, schema=schema))
            if len(block) == 14:
                writer.write_table(pa.Table.from_batches(block))
                block.clear()
        if block:
            writer.write_table(pa.Table.from_batches(block))


def reference_schedule(days, cutoffs, n, reports, fire_cells, causes, neighbours):
    """Paint each fire's valid output interval, independently of the rolling writer."""
    starts = np.searchsorted(days, reports, side="left")
    covered = starts < len(days)
    if (days[starts[covered]] < reports[covered]).any():
        raise ValueError("Reference interval opens before a fire's report date")

    destinations = [np.r_[c, n + neighbours[c]] for c in fire_cells]
    incidence = np.repeat(np.arange(len(reports)), [len(dst) for dst in destinations])
    scope, cell = np.divmod(np.concatenate(destinations), n)
    all_cols = scope * len(CAUSES) * n + cell
    typed = causes[incidence] > 0
    cols = np.r_[all_cols, all_cols[typed] + causes[incidence[typed]] * n]
    event = np.r_[incidence, incidence[typed]]
    rows_out, cols_out, deltas = [], [], []
    for w, lower in enumerate(cutoffs):
        # Since the lower bound is exclusive, an event expires at the first lower bound >= its report date.
        stops = np.searchsorted(lower, reports, side="left")
        use = starts[event] < stops[event]
        begin, end = starts[event[use]], stops[event[use]]
        fields = cols[use] + w * len(SCOPES) * len(CAUSES) * n
        rows_out.extend((begin, end))
        cols_out.extend((fields, fields))
        deltas.extend((np.ones(begin.size, dtype=np.int32), -np.ones(end.size, dtype=np.int32)))
    transitions = csr_matrix((np.concatenate(deltas), (np.concatenate(rows_out), np.concatenate(cols_out))),
                             shape=(len(days) + 1, len(COUNTS) * n))
    return starts, transitions


def audit_history(path, ids, days, cutoffs, reports, fire_cells, causes, neighbours):
    n = len(ids)
    starts, transitions = reference_schedule(days, cutoffs, n, reports, fire_cells, causes, neighbours)
    reference = np.zeros(len(COUNTS) * n, dtype=np.int32)
    latest = np.full((len(RECENCY), n), NO_FIRE, dtype=np.int32)
    full = checked_days(path, SCHEMA.names, n)
    keys = checked_days(LABELS, KEYS, n)
    sums = {year: np.zeros(len(FEATURES), dtype=np.float64) for year in LABEL_YEARS}
    at_cap = {year: np.zeros(len(RECENCY), dtype=np.int64) for year in LABEL_YEARS}
    rows = dict.fromkeys(LABEL_YEARS, 0)
    for j, ((d, batch), (_, labels)) in enumerate(zip(full, keys, strict=True)):
        day = days[j]
        if not all(batch.column(key).equals(labels.column(key)) for key in KEYS) or not np.array_equal(
                batch.column("cell_id").to_numpy(), ids):
            raise ValueError(f"History and label keys differ on {d}")
        start, stop = transitions.indptr[j:j + 2]
        reference[transitions.indices[start:stop]] += transitions.data[start:stop]
        lo, hi = np.searchsorted(starts, (j, j + 1))
        typed = causes[lo:hi] > 0
        np.maximum.at(latest, (causes[lo:hi][typed] - 1, fire_cells[lo:hi][typed]), reports[lo:hi][typed])
        year = d.astype(object).year
        expected_age = np.where(latest > day - CAP, day - latest, CAP)
        expected = (*reference.reshape(len(COUNTS), n), *expected_age)
        for i, (name, want) in enumerate(zip(FEATURES, expected, strict=True)):
            column = batch.column(name)
            if column.null_count:
                raise ValueError(f"Unexpected Arrow null in {name} on {d}")
            values = column.to_numpy()
            if not np.array_equal(values, want):
                raise ValueError(f"{name} on {d} disagrees with the causal per-fire reference")
            sums[year][i] += values.sum(dtype=np.float64)
            if i >= len(COUNTS):
                at_cap[year][i - len(COUNTS)] += int(np.count_nonzero(values == CAP))
        rows[year] += n
    return {year: {"rows": rows[year], "means": {
        name: float(sums[year][i] / rows[year]) for i, name in enumerate(FEATURES)
    }, "recency_at_cap_fraction": {name: float(at_cap[year][i] / rows[year])
                                   for i, name in enumerate(RECENCY)}}
            for year in LABEL_YEARS}


def hand_checks(path, reports, fire_cells, causes, fire_ids):
    results = []
    for fire_id in SPOTS:
        picks = [i for i, name in enumerate(fire_ids) if name == fire_id]
        if len(picks) != 1:
            raise ValueError(f"Hand check requires one retained {fire_id} record")
        i = picks[0]
        report_date = np.datetime64(int(reports[i]), "D").astype(object)
        cell, cause = int(fire_cells[i]), CAUSES[causes[i]] if causes[i] else "unknown"
        days = [report_date - timedelta(days=offset) for offset in (3, 2, 1, 0)]
        rows = pq.read_table(path, columns=["date", *FEATURES], partitioning=None,
                             filters=[("cell_id", "=", cell), ("date", "in", days)]).to_pylist()
        if [r["date"] for r in rows] != days:
            raise ValueError(f"Missing or out-of-order hand-check dates for {fire_id}")
        for row in rows:
            on_report = row["date"] == report_date
            for name in COUNTS:
                if row[name] != int(on_report and name.rsplit("_", 1)[1] in {"all", cause}):
                    raise ValueError(f"Hand check did not show a first appearance on the report date for {fire_id}")
            if on_report and cause != "unknown" and row[f"days_since_last_fire_{cause}"] != 0:
                raise ValueError(f"Wrong recency on the report date for {fire_id}")
            row["date"] = row["date"].isoformat()
        results.append({"nfdb_fire_id": fire_id, "cell_id": cell, "cause": cause,
                        "report_date": report_date.isoformat(), "observed_rows": rows})
    return results


def build():
    ids, cells, _ = read_domain()
    neighbours = neighbourhoods(ids, cells)
    reports, fire_cells, causes, fire_ids = read_events(ids)
    days = expected_days().astype(np.int32)
    cutoffs = cutoff_days(days)
    schema = SCHEMA.with_metadata({
        **PIN_META, b"grid_sha256": table_hash(GRID).encode(), b"events_sha256": table_hash(EVENTS).encode(),
        b"label_keys_source": str(LABELS).encode(), b"timing": b"Key date D includes reports <= D; target day is D+1",
        b"windows": b"(D minus 30 days, 365 days, or 5 calendar years, D]",
        b"neighbourhood": b"Containing tile and its eight adjacent lattice positions, retained cells only",
        b"recency": f"Days since latest human/lightning report in (D minus {CAP} days, D]; {CAP} if none".encode(),
    })
    with staged(OUTPUT) as tmp, staged(CHECKS) as checks_tmp:
        write_history(tmp, schema, ids, days, cutoffs, reports, fire_cells, causes, neighbours)
        yearly = audit_history(tmp, ids, days, cutoffs, reports, fire_cells, causes, neighbours)
        hand = hand_checks(tmp, reports, fire_cells, causes, fire_ids)
        report_months = reports.astype("datetime64[D]").astype("datetime64[M]").astype(int) % 12 + 1
        report = {
            "rows": sum(year["rows"] for year in yearly.values()), "features": list(FEATURES),
            "source_fires": len(reports),
            "pre_2010_fires": int(np.count_nonzero(reports < epoch_day(date(2010, 1, 1)))),
            "offseason_fires": int(np.count_nonzero((report_months < 5) | (report_months > 10))),
            "source_first_report_date": str(np.datetime64(int(reports[0]), "D")),
            "source_last_report_date": str(np.datetime64(int(reports[-1]), "D")),
            "first_feature_date": str(np.datetime64(int(days[0]), "D")),
            "first_5y_window_lower_bound_exclusive": str(np.datetime64(int(cutoffs[-1, 0]), "D")),
            "whole_table_causality": {"rows_checked": sum(year["rows"] for year in yearly.values()),
                                      "future_report_violations": 0,
                                      "method": "Independent per-fire output intervals cannot start before report date; every saved feature compared to reference"},
            "yearly": yearly, "hand_checked_fires": hand,
        }
        checks_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(OUTPUT)
        checks_tmp.replace(CHECKS)
    return report


if __name__ == "__main__":
    report = build()
    print(f"Audited {report['rows']:,} rows and {len(FEATURES)} causal features against {report['source_fires']:,} fires; see {CHECKS}")
