"""Build Ontario NFDB report-date labels on the published 10 km grid."""

from collections import Counter, defaultdict
from datetime import date, timedelta
from io import TextIOWrapper
import csv
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS, Geod, Transformer
import requests
import shapely

from build_grid_weather import (
    DATA, GRID, GRID_SCHEMA, CRS_GRID, TILE, PIN_META, checked_days,
    expected_days, require_pin, square_at, staged, table_hash,
)

RELEASE = "20260811"
MEMBER = f"NFDB_point_{RELEASE}.txt"
URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_txt.zip"
SHA256 = "9eb4773dfc60cfcfe7d76efa0f2470cdd341f4967fbec8084041f3fab87f8be3"
ARCHIVE = Path(".cache/nfdb/NFDB_point_txt.zip")
LABELS = DATA / "labels.parquet"
EVENTS = DATA / "ontario_nfdb_fires.parquet"
CHECKS = DATA / "labels.checks.json"
CAUSES = {"H": "human", "N": "lightning", "U": "unknown"}
YEARS = range(2000, 2025)
LABEL_YEARS = range(2010, 2025)
ONE_DAY = timedelta(days=1)
PROJECT = Transformer.from_crs(4326, CRS_GRID, always_xy=True)
GEODESIC = Geod(ellps="WGS84")
NEAR_METRES = 100
EVENT_SCHEMA = pa.schema([
    ("source_line", pa.int32()), ("nfdb_fire_id", pa.string()), ("cell_id", pa.int32()),
    ("date", pa.date32()), ("cause", pa.string()), ("fire_type", pa.string()),
    ("longitude", pa.float64()), ("latitude", pa.float64()),
    ("fire_id", pa.string()), ("merged_fire_ids", pa.list_(pa.string())), ("near_duplicate", pa.bool_()),
])
LABEL_SCHEMA = pa.schema([
    ("cell_id", pa.int32()), ("date", pa.date32()), ("fire_any", pa.bool_()),
    ("fire_human", pa.bool_()), ("fire_lightning", pa.bool_()), ("n_fires", pa.uint32()),
    pa.field("fire_any_pm1", pa.bool_(), metadata={b"role": b"check_only_never_train"}),
])


def pinned_archive():
    def check(path):
        if table_hash(path) != SHA256:
            raise ValueError(f"Expected NFDB {RELEASE}, SHA-256 {SHA256}, at {ARCHIVE}; no version drift allowed")

    if ARCHIVE.exists():
        check(ARCHIVE)
    else:
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with staged(ARCHIVE) as tmp:
            with requests.get(URL, stream=True, timeout=(15, 120)) as response, tmp.open("wb") as output:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1 << 20):
                    output.write(chunk)
            check(tmp)
            tmp.replace(ARCHIVE)
    return ARCHIVE


def read_domain():
    grid = pq.read_table(GRID)
    require_pin(grid.schema.metadata, GRID)
    geo = json.loads(grid.schema.metadata[b"geo"])
    if (grid.schema != GRID_SCHEMA or any(col.null_count for col in grid.columns) or not len(grid)
            or geo["primary_column"] != "geometry" or geo["columns"]["geometry"]["encoding"] != "WKB"
            or CRS.from_json_dict(geo["columns"]["geometry"]["crs"]) != CRS_GRID):
        raise ValueError(f"{GRID}: expected the published EPSG:3978 square grid")
    ids, x, y = (grid[name].to_numpy() for name in ("cell_id", "centre_x", "centre_y"))
    if (not np.array_equal(ids, np.arange(len(ids))) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not (x % TILE == TILE / 2).all() or not (y % TILE == TILE / 2).all()):
        raise ValueError(f"{GRID}: invalid grid IDs or lattice centres")
    cells = {(math.floor(a / TILE), math.floor(b / TILE)): int(i) for i, a, b in zip(ids, x, y, strict=True)}
    if len(cells) != len(ids):
        raise ValueError(f"{GRID}: duplicate square locations")
    squares = shapely.from_wkb(grid["geometry"].to_numpy())
    if not shapely.equals(squares, square_at(x, y)).all():
        raise ValueError(f"{GRID}: geometry disagrees with lattice centres")
    return ids, cells, squares


def ontario_rows(archive):
    with ZipFile(archive) as bundle, bundle.open(MEMBER) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        for row in reader:
            if row["SRC_AGENCY"] == "ON" and int(row["YEAR"]) in YEARS:
                yield reader.line_num, row


def prescribed(row):
    return (row["FIRE_TYPE"].strip() == "PB" or row["PRESCRIBED"].strip() in {"PB", "1", "2", "3", "4"}
            or row["CAUSE"].strip() == "H-PB")


def read_events(archive, cells):
    stats = {year: dict.fromkeys(("ontario_points", "prescribed_dropped", "non_prescribed_points",
                                 "outside_grid", "missing_report_date", "dated_in_grid",
                                 "exact_duplicate_rows_removed", "retained_dated_in_grid",
                                 "label_season_reports_before_dedup", "label_season_fires"), 0)
             for year in YEARS}
    events, outside, undated = [], [], []
    for line, row in ontario_rows(archive):
        year = int(row["YEAR"])
        yearly = stats[year]
        yearly["ontario_points"] += 1
        if prescribed(row):
            yearly["prescribed_dropped"] += 1
            continue
        yearly["non_prescribed_points"] += 1
        month, day = int(row["MONTH"]), int(row["DAY"])
        reported = row["REP_DATE"].strip()
        report_date = None if (month, day) == (0, 0) and not reported else date(year, month, day)
        if report_date is not None and reported.split(" ", 1)[0] != report_date.isoformat():
            raise ValueError(f"{MEMBER}:{line}: YEAR/MONTH/DAY disagrees with REP_DATE")
        lon, lat = float(row["LONGITUDE"]), float(row["LATITUDE"])
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError(f"{MEMBER}:{line}: invalid geographic coordinates")
        x, y = PROJECT.transform(lon, lat, errcheck=True)
        cell_id = cells.get((math.floor(x / TILE), math.floor(y / TILE)))
        event = {"source_line": line, "nfdb_fire_id": row["NFDBFIREID"], "fire_id": row["FIRE_ID"].strip(),
                 "cell_id": cell_id,
                 "date": report_date, "cause": CAUSES[row["CAUSE"].strip()], "fire_type": row["FIRE_TYPE"].strip(),
                 "longitude": lon, "latitude": lat}
        if cell_id is None:
            yearly["outside_grid"] += 1
            outside.append({**event, "year": year, "date": report_date.isoformat() if report_date else None,
                            "x_3978": x, "y_3978": y})
        if report_date is None:
            yearly["missing_report_date"] += 1
            undated.append({**event, "year": year, "source_month": month, "source_day": day})
        elif cell_id is not None:
            yearly["dated_in_grid"] += 1
            if year in LABEL_YEARS and 5 <= report_date.month <= 10:
                yearly["label_season_reports_before_dedup"] += 1
            events.append(event)
    return sorted(events, key=lambda e: (e["date"], e["cell_id"], e["source_line"])), stats, outside, undated


def fire_id_order(event):
    fire_id = event["fire_id"]
    suffix = re.fullmatch(r"(.*?)([0-9]+)", fire_id)
    prefix, number = (suffix[1], int(suffix[2])) if suffix else (fire_id, -1)
    return prefix, number, fire_id, event["source_line"]


def merge_exact_records(events, stats):
    groups = defaultdict(list)
    for e in events:
        groups[e["date"], e["latitude"], e["longitude"], e["cause"]].append(e)
    retained = []
    for members in groups.values():
        members.sort(key=fire_id_order)
        first = members[0]
        day = first["date"]
        yearly = stats[day.year]
        yearly["exact_duplicate_rows_removed"] += len(members) - 1
        yearly["retained_dated_in_grid"] += 1
        if day.year in LABEL_YEARS and 5 <= day.month <= 10:
            yearly["label_season_fires"] += 1
        retained.append({**first, "merged_fire_ids": [e["fire_id"] for e in members], "near_duplicate": False})
    return sorted(retained, key=lambda e: (e["date"], e["cell_id"], e["source_line"]))


def flag_near_duplicates(events):
    days = defaultdict(list)
    for event in events:
        days[event["date"]].append(event)
    pairs = 0
    for day in days.values():
        if len(day) < 2:
            continue
        lon = np.array([e["longitude"] for e in day])
        lat = np.array([e["latitude"] for e in day])
        a, b = np.triu_indices(len(day), k=1)
        _, _, distance = GEODESIC.inv(lon[a], lat[a], lon[b], lat[b])
        hits = (distance > 0) & (distance <= NEAR_METRES)
        pairs += int(hits.sum())
        for i in np.union1d(a[hits], b[hits]):
            day[i]["near_duplicate"] = True
    return pairs


def event_index(events):
    days = defaultdict(dict)
    for event in events:
        if event["date"].year in LABEL_YEARS:
            counts = days[event["date"]].setdefault(event["cell_id"], Counter())
            counts[event["cause"]] += 1
    return days


def write_panel(path, schema, ids, index):
    ids_arrow = pa.array(ids)
    block = []
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for day64 in expected_days():
            day = day64.astype(object)
            n_fires = np.zeros(len(ids), dtype=np.uint32)
            human, lightning, nearby = np.zeros((3, len(ids)), dtype=bool)
            for cell, counts in index.get(day, {}).items():
                n_fires[cell] = counts.total()
                human[cell] = counts["human"] > 0
                lightning[cell] = counts["lightning"] > 0
            for adjacent in (day - ONE_DAY, day, day + ONE_DAY):
                nearby[list(index.get(adjacent, {}))] = True
            dates = pa.repeat(pa.scalar(day, type=pa.date32()), len(ids))
            columns = [ids_arrow, dates, *(pa.array(a) for a in (n_fires > 0, human, lightning, n_fires, nearby))]
            block.append(pa.RecordBatch.from_arrays(columns, schema=schema))
            if len(block) == 14:
                writer.write_table(pa.Table.from_batches(block))
                block.clear()
        if block:
            writer.write_table(pa.Table.from_batches(block))


def audit_panel(path, ids, index, stats):
    names = LABEL_SCHEMA.names
    if not pq.read_schema(path).equals(LABEL_SCHEMA, check_metadata=False):
        raise ValueError("Label schema differs from the output contract")
    report = {year: dict.fromkeys(("rows", "positive_any", "positive_human", "positive_lightning",
                                  "positive_pm1_check_only", "n_fires"), 0) for year in LABEL_YEARS}
    for day64, batch in checked_days(path, names, len(ids)):
        day = day64.astype(object)
        if any(column.null_count for column in batch.columns) or not np.array_equal(batch.column("cell_id").to_numpy(), ids):
            raise ValueError(f"Invalid or incomplete label keys or null values on {day}")
        any_fire, human, lightning, counts, pm1 = (batch.column(name).to_numpy(zero_copy_only=False) for name in names[2:])
        same_day = index.get(day, {})
        if (not np.array_equal(any_fire, counts > 0) or set(np.flatnonzero(any_fire)) != same_day.keys()
                or any(counts[i] != c.total() for i, c in same_day.items())
                or set(np.flatnonzero(human)) != {i for i, c in same_day.items() if c["human"]}
                or set(np.flatnonzero(lightning)) != {i for i, c in same_day.items() if c["lightning"]}
                or set(np.flatnonzero(pm1)) != (same_day.keys() | index.get(day - ONE_DAY, {}).keys()
                                               | index.get(day + ONE_DAY, {}).keys())):
            raise ValueError(f"Stored labels disagree with dated NFDB points on {day}")
        yearly = report[day.year]
        yearly["rows"] += len(ids)
        yearly["positive_any"] += int(any_fire.sum())
        yearly["positive_human"] += int(human.sum())
        yearly["positive_lightning"] += int(lightning.sum())
        yearly["positive_pm1_check_only"] += int(pm1.sum())
        yearly["n_fires"] += int(counts.sum())
    for year, yearly in report.items():
        if yearly["rows"] != len(ids) * 184 or yearly["n_fires"] != stats[year]["label_season_fires"]:
            raise ValueError(f"Label rows or fire totals differ from the calendar or source for {year}")
        for key in ("positive_any", "positive_human", "positive_lightning", "positive_pm1_check_only"):
            yearly[f"{key}_rate"] = yearly[key] / yearly["rows"]
    return report


def label_changes(before, after):
    changed = []
    fire_any = 0
    for day, old_cells in sorted(before.items()):
        if not 5 <= day.month <= 10:
            continue
        new_cells = after.get(day, {})
        for cell in sorted(old_cells.keys() | new_cells.keys()):
            old = sum(old_cells.get(cell, {}).values())
            new = sum(new_cells.get(cell, {}).values())
            if old != new:
                changed.append({"date": day.isoformat(), "cell_id": cell, "before": old, "after": new})
                fire_any += (old > 0) != (new > 0)
    return {"baseline": "one fire per filtered NFDB source row", "n_fires_values_changed": len(changed),
            "fire_any_values_changed": fire_any, "changed_cell_days": changed}


def hand_checks(path, archive, events, squares):
    # One actual fire per cause; prefer a mixed/multiple-fire cell-day within each cause.
    eligible = [e for e in events if e["date"].year in LABEL_YEARS and 5 <= e["date"].month <= 10]
    multiplicity = Counter((e["cell_id"], e["date"]) for e in eligible)
    picks = {}
    used = set()
    for cause in CAUSES.values():
        fire = min((e for e in eligible if e["cause"] == cause and (e["cell_id"], e["date"]) not in used),
                   key=lambda e: (-multiplicity[e["cell_id"], e["date"]], e["date"], e["source_line"]))
        picks[fire["source_line"]] = fire
        used.add((fire["cell_id"], fire["date"]))
    results = []
    for line, row in ontario_rows(archive):
        if line not in picks:
            continue
        fire = picks[line]
        day = date.fromisoformat(row["REP_DATE"].split()[0])
        px, py = PROJECT.transform(float(row["LONGITUDE"]), float(row["LATITUDE"]))
        # Check against the persisted square polygons, without the lattice-floor join.
        containing = np.flatnonzero(shapely.covers(squares, shapely.Point(px, py)))
        cell = fire["cell_id"]
        if (list(containing) != [cell] or fire["date"] != day or CAUSES[row["CAUSE"]] != fire["cause"]
                or row["FIRE_ID"].strip() != fire["fire_id"]):
            raise ValueError(f"Hand check failed for {MEMBER}:{line}")
        nearby = [e for e in events if e["cell_id"] == cell and abs((e["date"] - day).days) <= 1]
        same_day = [e for e in nearby if e["date"] == day]
        expected = {"fire_any": bool(same_day), "fire_human": any(e["cause"] == "human" for e in same_day),
                    "fire_lightning": any(e["cause"] == "lightning" for e in same_day),
                    "n_fires": len(same_day), "fire_any_pm1": bool(nearby)}
        output = pq.read_table(path, columns=LABEL_SCHEMA.names[2:],
                               filters=[("date", "=", day), ("cell_id", "=", cell)], partitioning=None).to_pylist()
        if output != [expected] or not same_day:
            raise ValueError(f"Hand arithmetic differs from labels for {MEMBER}:{line}")
        results.append({**fire, "date": day.isoformat(), "raw_report_date": row["REP_DATE"],
                        "raw_cause": row["CAUSE"], "x_3978": px, "y_3978": py,
                        "same_day_source_lines": [e["source_line"] for e in same_day],
                        "pm1_source_lines": [e["source_line"] for e in nearby], "expected": expected, "stored": output[0]})
    if len(results) != 3:
        raise ValueError("Expected three distinct source-row checks")
    return results


def build():
    archive = pinned_archive()
    ids, cells, squares = read_domain()
    raw_events, stats, outside, undated = read_events(archive, cells)
    events = merge_exact_records(raw_events, stats)
    near_pairs = flag_near_duplicates(events)
    meta = {**PIN_META, b"grid_sha256": table_hash(GRID).encode(), b"nfdb_release": RELEASE.encode(),
            b"nfdb_archive_sha256": SHA256.encode(), b"nfdb_url": URL.encode(),
            b"report_date": b"NFDB YEAR/MONTH/DAY, unchanged", b"grid_assignment": b"EPSG:3978 half-open tiles, west/south inclusive"}
    meta[b"deduplication"] = b"Exact report date, latitude, longitude and cause; lowest FIRE_ID with numeric suffix ordering"
    label_schema = LABEL_SCHEMA.with_metadata({**meta, b"fire_any_pm1": b"Check only; never train; includes Apr 30 and Nov 1 reports"})
    event_schema = EVENT_SCHEMA.with_metadata({
        **meta, b"merged_fire_ids": b"Source FIRE_ID list, including the retained representative",
        b"near_duplicate": b"Another retained fire on the same date, at a different position within 100 m (WGS84 geodesic)",
    })
    with staged(LABELS) as label_tmp, staged(EVENTS) as event_tmp, staged(CHECKS) as checks_tmp:
        pq.write_table(pa.Table.from_pylist(events, schema=event_schema), event_tmp, compression="zstd")
        saved_events = pq.read_table(event_tmp).to_pylist()
        if saved_events != events:
            raise ValueError("Saved event inventory differs from the filtered NFDB source")
        index = event_index(saved_events)
        write_panel(label_tmp, label_schema, ids, index)
        summary = {"release": RELEASE, "archive_sha256": SHA256, "grid_cells": len(ids),
                   "points_by_year": stats, "labels_by_year": audit_panel(label_tmp, ids, index, stats),
                   "deduplication": {"source_rows": len(raw_events), "retained_fires": len(events),
                                     "exact_rows_removed": len(raw_events) - len(events),
                                     "merged_groups": sum(len(e["merged_fire_ids"]) > 1 for e in events),
                                     "near_duplicate_pairs": near_pairs,
                                     "near_duplicate_flagged_fires": sum(e["near_duplicate"] for e in events)},
                   "label_changes": label_changes(event_index(raw_events), index),
                   "hand_checked_fires": hand_checks(label_tmp, archive, saved_events, squares)}
        checks_tmp.write_text(json.dumps({**summary, "outside_grid_points": outside,
                                          "undated_points": undated}, indent=2) + "\n", encoding="utf-8")
        event_tmp.replace(EVENTS)
        label_tmp.replace(LABELS)
        checks_tmp.replace(CHECKS)
    return summary


if __name__ == "__main__":
    report = build()
    rows = sum(year["rows"] for year in report["labels_by_year"].values())
    print(f"NFDB {RELEASE}: audited {rows:,} label rows over {report['grid_cells']:,} cells; see {CHECKS}")
