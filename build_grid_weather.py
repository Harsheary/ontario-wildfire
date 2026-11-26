"""Build and audit the statutory 10 km Ontario fire-weather grid and daily panel."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS, Transformer
from scipy.sparse import csr_matrix
import shapely
from shapely.geometry import shape

from build_fire_weather import CASR, FIELDS, dap, fetch, session

DATA = Path("data")
ZONES = DATA / "ontario-fire-zones.geojson"
NATIVE = DATA / "ontario-fire-weather.parquet"
GRID = DATA / "grid_10km.parquet"
MAP = DATA / "grid_10km.svg"
WEIGHTS = DATA / "casr_to_grid_weights.parquet"
DAILY = DATA / "grid_fire_weather.parquet"
CHECKS = DATA / "grid_fire_weather.checks.json"
PIN = "d29e12e479fb41a88f922a435bafb762284e641004858b6b52a529a2801e72a6"
PIN_META = {b"zone_sha256": PIN.encode()}
CRS_GRID = CRS.from_epsg(3978)
PROJECT = Transformer.from_crs(4326, CRS_GRID, always_xy=True)
TILE = 10_000
TILE_AREA = TILE ** 2
NR, NC = 778, 706
SEGMENTS = 8
TOL = 1e-6
CF = {
    "grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": 31.758312454493154,
    "grid_north_pole_longitude": 87.59703130293302,
    "north_pole_grid_longitude": 0.0,
    "earth_radius": 6370997.0,
}
YEARS = range(2010, 2025)
FIRST_DAY = date(2010, 5, 1)
AUDIT_DAY = date(2015, 7, 1)
GRID_SCHEMA = pa.schema([
    ("cell_id", pa.int32()), ("centre_x", pa.float64()), ("centre_y", pa.float64()),
    ("geometry", pa.binary()), ("area_km2", pa.float64()),
])
WEIGHT_SCHEMA = pa.schema([
    ("cell_id", pa.int32()), ("casr_cell_id", pa.int32()),
    ("overlap_m2", pa.float64()), ("weight", pa.float64()),
])
PANEL_SCHEMA = pa.schema([
    ("cell_id", pa.int32()), ("date", pa.date32()), ("noon_utc_hour", pa.uint8()),
    *((field, pa.float64()) for field in FIELDS),
])


@contextmanager
def staged(path):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        tmp = Path(file.name)
    try:
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)


def require_pin(metadata, path):
    if (metadata or {}).get(b"zone_sha256") != PIN.encode():
        raise ValueError(f"{path}: expected zone_sha256={PIN} for {ZONES}")


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        if not geometry.is_empty:
            yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def load_region():
    try:
        raw = ZONES.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"Expected {ZONES} with SHA-256 {PIN}") from error
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError(f"{ZONES}: expected SHA-256 {PIN}")
    if not NATIVE.is_file():
        raise ValueError(f"Expected {NATIVE} with zone_sha256={PIN} for {ZONES}")
    require_pin(pq.read_schema(NATIVE).metadata, NATIVE)
    features = json.loads(raw)["features"]
    if sorted(int(f["properties"]["OFFICIAL_NAME"]) for f in features) != list(range(1, 37)):
        raise ValueError(f"{ZONES}: expected one feature per zone 1–36 at pin {PIN}")
    polygons = []
    for feature in features:
        projected = shapely.transform(shape(feature["geometry"]), PROJECT.transform, interleaved=False)
        parts = list(polygon_parts(shapely.make_valid(projected)))
        if not parts:
            raise ValueError(f"Zone {feature['properties']['OFFICIAL_NAME']} has no projected polygons")
        polygons.extend(parts)
    region = shapely.union_all(polygons)
    if region.is_empty or not region.is_valid or region.area <= 0:
        raise ValueError("Empty or invalid statutory region")
    return region


def square_at(x, y):
    half = TILE / 2
    return shapely.box(x - half, y - half, x + half, y + half)


def build_grid(region):
    xmin, ymin, xmax, ymax = region.bounds
    cols, rows = np.meshgrid(np.arange(math.floor(xmin / TILE), math.ceil(xmax / TILE)),
                             np.arange(math.floor(ymin / TILE), math.ceil(ymax / TILE)))
    x, y = (cols + 0.5) * TILE, (rows + 0.5) * TILE
    inside = shapely.contains_xy(region, x, y)
    x, y = x[inside], y[inside]
    if not x.size:
        raise ValueError("No lattice centres strictly inside the statutory region")
    squares = square_at(x, y)
    geo = {"version": "1.0.0", "primary_column": "geometry", "columns": {"geometry": {
        "encoding": "WKB", "geometry_types": ["Polygon"], "crs": CRS_GRID.to_json_dict(),
    }}}
    metadata = {**PIN_META, b"geo": json.dumps(geo).encode()}
    return pa.Table.from_arrays([
        np.arange(x.size, dtype=np.int32), x, y, shapely.to_wkb(squares), shapely.area(squares) / 1e6,
    ], schema=GRID_SCHEMA.with_metadata(metadata))


def read_grid(path, region):
    table = pq.read_table(path)
    require_pin(table.schema.metadata, path)
    if table.schema != GRID_SCHEMA or any(c.null_count for c in table.columns) or not len(table):
        raise ValueError("Invalid or empty grid table")
    geo = json.loads(table.schema.metadata[b"geo"])
    geometry = geo["columns"]["geometry"]
    if (geo["version"] != "1.0.0" or geo["primary_column"] != "geometry"
            or geometry["encoding"] != "WKB" or geometry["geometry_types"] != ["Polygon"]
            or CRS.from_json_dict(geometry["crs"]) != CRS_GRID):
        raise ValueError("Invalid GeoParquet geometry metadata")
    ids, x, y, wkb, area = (col.to_numpy() for col in table.columns)
    squares = shapely.from_wkb(wkb)
    centres = shapely.get_coordinates(shapely.centroid(squares))
    if (not np.array_equal(ids, np.arange(len(ids))) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not (x % TILE == TILE / 2).all() or not (y % TILE == TILE / 2).all()
            or not np.array_equal(np.lexsort((x, y)), ids)
            or not (shapely.get_type_id(squares) == shapely.GeometryType.POLYGON).all()
            or not (shapely.get_num_coordinates(squares) == 5).all() or not shapely.is_valid(squares).all()
            or not shapely.equals(squares, square_at(x, y)).all()
            or not np.array_equal(centres, np.column_stack((x, y)))
            or not (shapely.area(squares) == TILE_AREA).all() or not (area == TILE_AREA / 1e6).all()):
        raise ValueError("Grid IDs, lattice order, square geometry, centres or area do not match")

    # Independent enumeration: exact envelope limits for centres and the point-within predicate.
    xmin, ymin, xmax, ymax = region.envelope.bounds
    envelope_centres = np.array([
        ((col + 0.5) * TILE, (row + 0.5) * TILE)
        for row in range(math.ceil(ymin / TILE - 0.5), math.floor(ymax / TILE - 0.5) + 1)
        for col in range(math.ceil(xmin / TILE - 0.5), math.floor(xmax / TILE - 0.5) + 1)
    ])
    shapely.prepare(region)
    inside = shapely.within(shapely.points(envelope_centres), region)
    expected = set(map(tuple, envelope_centres[inside]))
    actual = set(zip(x, y, strict=True))
    report = {"cells": len(ids), "duplicate_centres": len(ids) - len(actual),
              "extra_centres": len(actual - expected), "missing_centres": len(expected - actual)}
    if any(report[key] for key in ("duplicate_centres", "extra_centres", "missing_centres")):
        raise ValueError(f"Grid domain differs from statutory centre set: {report}")
    return ids, x, y, squares, report


def write_map(path, region, squares):
    bounds = np.concatenate((shapely.bounds(region)[None, :], shapely.bounds(squares)))
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)
    scale = min(960 / (xmax - xmin), 730 / (ymax - ymin))
    width = math.ceil((xmax - xmin) * scale) + 40
    height = math.ceil((ymax - ymin) * scale) + 100

    def ring_path(ring):
        return "M" + "L".join(f"{20 + (x - xmin) * scale:.2f},{52 + (ymax - y) * scale:.2f}"
                              for x, y in list(ring.coords)[:-1]) + "Z"

    display = shapely.simplify(region, 100, preserve_topology=True)
    boundary = "".join(ring_path(ring) for p in polygon_parts(display)
                       for ring in (p.exterior, *p.interiors))
    tiles = "".join(ring_path(square.exterior) for square in squares)
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
  <title id="title">Ontario statutory fire region and 10 km grid</title>
  <defs><path id="region" d="{boundary}"/></defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <g font-family="sans-serif" fill="#183b46">
    <text x="20" y="29" font-size="19" font-weight="bold">Ontario statutory fire region · 10 km grid</text>
    <text x="{width - 23}" y="47" text-anchor="end" font-size="13">↑ Grid north</text>
    <use href="#region" fill="#dee9df" fill-rule="evenodd"/>
    <path d="{tiles}" fill="none" stroke="#396d83" stroke-width="0.35"/>
    <use href="#region" fill="none" stroke="#304e3c" stroke-width="0.85"/>
    <rect x="20" y="{height - 29}" width="16" height="12" fill="#dee9df" stroke="#304e3c"/>
    <text x="42" y="{height - 19}" font-size="13">Statutory footprint (zones 1–36)</text>
    <rect x="263" y="{height - 29}" width="16" height="12" fill="none" stroke="#396d83"/>
    <text x="285" y="{height - 19}" font-size="13">Full target square · EPSG:3978</text>
  </g>
</svg>
''', encoding="utf-8")


def date_table(path, day, columns):
    return pq.read_table(path, columns=columns, filters=[("date", "=", day)], partitioning=None)


def native_centres():
    schema = pq.read_schema(NATIVE)
    require_pin(schema.metadata, NATIVE)
    if (schema.metadata.get(b"CaSR_version") != b"3.2"
            or schema.metadata.get(b"cell_id") != b"zero-based rlat_index * 706 + rlon_index"
            or schema.field("cell_id").type != pa.int32() or schema.field("date").type != pa.date32()):
        raise ValueError(f"{NATIVE}: expected CaSR v3.2, date32, and documented 706-column native indexing")
    table = date_table(NATIVE, FIRST_DAY, ["cell_id", "longitude", "latitude"])
    if not len(table) or any(c.null_count for c in table.columns):
        raise ValueError(f"{NATIVE}: missing native IDs or coordinates on {FIRST_DAY}")
    ids, lon, lat = (c.to_numpy() for c in table.columns)
    order = np.argsort(ids)
    ids = ids[order]
    if ids[0] < 0 or ids[-1] >= NR * NC or (np.diff(ids) <= 0).any():
        raise ValueError(f"{NATIVE}: invalid or duplicate donor IDs on {FIRST_DAY}")
    return ids, lon[order], lat[order]


def native_axes():
    with session() as s:
        das = fetch(s, f"{CASR}.das").text
        # Read one attribute block, rather than mixing attributes across variables.
        blocks = [m for m in re.findall(r"\{([^{}]*)\}", das) if re.search(
            r'\bgrid_mapping_name\s+"rotated_latitude_longitude"\s*;', m)]
        if len(blocks) != 1:
            raise ValueError("CaSR DAS must declare one rotated_latitude_longitude mapping")
        attrs = dict(re.findall(r'\b(\w+)\s+("[^"]*"|[-+\d.eE]+)\s*;', blocks[0]))
        for key, expected in CF.items():
            value = attrs.get(key, '""').strip('"')
            if (float(value) if value and isinstance(expected, float) else value) != expected:
                raise ValueError(f"CaSR DAS: {key}={value!r}; expected {expected!r}")
        axes = dap(s, {"rlat": "rlat", "rlon": "rlon"})
    result = []
    for name, size in (("rlat", NR), ("rlon", NC)):
        axis = axes[name].astype(np.float64)
        diffs = np.diff(axis)
        if (axis.shape != (size,) or not np.isfinite(axis).all()
                or not (abs(diffs - 0.09) <= 0.00005).all()):
            raise ValueError(f"CaSR {name}: expected {size} increasing centres spaced 0.09° ± 0.00005°")
        result.append(axis)
    return result


def subdivided_edges(axis):
    edges = np.empty(len(axis) + 1)
    edges[1:-1] = (axis[:-1] + axis[1:]) / 2
    edges[0] = axis[0] - (axis[1] - axis[0]) / 2
    edges[-1] = axis[-1] + (axis[-1] - axis[-2]) / 2
    fractions = np.arange(SEGMENTS) / SEGMENTS
    return np.append((edges[:-1, None] + np.diff(edges)[:, None] * fractions).ravel(), edges[-1])


def donor_footprints(ids, lon, lat):
    rlat, rlon = native_axes()
    rotated = CRS.from_cf(CF)
    unrotate = Transformer.from_crs(rotated, rotated.source_crs, always_xy=True)
    row, col = np.divmod(ids, NC)
    pred_lon, pred_lat = unrotate.transform(rlon[col], rlat[row])
    lon_error = abs((pred_lon - lon + 180) % 360 - 180)
    if not (lon_error <= 0.0001).all() or not (abs(pred_lat - lat) <= 0.0001).all():
        raise ValueError("CaSR rotated axes and donor ID indexing do not match saved geographic centres")

    ys, xs = subdivided_edges(rlat), subdivided_edges(rlon)
    k = np.arange(SEGMENTS)
    # Encode the canonical shared vertices, then project each used vertex only once.
    dx = np.concatenate((k, np.full(SEGMENTS, SEGMENTS), SEGMENTS - k, np.zeros(SEGMENTS, dtype=int)))
    dy = np.concatenate((np.zeros(SEGMENTS, dtype=int), k, np.full(SEGMENTS, SEGMENTS), SEGMENTS - k))
    keys = (row[:, None] * SEGMENTS + dy) * len(xs) + col[:, None] * SEGMENTS + dx
    vertices, inverse = np.unique(keys, return_inverse=True)
    iy, ix = np.divmod(vertices, len(xs))
    xy = np.column_stack(PROJECT.transform(*unrotate.transform(xs[ix], ys[iy])))
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite native footprint vertices")
    donors = shapely.polygons(xy[inverse.reshape(keys.shape)])
    if not shapely.is_valid(donors).all() or not (shapely.area(donors) > 0).all():
        raise ValueError("Invalid or zero-area projected donor footprint")
    return donors


def build_weights(squares, donor_ids, donors):
    target, source = shapely.STRtree(donors).query(squares)
    areas = shapely.area(shapely.intersection(squares[target], donors[source]))
    if not np.isfinite(areas).all():
        raise ValueError("Nonfinite intersection area")
    keep = areas > 0
    target, source, areas = target[keep], source[keep], areas[keep]
    order = np.lexsort((source, target))
    target, source, areas = target[order], source[order], areas[order]
    total = np.bincount(target, weights=areas, minlength=len(squares))
    metadata = {**PIN_META, b"CaSR_version": b"3.2", b"coordinate_source": CASR.encode(),
                b"coordinate_requests": b".das and .dods?rlat,rlon", b"rotated_grid_cf": json.dumps(CF).encode(),
                b"bounds": b"Adjacent axis midpoints; extrapolated endpoint half-spacings; eight segments per side",
                b"area": b"EPSG:3978 planar intersection m2 against full target squares; normalize over covered area"}
    return pa.Table.from_arrays([target.astype(np.int32), donor_ids[source], areas, areas / total[target]],
                                schema=WEIGHT_SCHEMA.with_metadata(metadata))


def read_weights(path, ids, donor_ids):
    table = pq.read_table(path)
    require_pin(table.schema.metadata, path)
    if table.schema != WEIGHT_SCHEMA or any(c.null_count for c in table.columns):
        raise ValueError("Invalid weight table schema or null edge values")
    target, donor, areas, weights = (c.to_numpy() for c in table.columns)
    columns = np.searchsorted(donor_ids, donor)
    if ((target < 0).any() or (target >= len(ids)).any() or (columns >= len(donor_ids)).any()
            or not np.array_equal(donor_ids[columns], donor)
            or (target[1:] < target[:-1]).any()
            or ((target[1:] == target[:-1]) & (donor[1:] <= donor[:-1])).any()
            or not (np.isfinite(areas) & (areas > 0) & np.isfinite(weights) & (weights > 0)).all()):
        raise ValueError("Weights must be sorted unique positive edges between known targets and native donors")
    totals = np.bincount(target, areas, minlength=len(ids))
    coverage = totals / TILE_AREA
    covered = coverage > 0
    sums = np.bincount(target, weights, minlength=len(ids))
    deviations = abs(sums[covered] - 1)
    bad_sums = ids[covered][deviations > TOL]
    overcovered = ids[coverage > 1 + TOL]
    if bad_sums.size or overcovered.size or not np.allclose(weights, areas / totals[target], rtol=1e-12, atol=0):
        raise ValueError(f"Invalid weight normalization or overlap: bad sums {bad_sums}, overcovered {overcovered}")
    report = {"edges": len(table), "covered_cells": int(covered.sum()),
              "max_weight_sum_deviation": float(deviations.max(initial=0)),
              "weight_sum_outside_tolerance_ids": bad_sums.tolist(),
              "raw_coverage_min": float(coverage.min()), "raw_coverage_max": float(coverage.max()),
              "partial_coverage": [{"cell_id": int(i), "fraction": float(coverage[i])}
                                   for i in ids[covered & (coverage < 1 - TOL)]],
              "uncovered_ids": ids[~covered].tolist()}
    return table, columns, covered, report


def spatial():
    region = load_region()
    grid = build_grid(region)
    with ExitStack() as stack:
        grid_tmp, weights_tmp, map_tmp = (stack.enter_context(staged(p)) for p in (GRID, WEIGHTS, MAP))
        pq.write_table(grid, grid_tmp, compression="zstd")
        ids, _, _, squares, domain = read_grid(grid_tmp, region)
        donor_ids, lon, lat = native_centres()
        donors = donor_footprints(donor_ids, lon, lat)
        pq.write_table(build_weights(squares, donor_ids, donors), weights_tmp, compression="zstd")
        _, _, _, weights = read_weights(weights_tmp, ids, donor_ids)
        write_map(map_tmp, region, squares)
        for tmp, path in ((grid_tmp, GRID), (map_tmp, MAP), (weights_tmp, WEIGHTS)):
            tmp.replace(path)
    return {"domain": domain, "weights": weights}


def expected_days():
    return np.concatenate([np.arange(f"{year}-05-01", f"{year}-11-01", dtype="datetime64[D]")
                           for year in YEARS])


def day_groups(path, names, rows_per_day):
    """Yield ordered date32 groups, carrying the last group across Arrow batches."""
    pending = []
    pending_day = None
    pending_rows = 0
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(batch_size=131_072, columns=names):
            dates = batch.column("date")
            if dates.type != pa.date32() or dates.null_count:
                raise ValueError(f"{path}: dates must be non-null date32")
            days = dates.to_numpy(zero_copy_only=False)
            cuts = np.flatnonzero(days[1:] != days[:-1]) + 1
            start = 0
            for end in (*cuts, len(batch)):
                day = days[start]
                if pending and day != pending_day:
                    yield pending_day, pending
                    pending = []
                    pending_rows = 0
                pending_day = day
                pending_rows += end - start
                if pending_rows > rows_per_day:
                    raise ValueError(f"{path}: too many rows on {day}; expected {rows_per_day}")
                pending.append(batch.slice(start, end - start))
                start = end
    if pending:
        yield pending_day, pending


def checked_days(path, names, rows_per_day):
    for expected, (day, batches) in zip(expected_days(), day_groups(path, names, rows_per_day), strict=True):
        if day != expected:
            raise ValueError(f"{path}: expected date {expected}, found {day}")
        yield day, batches[0] if len(batches) == 1 else pa.Table.from_batches(batches).combine_chunks().to_batches()[0]


def table_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def make_panel(path, ids, donor_ids, noon, matrix, covered):
    native_meta = pq.read_schema(NATIVE).metadata
    schema = PANEL_SCHEMA.with_metadata({
        **PIN_META, b"CaSR_version": b"3.2", b"weights_sha256": table_hash(WEIGHTS).encode(),
        b"units": native_meta[b"units"], b"FWI": native_meta[b"FWI"],
        b"timing": b"Native date unchanged; target centre at/east of -90 degrees: 16 UTC, otherwise 17; no retiming",
        b"aggregation": b"Static planar overlap-weighted mean of each native field; null/nonfinite donors propagate per field",
    })
    names = ["cell_id", "date", *FIELDS]
    ids_arrow, noon_arrow = pa.array(ids), pa.array(noon)
    values = np.empty((len(donor_ids), len(FIELDS)), dtype=np.float64)
    block = []
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for day, batch in checked_days(NATIVE, names, len(donor_ids)):
            keys = batch.column("cell_id")
            if keys.null_count or len(keys) != len(donor_ids):
                raise ValueError(f"{NATIVE}: wrong donor count or null donor ID on {day}")
            found = keys.to_numpy()
            if np.array_equal(found, donor_ids):
                order = slice(None)
            else:
                order = np.argsort(found)
                if not np.array_equal(found[order], donor_ids):
                    raise ValueError(f"{NATIVE}: duplicate, missing or unexpected donor ID on {day}")
            for i, field in enumerate(FIELDS):
                values[:, i] = batch.column(field).to_numpy(zero_copy_only=False)[order]
            missing = ~np.isfinite(values)
            if missing.any():
                values[missing] = 0
                invalid = (matrix @ missing.astype(np.float64)) > 0
                invalid[~covered] = True
            else:
                invalid = np.broadcast_to(~covered[:, None], (len(ids), len(FIELDS)))
            mapped = matrix @ values
            columns = [ids_arrow, pa.repeat(pa.scalar(day.astype(np.int32).item(), type=pa.date32()), len(ids)),
                       noon_arrow]
            columns.extend(pa.array(mapped[:, i], mask=invalid[:, i]) for i in range(len(FIELDS)))
            block.append(pa.RecordBatch.from_arrays(columns, schema=schema))
            if len(block) == 8:
                writer.write_table(pa.Table.from_batches(block))
                block.clear()
        if block:
            writer.write_table(pa.Table.from_batches(block))


def panel_audit(path, ids, noon, covered):
    schema = pq.read_schema(path)
    require_pin(schema.metadata, path)
    if schema != PANEL_SCHEMA:
        raise ValueError("Finished daily table schema differs from the target contract")
    names = schema.names
    nulls = dict.fromkeys(names, 0)
    sizes = dict.fromkeys(YEARS, 0)
    days = 0
    for day, batch in checked_days(path, names, len(ids)):
        found, dates, hours = batch.columns[:3]
        if (found.null_count or dates.null_count or hours.null_count or not np.array_equal(found.to_numpy(), ids)
                or not np.array_equal(hours.to_numpy(), noon)):
            raise ValueError(f"{path}: invalid target IDs or noon times on {day}")
        for name, col in zip(names, batch.columns, strict=True):
            nulls[name] += col.null_count
        for name, col in zip(FIELDS, batch.columns[3:], strict=True):
            missing = col.is_null().to_numpy(zero_copy_only=False)
            values = col.to_numpy(zero_copy_only=False)
            if not missing[~covered].all() or not np.isfinite(values[~missing]).all():
                raise ValueError(f"{path}: uncovered non-null or nonfinite weather in {name} on {day}")
        sizes[day.astype(object).year] += len(batch)
        days += 1
    if days != 2760 or any(n != 184 * len(ids) for n in sizes.values()):
        raise ValueError(f"Daily row/date counts differ from the expected panel: {days} days, {sizes}")
    return {"dates": days, "rows_by_year": sizes, "columns": names, "null_counts": nulls}


def hand_audit(path, weights, longitude):
    fields = ("FWI", "rain_24h")
    native = {row.pop("cell_id"): row for row in date_table(NATIVE, AUDIT_DAY, ["cell_id", *fields]).to_pylist()}
    mapped = {row.pop("cell_id"): row for row in date_table(path, AUDIT_DAY, ["cell_id", *fields]).to_pylist()}
    donors = {}
    for row in weights.select(["cell_id", "casr_cell_id", "weight"]).to_pylist():
        target = row.pop("cell_id")
        donors.setdefault(target, []).append({**row, **native[row["casr_cell_id"]]})
    rank = sorted((i for i, inputs in donors.items() if len(inputs) >= 2 and all(
        row[field] is not None and math.isfinite(row[field]) for row in inputs for field in fields)),
        key=lambda i: (longitude[i], i))
    if len(rank) < 3:
        raise ValueError(f"{AUDIT_DAY}: hand audit needs three targets with multiple complete donors")
    results = []
    for i in (rank[0], rank[len(rank) // 2], rank[-1]):
        comparisons = {}
        for field in fields:
            hand = math.fsum(row["weight"] * row[field] for row in donors[i])
            stored = mapped[i][field]
            if stored is None or not math.isclose(hand, stored, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{AUDIT_DAY} target {i}: {field} hand {hand} != table {stored}")
            comparisons[field] = {"hand": hand, "stored": stored, "absolute_error": abs(hand - stored)}
        results.append({"date": AUDIT_DAY.isoformat(), "cell_id": i, "longitude": float(longitude[i]),
                        "donors": donors[i], "comparisons": comparisons})
    return results


def daily():
    ids, x, y, _, domain = read_grid(GRID, load_region())
    donor_ids, _, _ = native_centres()
    edges, donor_columns, covered, weight_report = read_weights(WEIGHTS, ids, donor_ids)
    matrix = csr_matrix((edges["weight"].to_numpy(), (edges["cell_id"].to_numpy(), donor_columns)),
                        shape=(len(ids), len(donor_ids)))
    longitude, _ = Transformer.from_crs(CRS_GRID, 4326, always_xy=True).transform(x, y)
    noon = np.where(longitude >= -90.0, 16, 17).astype(np.uint8)
    with staged(DAILY) as tmp, staged(CHECKS) as checks_tmp:
        make_panel(tmp, ids, donor_ids, noon, matrix, covered)
        report = {"domain": domain, "weights": weight_report, "panel": panel_audit(tmp, ids, noon, covered),
                  "hand_audits": hand_audit(tmp, edges, longitude)}
        checks_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(DAILY)
        checks_tmp.replace(CHECKS)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("spatial", "daily"))
    action = {"spatial": spatial, "daily": daily}[parser.parse_args().command]
    print(json.dumps(action(), indent=2), flush=True)


if __name__ == "__main__":
    main()
