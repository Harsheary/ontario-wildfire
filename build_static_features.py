"""Build fixed Ontario access, recreation and fuel features on the saved 10 km grid."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import gzip
from io import TextIOWrapper
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import osmium
import pyarrow as pa
import pyarrow.parquet as pq
from pyogrio.raw import read as read_vector
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window
import requests
import shapely
from shapely.geometry import shape

from build_grid_weather import CRS_GRID, DATA, GRID, PIN_META, TILE, TILE_AREA, staged, table_hash
from build_labels import read_domain
from fetch_static_arcgis import arc_snapshot

MANIFEST = DATA / "static_sources.json"
CACHE = Path(".cache/static")
OUTPUT = DATA / "static_features.parquet"
CHECKS = DATA / "static_features.checks.json"
MAP = DATA / "static_features.svg"
PROJECT = Transformer.from_crs(4326, CRS_GRID, always_xy=True)
NEAR = 1000
FUEL = {"conifer": tuple(range(101, 108)), "deciduous": (108,), "mixedwood": (109, 110, 111, 112),
        "open_grass": (116, 117), "non_fuel": (119, 121, 122), "water": (118,)}
# NoData, omitted slash (S1–S3), and wetlands of unknown FBP type are not classified into these groups.
FUEL_EXCLUDED = (0, 113, 114, 115, 120)
COMMUNITY_TERMS = {
    "City", "Town", "Separated Town", "Village", "Hamlet", "Organized Hamlet", "Community",
    "Dispersed Rural Community", "Compact Rural Community", "Rural Community", "Urban Community",
    "Suburban Community", "Northern Community", "Residential Area", "Neighbourhood", "Settlement",
    "Rural Village", "Northern Village", "Northern Hamlet", "Northern Settlement", "First Nation Village",
    "Cree Village", "Metis Settlement", "Amerindian Settlement", "Charter Community", "Naskapi Village",
    "Forest Village", "Police Village", "Townsite", "Local Urban District",
}
LIMITATION = ("All inputs are fixed snapshots. Post-2010 infrastructure, parks, campground and recreation data, "
              "2021 census geography/population, and the 2014b fuel map provide hindsight for early panel years. "
              "The 2014b map uses historical burn scars; it is not a pre-2010 snapshot. "
              "These are not as-of-year features.")


def pinned_sources(manifest):
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, source in manifest.items():
        path = CACHE / source["file"]
        expected = source["sha256"]
        if not path.exists():
            if "arcgis" in source:
                arc_snapshot(path, source["url"], **source["arcgis"], sha256=expected)
            else:
                with staged(path) as tmp, requests.get(source["url"], stream=True, timeout=(15, 180)) as response:
                    response.raise_for_status()
                    with tmp.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            output.write(chunk)
                    if table_hash(tmp) != expected:
                        raise ValueError(f"{source['url']}: download differs from pinned SHA-256 {expected}")
                    tmp.replace(path)
        if table_hash(path) != expected:
            raise ValueError(f"{path}: expected pinned SHA-256 {expected}; do not substitute live data")
        paths[key] = path
    return paths


def arc_features(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


@contextmanager
def csv_zip(path, member, *, encoding="utf-8-sig"):
    with ZipFile(path) as bundle, bundle.open(member) as stream:
        yield csv.DictReader(TextIOWrapper(stream, encoding=encoding, newline=""))


def lines_for_rail(path):
    with ZipFile(path) as bundle:
        member, = (name for name in bundle.namelist() if name.endswith("_TRACK.shp"))
    meta, _, wkb, _ = read_vector(f"/vsizip/{path.resolve()}/{member}", columns=[])
    project = Transformer.from_crs(meta["crs"], CRS_GRID, always_xy=True)
    return shapely.transform(shapely.from_wkb(wkb), project.transform, interleaved=False)


def group_pairs(target, source, size):
    order = target.argsort(kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(target, minlength=size))]
    return source[order], offsets


def access(paths, squares, diagnostics):
    roads = []
    excluded = Counter()
    for feature in arc_features(paths["roads"]):
        road_type = feature["properties"]["ROAD_ELEMENT_TYPE"]
        if road_type == "ROAD ELEMENT":
            roads.append(shape(feature["geometry"]))
        elif road_type in {"VIRTUAL ROAD", "FERRY CONNECTION"}:
            excluded[road_type] += 1
        else:
            raise ValueError(f"Unknown ORN road element type: {road_type}")
    road = np.array(roads)
    rail = np.concatenate([lines_for_rail(paths[key]) for key in ("rail_on", "rail_mb", "rail_qc")])
    if (not len(road) or not len(rail) or any(not np.isin(shapely.get_type_id(g), [1, 5]).all() for g in (road, rail))
            or shapely.is_empty(road).any() or shapely.is_empty(rail).any()):
        raise ValueError("Empty or non-line road/rail input")
    n = len(squares)
    network = np.concatenate((road, rail))
    tree = shapely.STRtree(network)
    ti, si = tree.query(squares, predicate="dwithin", distance=NEAR)
    lines, offsets = group_pairs(ti, si, n)
    road_km = np.zeros(n)
    near_share = np.zeros(n)
    for i, square in enumerate(squares):
        candidates = lines[offsets[i]:offsets[i + 1]]
        if not candidates.size:
            continue
        road_km[i] = shapely.length(shapely.intersection(road[candidates[candidates < len(road)]], square)).sum() / 1000
        xmin, ymin, xmax, ymax = square.bounds
        # Cut distant portions just outside the buffer's reach before building the local union.
        margin = NEAR + 1
        local = shapely.clip_by_rect(network[candidates], xmin - margin, ymin - margin, xmax + margin, ymax + margin)
        buffers = shapely.buffer(local, NEAR, quad_segs=16)
        near_share[i] = square.intersection(shapely.union_all(buffers)).area / TILE_AREA
    diagnostics["roads_rail"] = {"orn_road_elements": len(road), "orn_excluded_elements": dict(excluded),
                                 "nrwn_tracks": len(rail), "buffer_radius_m": NEAR,
                                 "buffer_segments_per_quadrant": 16}
    return {"road_length_km": road_km, "road_rail_1km_share": near_share}


def project_points(lonlat):
    xy = np.array(lonlat, dtype=np.float64)
    if not len(xy) or not ((abs(xy[:, 0]) <= 180) & (abs(xy[:, 1]) <= 90)).all():
        raise ValueError("Empty or invalid geographic point collection")
    return shapely.points(*PROJECT.transform(xy[:, 0], xy[:, 1], errcheck=True))


def communities(path, centres, diagnostics):
    points = []
    with csv_zip(path, "cgn_canada_csv_eng.csv") as reader:
        for row in reader:
            if row["Generic Category"] == "Populated Place" and row["Generic Term"] in COMMUNITY_TERMS:
                points.append((float(row["Longitude"]), float(row["Latitude"])))
    places = project_points(points)
    target, source = shapely.STRtree(places).query_nearest(centres, all_matches=False)
    distance = np.full(len(centres), np.nan)
    distance[target] = shapely.distance(centres[target], places[source]) / 1000
    diagnostics["populated_place_points"] = len(places)
    return {"nearest_community_km": distance}


def polygon_snapshots(path, field):
    keys, geometries = [], []
    for feature in arc_features(path):
        keys.append(feature["properties"][field])
        geometries.append(shape(feature["geometry"]))
    polygons = np.array(geometries)
    invalid = ~shapely.is_valid(polygons)
    polygons[invalid] = shapely.make_valid(polygons[invalid])
    if not len(polygons) or not (shapely.area(polygons) > 0).all():
        raise ValueError(f"Missing or zero-area polygons in {path}")
    return keys, polygons, int(invalid.sum())

def intersecting_pieces(squares, polygons):
    target, source = shapely.STRtree(polygons).query(squares, predicate="intersects")
    pieces = shapely.intersection(squares[target], polygons[source])
    area = shapely.area(pieces)
    keep = area > 0
    return target[keep], source[keep], pieces[keep], area[keep]


def union_area(target, pieces, n):
    sources, offsets = group_pairs(target, np.arange(len(pieces)), n)
    area = np.zeros(n)
    for i in np.flatnonzero(np.diff(offsets)):
        area[i] = shapely.area(shapely.union_all(pieces[sources[offsets[i]:offsets[i + 1]]]))
    return area



def population(paths, squares, diagnostics):
    populations = {}
    pop_field = "Population and dwelling counts (5): Population, 2021 [1]"
    with csv_zip(paths["da_population"], "98100015.csv") as rows:
        for row in rows:
            if row["DGUID"].startswith("2021S0512"):
                populations[row["DGUID"]] = float(row[pop_field]) if row[pop_field] else np.nan
    keys, polygons, repaired = polygon_snapshots(paths["da_boundaries"], "DGUID")
    if len(set(keys)) != len(keys):
        raise ValueError("DA DGUID is not unique")
    n = len(squares)
    target, source, pieces, overlap = intersecting_pieces(squares, polygons)
    people = np.array([populations.get(key, np.nan) for key in keys])
    assigned = np.bincount(target, overlap / shapely.area(polygons)[source] * people[source], minlength=n)
    coverage = union_area(target, pieces, n) / TILE_AREA
    summed_coverage = np.bincount(target, overlap, minlength=n) / TILE_AREA
    diagnostics["census"] = {"da_polygons": len(polygons), "repaired_polygons": repaired,
                             "intersecting_das_with_missing_population": int(np.unique(source[~np.isfinite(people[source])]).size),
                             "max_overlapping_da_area_share": float(np.max(summed_coverage - coverage)),
                             "target_cells_with_no_da_area": int(np.count_nonzero(coverage == 0))}
    missing = np.isnan(assigned)
    return {"population_per_km2": np.where(missing, 0, assigned / (TILE_AREA / 1e6)),
            "pop_missing": missing, "census_da_coverage_share": coverage}


def point_counts(points, cells, n):
    xy = shapely.get_coordinates(points)
    cols, rows = np.floor(xy / TILE).astype(np.int64).T
    hits = [cells[tile] for tile in zip(cols, rows, strict=True) if tile in cells]
    return np.bincount(hits, minlength=n).astype(np.int32)


def campgrounds(path):
    # OSM nodes are positions; sites mapped as polygons contribute one representative point.
    filt = osmium.filter.TagFilter(("tourism", "camp_site"), ("tourism", "caravan_site"))
    processor = osmium.FileProcessor(path).with_locations("sparse_file_array").with_areas().with_filter(filt)
    factory = osmium.geom.WKBFactory()
    points = []
    for obj in processor:
        if obj.is_node():
            points.append((obj.location.lon, obj.location.lat))
        elif obj.is_area():
            p = shapely.from_wkb(bytes.fromhex(factory.create_multipolygon(obj))).representative_point()
            points.append((p.x, p.y))
    return project_points(points)


def parks_and_recreation(paths, squares, cells, diagnostics):
    _, parks, repaired = polygon_snapshots(paths["park_boundaries"], "PROTECTED_SITE_IDENT")
    n = len(squares)
    target, _, pieces, _ = intersecting_pieces(squares, parks)
    park_share = union_area(target, pieces, n) / TILE_AREA
    park_xy = []
    for feature in arc_features(paths["park_points"]):
        p = feature["properties"]
        park_xy.append((p["LONGITUDE_DECIMAL_DEGREES"], p["LATITUDE_DECIMAL_DEGREES"]))
    recreation_xy = []
    unlocated = 0
    with csv_zip(paths["recreation"], "ODRSF_V1.0/ODRSF_v1.0.csv", encoding="cp1252") as reader:
        for row in reader:
            if row["PR_UID"] != "35":
                continue
            if row["Longitude"] in {"", ".."} or row["Latitude"] in {"", ".."}:
                unlocated += 1
                continue
            recreation_xy.append((float(row["Longitude"]), float(row["Latitude"])))
    sources = {"provincial_park_point_count": project_points(park_xy), "campground_point_count": campgrounds(paths["campgrounds"]),
               "recreation_point_count": project_points(recreation_xy)}
    columns = {name: point_counts(points, cells, n) for name, points in sources.items()}
    diagnostics["recreation"] = {"park_polygons": len(parks), "repaired_park_polygons": repaired,
                                 "odrsf_ontario_records_without_position": unlocated,
                                 "points": {name: {"source": len(points), "in_grid": int(columns[name].sum())}
                                            for name, points in sources.items()}}
    return {**columns, "provincial_park_share": park_share}


def fuel(path, centres, diagnostics):
    x, y = shapely.get_coordinates(centres).T
    raw_area = np.zeros((len(FUEL), len(x)))
    codes_observed = Counter()
    groups = {code: group for group, codes in enumerate(FUEL.values()) for code in codes}
    with rasterio.open(f"/vsizip/{path.resolve()}/nat_fbpfuels_2014b.tif") as dataset:
        tf = dataset.transform
        project = Transformer.from_crs(CRS_GRID, CRS.from_wkt(dataset.crs.to_wkt()), always_xy=True)
        x_native, y_native = project.transform(x, y, errcheck=True)
        shift = y_native[0] - y[0]
        if (tf.a != 250 or tf.e != -250 or tf.b or tf.d or dataset.nodata != 0
                or not np.allclose(x_native, x, atol=1e-6, rtol=0)
                or not np.allclose(y_native, y + shift, atol=1e-6, rtol=0)):
            raise ValueError("Expected the 250 m 2014b FBP raster in the verified Lambert CRS")
        west = (x - TILE / 2 - tf.c) / tf.a
        col = np.rint(west).astype(int)
        if not np.allclose(west, col, atol=1e-8, rtol=0):
            raise ValueError("FBP columns must align with target grid edges")
        span = TILE // int(tf.a)
        for centre_y in np.unique(y):
            ids = np.flatnonzero(y == centre_y)
            north = centre_y + shift + TILE / 2
            south = north - TILE
            first = math.floor((tf.f - north) / tf.a)
            stop = math.ceil((tf.f - south) / tf.a)
            left, right = col[ids].min(), col[ids].max() + span
            if not (0 <= first < stop <= dataset.height and 0 <= left < right <= dataset.width):
                raise ValueError("Target grid extends beyond the pinned FBP raster extent")
            band = dataset.read(1, window=Window(left, first, right - left, stop - first))
            top = tf.f - tf.a * np.arange(first, stop)
            height = np.minimum(top, north) - np.maximum(top - tf.a, south)
            if (height <= 0).any() or not np.isclose(height.sum(), TILE, atol=1e-7, rtol=0):
                raise ValueError("FBP overlap rows do not cover the target height")
            weights = np.repeat(tf.a * height, span)
            for i in ids:
                c = col[i] - left
                values, bins = np.unique(band[:, c:c + span].ravel(), return_inverse=True)
                areas = np.bincount(bins, weights=weights)
                for code, area in zip(values.tolist(), areas, strict=True):
                    codes_observed[code] += float(area)
                    if code in groups:
                        raw_area[groups[code], i] += area
                    elif code not in FUEL_EXCLUDED:
                        raise ValueError(f"Unmapped CFS fuel class {code}; check the pinned legend")
    total = raw_area.sum(axis=0)
    if (total <= 0).any():
        raise ValueError(f"No classified fuel or water for cells {np.flatnonzero(total <= 0).tolist()}")
    fraction = total / TILE_AREA
    diagnostic = {"coded_m2": dict(codes_observed), "fuel_groups": FUEL, "min_coverage_share": float(fraction.min()),
                  "partial_cells": int(np.count_nonzero(fraction < 1 - 1e-9)), "excluded_codes": FUEL_EXCLUDED}
    diagnostics["fuel"] = diagnostic
    return {**dict(zip((f"fuel_{group}_share" for group in FUEL), raw_area / total, strict=True)),
            "fuel_classified_share": fraction}


def write_map(path, squares, table):
    bounds = shapely.bounds(squares)
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)
    scale = min(294 / (xmax - xmin), 314 / (ymax - ymin))
    palette = ("#f1f5f5", "#dadde8", "#b8bdd4", "#949bc0", "#7078a8", "#555b8b", "#3e416f", "#282952")
    panels = (("road_rail_1km_share", "Within 1 km of road or rail"), ("provincial_park_share", "Inside provincial parks"),
              ("fuel_conifer_share", "Conifer share of classified fuel area"))
    with path.open("w", encoding="utf-8") as svg:
        svg.write('<svg xmlns="http://www.w3.org/2000/svg" width="970" height="440" viewBox="0 0 970 440" role="img">\n'
                  '<title>Ontario 10 km static feature checks</title><rect width="970" height="440" fill="white"/>\n'
                  '<g font-family="sans-serif" fill="#20252d"><text x="20" y="28" font-size="20" font-weight="bold">'
                  'Ontario 10 km static feature checks</text><text x="950" y="28" text-anchor="end" font-size="13">↑ Grid north</text>\n')
        size = TILE * scale
        for panel, (col, title) in enumerate(panels):
            offset = 20 + panel * 319
            svg.write(f'<text x="{offset}" y="58" font-size="14" font-weight="bold">{title}</text>\n')
            colors = defaultdict(list)
            for (left, _, _, top), val in zip(bounds, table[col].to_pylist(), strict=True):
                color = palette[min(int(val * len(palette)), len(palette) - 1)] if val is not None else "#a25d42"
                colors[color].append(f'M{offset + (left - xmin) * scale:.3f},{72 + (ymax - top) * scale:.3f}h{size:.3f}v{size:.3f}h-{size:.3f}z')
            for color, pieces in colors.items():
                svg.write(f'<path fill="{color}" d="{"".join(pieces)}"/>\n')
        svg.write('<text x="20" y="423" font-size="13">Share:   0%</text>\n')
        for i, color in enumerate(palette):
            svg.write(f'<rect x="{90 + i * 24}" y="411" width="24" height="14" fill="{color}"/>\n')
        svg.write('<text x="291" y="423" font-size="13">100%</text><rect x="354" y="411" width="14" height="14" fill="#a25d42"/>'
                  '<text x="375" y="423" font-size="13">Missing</text></g></svg>\n')


def audit(path, ids):
    table = pq.read_table(path)
    if not np.array_equal(table["cell_id"].to_numpy(), ids) or table["cell_id"].null_count:
        raise ValueError("Static features differ from saved grid keys")
    missing = {}
    for name in table.column_names[1:]:
        col = table[name]
        missing[name] = col.null_count
        values = col.to_numpy()
        nan = np.isnan(values)
        if int(nan.sum()) != col.null_count:
            raise ValueError(f"Non-null NaN in {name}")
        present = values[~nan]
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"Invalid values in {name}")
        if name.endswith("_share") and (present > 1 + 1e-8).any():
            raise ValueError(f"Area share exceeds one in {name}: {present.max()}")
    if (table["pop_missing"].type != pa.bool_() or missing["pop_missing"] or missing["population_per_km2"]
            or (table["population_per_km2"].to_numpy()[table["pop_missing"].to_numpy()] != 0).any()):
        raise ValueError("Missing census population must be flagged and replaced with zero density")
    totals = np.stack([table[f"fuel_{group}_share"].to_numpy() for group in FUEL]).sum(axis=0)
    error = abs(totals - 1)
    if not (error <= 1e-10).all():
        raise ValueError("Grouped fuel shares do not sum to one for the entire grid")
    return table, {"rows": len(table), "missing_by_feature": missing,
                   "pop_missing_cells": int(np.count_nonzero(table["pop_missing"].to_numpy())),
                   "fuel_mean_shares": {name: float(table[name].to_numpy().mean())
                                        for name in table.column_names if name.startswith("fuel_")},
                   "fuel_share_sum": {"min": float(totals.min()), "max": float(totals.max()),
                                      "max_absolute_error": float(error.max()), "outside_tolerance": 0}}


def build():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = pinned_sources(manifest)
    ids, cells, squares = read_domain()
    centres = shapely.centroid(squares)
    diagnostics = {}
    columns = {"cell_id": ids, **access(paths, squares, diagnostics), **communities(paths["communities"], centres, diagnostics),
               **population(paths, squares, diagnostics), **parks_and_recreation(paths, squares, cells, diagnostics),
               **fuel(paths["fuel"], centres, diagnostics)}
    metadata = {**PIN_META, b"grid_sha256": table_hash(GRID).encode(), b"source_manifest": json.dumps(manifest).encode(),
                b"temporal_limitation": LIMITATION.encode(),
                b"fuel_denominator": b"Area classified into the six retained FBP groups; see fuel_classified_share",
                b"pop_missing": b"At least one overlapping DA has unavailable population; density filled with zero"}
    table = pa.table({name: pa.array(values, mask=~np.isfinite(values)) for name, values in columns.items()}, metadata=metadata)
    with staged(OUTPUT) as table_tmp, staged(CHECKS) as checks_tmp, staged(MAP) as map_tmp:
        pq.write_table(table, table_tmp, compression="zstd")
        saved, report = audit(table_tmp, ids)
        write_map(map_tmp, squares, saved)
        report.update({"sources": manifest, "temporal_limitation": LIMITATION, **diagnostics})
        checks_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for src, dst in ((map_tmp, MAP), (checks_tmp, CHECKS), (table_tmp, OUTPUT)):
            src.replace(dst)
    return report


if __name__ == "__main__":
    result = build()
    print(f"Audited {result['rows']:,} cells; see {CHECKS}")
