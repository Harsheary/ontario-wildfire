"""Extract a complete ArcGIS layer as deterministic gzip GeoJSON sequences."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from build_grid_weather import staged


def arc_snapshot(path, url, where, fields, oid, page_size, sha256=None, bbox=None, workers=8, max_offset=None):
    spatial = ({"geometry": bbox, "geometryType": "esriGeometryEnvelope", "inSR": 3978}
               if bbox is not None else {})

    def query(**params):
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
            response = session.get(f"{url}/query", params={"where": where, **spatial, **params}, timeout=(15, 120))
            response.raise_for_status()
            return response.json()

    count = query(f="json", returnCountOnly="true")["count"]
    if count <= 0 or oid not in fields.split(","):
        raise ValueError(f"{url}: empty layer or absent object ID field")

    def page(offset):
        geometry = {"maxAllowableOffset": max_offset} if max_offset is not None else {}
        result = query(f="geojson", outSR=3978, outFields=fields, orderByFields=f"{oid} ASC",
                       resultOffset=offset, resultRecordCount=page_size, **geometry)
        if result.get("crs", {}).get("properties", {}).get("name") != "EPSG:3978":
            raise ValueError(f"{url}: expected projected GeoJSON in EPSG:3978")
        features = result["features"]
        if len(features) != min(page_size, count - offset):
            raise ValueError(f"{url}: incomplete page at offset {offset}")
        return features

    path.parent.mkdir(parents=True, exist_ok=True)
    with staged(path) as tmp:
        seen = None
        with (tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
              ThreadPoolExecutor(workers) as pool):
            for features in pool.map(page, range(0, count, page_size)):
                for feature in features:
                    obj_id = feature["properties"][oid]
                    if seen is not None and obj_id <= seen:
                        raise ValueError(f"{url}: repeated or unordered object ID {obj_id}")
                    seen = obj_id
                    stream.write(json.dumps(feature, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
                    stream.write(b"\n")
        with tmp.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if sha256 is not None and digest != sha256:
            raise ValueError(f"{url}: got {digest}; expected pinned snapshot {sha256}")
        tmp.replace(path)
    return count, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("url")
    parser.add_argument("where")
    parser.add_argument("fields")
    parser.add_argument("oid")
    parser.add_argument("page_size", type=int)
    parser.add_argument("--sha256")
    parser.add_argument("--bbox")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-offset", type=int)
    count, digest = arc_snapshot(**vars(parser.parse_args()))
    print(f"{count} features; SHA-256 {digest}")


if __name__ == "__main__":
    main()
