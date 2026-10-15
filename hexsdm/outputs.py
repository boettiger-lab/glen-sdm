"""Result storage and map-ready outputs.

Each run writes these files under SDM_OUTPUT_DIR/<run_id>/:
  predictions.parquet   one row per model cell (full resolution)
  range.geojson         dissolved range polygon
  cells.json.gz         compact [h3, suitability*1000, presence] rows for the viewer
  summary.json          spec, metrics, artifacts

Map layers for geo-agent go under tiles/hex/<key>/ (data.geojson + metadata.json), in
the sidecar layout that geo-agent's add_hex_tile_layer reads with format="geojson".
The suitability layer is coarsened to at most SDM_DISPLAY_MAX_CELLS cells: in MapLibre
benchmarks (docs/benchmarks.md) GeoJSON up to ~40k hexes renders as fast as vector
tiles, with no tile build and no extra tool call, while 255k hexes took 7-8 s.

When SDM_OUTPUT_S3 is set (e.g. s3://public-output/sdm) every file is also PUT to S3
and the returned URLs point at the public S3 endpoint, so links outlive the pod and
any replica can answer get_sdm_result from S3.
"""
from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path

import h3
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(os.environ.get("SDM_OUTPUT_DIR", "./sdm-output")).resolve()
OUTPUT_S3 = os.environ.get("SDM_OUTPUT_S3", "").rstrip("/")
PUBLIC_BASE = os.environ.get("SDM_PUBLIC_BASE_URL", "http://localhost:8001").rstrip("/")
S3_PUBLIC_HTTP = os.environ.get("SDM_OUTPUT_S3_HTTP", "https://s3-west.nrp-nautilus.io").rstrip("/")
POLYGON_MAX_CELLS = int(os.environ.get("SDM_POLYGON_MAX_CELLS", "12000"))
DISPLAY_MAX_CELLS = int(os.environ.get("SDM_DISPLAY_MAX_CELLS", "40000"))


def _s3_parts() -> tuple[str, str]:
    bucket, _, prefix = OUTPUT_S3.removeprefix("s3://").partition("/")
    return bucket, prefix.strip("/")


def public_url(rel: str) -> str:
    """URL a browser or agent can fetch for a stored file (rel = '<rid>/name' or 'tiles/...')."""
    if OUTPUT_S3:
        bucket, prefix = _s3_parts()
        return f"{S3_PUBLIC_HTTP}/{bucket}/{prefix + '/' if prefix else ''}{rel}"
    if rel.startswith("tiles/"):
        return f"{PUBLIC_BASE}/{rel}"
    rid, name = rel.split("/", 1)
    return f"{PUBLIC_BASE}/results/{rid}/{name}"


def _s3_put(rel: str, body: bytes, content_type: str, gzipped: bool = False) -> None:
    """Anonymous PUT to the deployment's S3 endpoint (public-output allows it in-cluster)."""
    from .data import S3_ENDPOINT
    bucket, prefix = _s3_parts()
    scheme = "http" if S3_ENDPOINT.startswith("rook") else "https"
    key = f"{prefix + '/' if prefix else ''}{rel}"
    headers = {"Content-Type": content_type}
    if gzipped:
        headers["Content-Encoding"] = "gzip"
    req = urllib.request.Request(f"{scheme}://{S3_ENDPOINT}/{bucket}/{key}", data=body,
                                 method="PUT", headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status >= 300:
            raise RuntimeError(f"S3 PUT {key} failed: {r.status}")


def store(rel: str, body: bytes | str, content_type: str, gzip_on_s3: bool = False,
          pre_gzipped: bool = False) -> str:
    """Write a file locally (and to S3 when configured); return its public URL.

    gzip_on_s3 compresses on upload (served with Content-Encoding: gzip);
    pre_gzipped marks a body that is already gzip bytes (local and S3)."""
    if isinstance(body, str):
        body = body.encode()
    p = OUTPUT_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    if OUTPUT_S3:
        if pre_gzipped:
            _s3_put(rel, body, content_type, gzipped=True)
        elif gzip_on_s3:
            _s3_put(rel, gzip.compress(body, 6), content_type, gzipped=True)
        else:
            _s3_put(rel, body, content_type)
    return public_url(rel)


def load_summary(rid: str) -> dict | None:
    p = OUTPUT_DIR / rid / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    if OUTPUT_S3:
        try:
            with urllib.request.urlopen(public_url(f"{rid}/summary.json"), timeout=20) as r:
                s = json.loads(r.read())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(s))
            return s
        except Exception:
            return None
    return None


def save_summary(rid: str, summary: dict) -> None:
    store(f"{rid}/summary.json", json.dumps(summary, indent=1, default=_json_default), "application/json")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def cell_area_km2(res: int) -> float:
    return h3.average_hexagon_area(res, unit="km^2")


def _coarsen(cells: np.ndarray, res: int, max_cells: int) -> tuple[list[str], int]:
    """Majority-rule coarsening of a binary cell set until it is <= max_cells."""
    cur = [h3.int_to_str(int(c)) for c in cells]
    while len(cur) > max_cells and res > 3:
        res -= 1
        s = pd.Series([h3.cell_to_parent(c, res) for c in cur]).value_counts()
        # children are counted at res+1 -> majority means >= 4 of 7 children suitable
        cur = list(s[s >= 4].index)
    return cur, res


def range_polygon(cells: np.ndarray, res: int) -> tuple[dict, int]:
    if len(cells) == 0:
        return {"type": "FeatureCollection", "features": []}, res
    strs, pres = _coarsen(cells, res, POLYGON_MAX_CELLS)
    shape = h3.cells_to_h3shape(strs)
    geom = shape.__geo_interface__
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": geom,
         "properties": {"in_range": 1, "h3_res": pres, "n_cells": len(strs)}}]}, pres


def _display_cells(pred: pd.DataFrame, res: int) -> tuple[pd.DataFrame, int]:
    """Mean suitability at the finest resolution with <= DISPLAY_MAX_CELLS cells."""
    df = pred[["cell", "suitability"]]
    r = res
    while len(df) > DISPLAY_MAX_CELLS and r > 3:
        r -= 1
        parents = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), r)) for c in df.cell],
                           dtype="uint64")
        df = pd.DataFrame({"cell": parents, "suitability": df.suitability.to_numpy()}) \
            .groupby("cell", as_index=False).suitability.mean()
    return df, r


def _hex_geojson(df: pd.DataFrame) -> str:
    feats = []
    for c, s in zip(df.cell.to_numpy().tolist(), df.suitability.to_numpy().tolist()):
        ring = [[round(lng, 5), round(lat, 5)] for lat, lng in h3.cell_to_boundary(format(c, "x"))]
        ring.append(ring[0])
        feats.append('{"type":"Feature","properties":{"suitability":%.3f},"geometry":{"type":"Polygon","coordinates":[%s]}}'
                     % (s, json.dumps(ring, separators=(",", ":"))))
    return '{"type":"FeatureCollection","features":[' + ",".join(feats) + "]}"


def _layer(key: str, geojson: str, value_col: str, res: int, stats: dict, bounds, display_name: str,
           **style) -> dict:
    """Store an add_hex_tile_layer-compatible GeoJSON layer; return the tool arguments."""
    meta = {"format": "geojson", "value_columns": [value_col], "layer_name": "layer",
            "value_stats": {value_col: {"by_res": {str(res): stats}, "suggested_scale": "linear"}},
            "bounds": bounds, "finest_res": res, "min_res": res}
    gj_url = store(f"tiles/hex/{key}/data.geojson", geojson, "application/geo+json", gzip_on_s3=True)
    store(f"tiles/hex/{key}/metadata.json", json.dumps(meta), "application/json")
    tile_url = gj_url.rsplit("/", 1)[0] + "/{z}/{x}/{y}.pbf"
    return {"tile_url": tile_url, "format": "geojson", "geojson_url": gj_url,
            "value_column": value_col, "display_name": display_name, **style}


def _bounds(cells) -> list[float]:
    ll = np.array([h3.cell_to_latlng(format(int(c), "x")) for c in cells])
    return [float(ll[:, 1].min()), float(ll[:, 0].min()), float(ll[:, 1].max()), float(ll[:, 0].max())]


def map_layers(rid: str, pred: pd.DataFrame, res: int, range_gj: dict, label: str, suffix: str = "") -> dict:
    disp, dres = _display_cells(pred, res)
    s = disp.suitability
    bounds = _bounds(pred.cell[pred.suitable] if pred.suitable.any() else pred.cell)
    suit = _layer(f"{rid}{suffix}", _hex_geojson(disp), "suitability", dres,
                  {"min": 0.0, "max": 1.0, "mean": round(float(s.mean()), 4)}, bounds,
                  f"{label} habitat suitability")
    # in_range is constant 1. min == max tells geo-agent to paint one colour and
    # draw a single legend swatch (geo-agent #406) rather than a 0-1 ramp.
    rng = _layer(f"{rid}{suffix}-range", json.dumps(range_gj), "in_range", dres,
                 {"min": 1.0, "max": 1.0, "mean": 1.0}, bounds, f"{label} modelled range",
                 palette="ylorrd", opacity=0.45, fit_bounds=False)
    return {"suitability": suit, "range": rng, "display_h3_res": dres, "display_cells": int(len(disp))}


def write_all(rid: str, pred: pd.DataFrame, proj: pd.DataFrame | None, res: int, summary: dict) -> dict:
    col = f"h{res}"
    occ = summary.get("spec", {}).get("occurrences", {})
    label = (occ.get("gbif") or {}).get("species") or "Species"

    buf = pred.rename(columns={"cell": col}).to_parquet(index=False)
    art = {"predictions_parquet": store(f"{rid}/predictions.parquet", buf, "application/vnd.apache.parquet"),
           "map": f"{PUBLIC_BASE}/map/{rid}",
           "summary": public_url(f"{rid}/summary.json")}

    gj, pres = range_polygon(pred.cell[pred.suitable].to_numpy(), res)
    art["range_geojson"] = store(f"{rid}/range.geojson", json.dumps(gj), "application/geo+json")
    art["range_polygon_h3_res"] = pres
    store(f"{rid}/cells.json.gz", _cells_gz(pred), "application/json", pre_gzipped=True)
    art["map_layers"] = map_layers(rid, pred, res, gj, label)

    if proj is not None:
        buf = proj.rename(columns={"cell": col}).to_parquet(index=False)
        art["projection_parquet"] = store(f"{rid}/projection.parquet", buf, "application/vnd.apache.parquet")
        store(f"{rid}/cells_future.json.gz", _cells_gz(proj), "application/json", pre_gzipped=True)
        gjf, _ = range_polygon(proj.cell[proj.suitable].to_numpy(), res)
        art["range_future_geojson"] = store(f"{rid}/range_future.geojson", json.dumps(gjf), "application/geo+json")
        scen = summary.get("spec", {}).get("project_to", "future").replace("chelsa-2-1-", "")
        art["map_layers_future"] = map_layers(rid, proj, res, gjf, f"{label} {scen}", suffix="-future")

    if OUTPUT_S3:
        # Full-resolution hex tiles remain available through mcp-data-server.
        art["hex_tiles"] = {
            "tool": "register_hex_tiles (duckdb-geo / mcp-data-server)",
            "sql": f"SELECT {col}, suitability FROM read_parquet('{OUTPUT_S3}/{rid}/predictions.parquet')",
            "agg": "AVG",
        }
    return art


def _cells_gz(df: pd.DataFrame) -> bytes:
    pres = df["presence"].to_numpy() if "presence" in df else np.zeros(len(df), bool)
    suit = np.rint(df.suitability.to_numpy() * 1000).astype(int).tolist()
    body = ",".join(f'["{c:x}",{s},{p}]' for c, s, p in
                    zip(df.cell.to_numpy().tolist(), suit, pres.astype(int).tolist()))
    return gzip.compress(("[" + body + "]").encode(), 3)
