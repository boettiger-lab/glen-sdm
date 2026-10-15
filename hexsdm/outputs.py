"""Result storage: predictions parquet, range polygons (GeoJSON), compact hex JSON for maps.

Results live under SDM_OUTPUT_DIR/<run_id>/ (local disk). When SDM_OUTPUT_S3 is set
(e.g. s3://public-output/sdm) the predictions parquet is also copied there, so
mcp-data-server's register_hex_tiles can read it and build a vector-tile pyramid.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import h3
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(os.environ.get("SDM_OUTPUT_DIR", "./sdm-output")).resolve()
OUTPUT_S3 = os.environ.get("SDM_OUTPUT_S3", "").rstrip("/")
PUBLIC_BASE = os.environ.get("SDM_PUBLIC_BASE_URL", "http://localhost:8001").rstrip("/")
S3_PUBLIC_HTTP = os.environ.get("SDM_OUTPUT_S3_HTTP", "https://s3-west.nrp-nautilus.io").rstrip("/")
POLYGON_MAX_CELLS = int(os.environ.get("SDM_POLYGON_MAX_CELLS", "12000"))


def run_dir(rid: str) -> Path:
    d = OUTPUT_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_summary(rid: str) -> dict | None:
    p = OUTPUT_DIR / rid / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_summary(rid: str, summary: dict) -> None:
    (run_dir(rid) / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))


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
        {"type": "Feature", "geometry": geom, "properties": {"h3_res": pres, "n_cells": len(strs)}}]}, pres


def write_all(rid: str, pred: pd.DataFrame, proj: pd.DataFrame | None, res: int, summary: dict) -> dict:
    d = run_dir(rid)
    col = f"h{res}"
    out = pred.rename(columns={"cell": col})
    out.to_parquet(d / "predictions.parquet", index=False)
    art = {"predictions_parquet": f"{PUBLIC_BASE}/results/{rid}/predictions.parquet",
           "map": f"{PUBLIC_BASE}/map/{rid}",
           "summary": f"{PUBLIC_BASE}/results/{rid}/summary.json"}

    gj, pres = range_polygon(pred.cell[pred.suitable].to_numpy(), res)
    (d / "range.geojson").write_text(json.dumps(gj))
    art["range_geojson"] = f"{PUBLIC_BASE}/results/{rid}/range.geojson"
    art["range_polygon_h3_res"] = pres

    # compact per-cell JSON for the deck.gl viewer: [[h3, suitability*1000, presence], ...]
    _write_cells(d / "cells.json.gz", pred)
    if proj is not None:
        proj.rename(columns={"cell": col}).to_parquet(d / "projection.parquet", index=False)
        _write_cells(d / "cells_future.json.gz", proj)
        gjf, _ = range_polygon(proj.cell[proj.suitable].to_numpy(), res)
        (d / "range_future.geojson").write_text(json.dumps(gjf))
        art["projection_parquet"] = f"{PUBLIC_BASE}/results/{rid}/projection.parquet"
        art["range_future_geojson"] = f"{PUBLIC_BASE}/results/{rid}/range_future.geojson"

    if OUTPUT_S3:
        from .data import connect
        con = connect()
        s3_path = f"{OUTPUT_S3}/{rid}/predictions.parquet"
        con.sql(f"COPY (SELECT * FROM read_parquet('{d / 'predictions.parquet'}')) TO '{s3_path}' (FORMAT parquet)")
        con.close()
        art["predictions_s3"] = s3_path
        art["hex_tiles"] = {
            "tool": "register_hex_tiles (duckdb-geo / mcp-data-server)",
            "sql": f"SELECT {col}, suitability FROM read_parquet('{s3_path}')",
            "agg": "AVG",
        }
    return art


def _write_cells(path: Path, df: pd.DataFrame) -> None:
    pres = df["presence"].to_numpy() if "presence" in df else np.zeros(len(df), bool)
    suit = np.rint(df.suitability.to_numpy() * 1000).astype(int).tolist()
    body = ",".join(f'["{c:x}",{s},{p}]' for c, s, p in
                    zip(df.cell.to_numpy().tolist(), suit, pres.astype(int).tolist()))
    with gzip.open(path, "wt", compresslevel=3) as f:
        f.write("[" + body + "]")
