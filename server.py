"""MCP server: species distribution models on the H3 grid, on demand.

Tools
  list_sdm_algorithms   - what can be fit, data needs, references
  list_covariate_layers - catalog-backed covariate groups and future scenarios
  fit_sdm               - occurrences + covariates -> suitability surface, range polygon, metrics
  get_sdm_result        - poll a running fit or re-read a finished one

HTTP
  /results/<run_id>/<file>   predictions.parquet, range.geojson, summary.json, cells.json.gz
  /map/<run_id>              MapLibre + deck.gl viewer of the hex surface and range polygon
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from hexsdm import data, outputs, pipeline
from hexsdm.models import ALGORITHM_INFO

mcp = FastMCP(
    "hex-sdm",
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

INLINE_WAIT = float(os.environ.get("SDM_INLINE_WAIT_SECONDS", "45"))
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("SDM_MAX_CONCURRENT", "2")))
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _compact(summary: dict) -> dict:
    """What the agent needs to see; full details stay in summary.json."""
    keep = ["run_id", "status", "mode", "resolution", "occurrences", "background_cells",
            "background_bias", "domain", "covariates", "final_model", "ensemble_weights",
            "cv_metrics", "threshold", "suitable_area_km2", "range_by_region", "projection", "artifacts",
            "elapsed_s", "cached", "notes"]
    out = {k: summary[k] for k in keep if k in summary}
    vi = summary.get("variable_importance") or {}
    if vi:
        tot = {}
        for imp in vi.values():
            for c, v in imp.items():
                tot[c] = tot.get(c, 0) + v
        out["top_covariates"] = [c for c, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:6]]
    out["next_step"] = (
        "To display on a geo-agent map, call add_hex_tile_layer with the arguments in "
        "artifacts.map_layers.suitability, then again with artifacts.map_layers.range "
        "(second, so the range draws on top), passing both exactly as given. Otherwise "
        "link the user to artifacts.map. For full resolution tiles, pass "
        "artifacts.hex_tiles.sql/agg to register_hex_tiles. When describing where the "
        "species is predicted to occur, use range_by_region (suitable area by admin "
        "region, from this model) rather than prior knowledge of the species."
    )
    return out


@mcp.tool()
def list_sdm_algorithms() -> dict:
    """List the species distribution model algorithms this server can fit.

    Returns, per algorithm: data requirement (presence-only vs presence-background),
    relative speed, a one-line description and a reference. The default set
    (maxent, brt, rf, glm, combined as a CV-weighted ensemble) follows the top
    performers of the Valavi et al. 2022 benchmark."""
    return {"algorithms": ALGORITHM_INFO, "default": pipeline.DEFAULTS["algorithms"],
            "ensemble_default": True}


@mcp.tool()
def list_covariate_layers() -> dict:
    """List covariate groups available as predictors, and future climate scenarios.

    Each group is drawn from the STAC catalog (H3 hex parquet on S3). You may also
    pass custom covariates to fit_sdm as {"path": "s3://.../hex/h0=*/data_0.parquet",
    "columns": ["col1", ...], "h3_column": "h8"} for any H3-indexed parquet at
    resolution 8 or finer."""
    return {"layers": COV_INFO, "default": pipeline.DEFAULTS["covariates"],
            "future_scenarios": [f"chelsa-2-1-{s}" for s in data.FUTURE_SCENARIOS]}


COV_INFO = {k: {"description": v["description"], "columns": v["columns"], "stac_collection": v["source"]}
            for k, v in data.COVARIATE_LAYERS.items()}


def _submit(spec: dict) -> tuple[str, concurrent.futures.Future | None]:
    norm = pipeline.normalize(spec)
    rid = pipeline.run_id(norm)
    if outputs.load_summary(rid):
        return rid, None
    with _jobs_lock:
        j = _jobs.get(rid)
        if j and not j["future"].done():
            return rid, j["future"]
        fut = _executor.submit(_run_logged, spec)
        _jobs[rid] = {"future": fut, "started": time.time()}
        return rid, fut


def _run_logged(spec):
    try:
        return pipeline.run(spec, log=lambda m: print(m, file=sys.stderr, flush=True))
    except Exception:
        traceback.print_exc()
        raise


def fit_sdm(
    occurrences: dict,
    algorithms: list[str] | None = None,
    covariates: list | None = None,
    domain: dict | None = None,
    resolution: int | None = None,
    background: dict | None = None,
    cv_folds: int | None = None,
    threshold: str | float | None = None,
    ensemble: bool | None = None,
    project_to: str | None = None,
) -> dict:
    """Fit a species distribution model and return a suitability map, a range
    polygon, and cross-validated performance.

    Parameters
    - occurrences (required), one of:
        {"gbif": {"species": "Ambystoma californiense"}}   GBIF 2026-09, quality-filtered.
            Optional keys: specieskey, min_year, max_year, max_uncertainty_m (1000),
            basis_of_record (list; default excludes fossils and living specimens),
            outlier_filter (default true: drop spatially isolated records,
            CoordinateCleaner cc_outl quantile rule).
        {"geojson": <FeatureCollection of Points> or "https://...geojson"}
            A feature property presence=0 marks a true absence.
        {"points": [[lon, lat], ...], "absences": [[lon, lat], ...]}
    - algorithms: subset of bioclim, mahalanobis, glm, gam, maxent, brt, rf.
      Default ["maxent", "brt", "rf", "glm"].
    - covariates: groups from list_covariate_layers (default climate, elevation,
      landcover, human_modification) and/or custom {"path","columns","h3_column"} dicts.
    - domain (study area): {"type": "buffer", "buffer_km": 100} (default),
      {"type": "region", "region": "US-CA"}, {"type": "country", "country": "MX"},
      or {"type": "bbox", "bbox": [w, s, e, n]}.
    - resolution: H3 resolution 5-8 for the model grid. Default: the finest that
      keeps the domain under 2 million cells (8 for a region-sized range).
    - background: {"n": 20000, "bias": null | "target_group", "target_rank": "class",
      "target_name": "Amphibia"}. target_group samples background in proportion to
      GBIF records of the same taxonomic group, correcting for sampling effort.
    - cv_folds: spatial block cross-validation folds (default 5; 0 skips CV).
    - threshold: "maxSSS" (default), "p10", "mtp", or a number, used for the range polygon.
    - ensemble: combine algorithms weighted by CV AUC (default true).
    - project_to: a future climate scenario, e.g. "chelsa-2-1-ssp585-2041-2070".

    Returns status "done" with metrics, range_by_region (suitable area by
    first-level admin region, largest first; describe the predicted geography
    from this, not from prior knowledge), and artifact URLs (map layers, range
    GeoJSON, predictions parquet), or status "running" with a run_id: then call
    get_sdm_result(run_id, wait_seconds=60). Identical requests are cached.
    """
    spec = {"occurrences": occurrences, "algorithms": algorithms, "covariates": covariates,
            "domain": domain, "resolution": resolution, "background": background,
            "cv_folds": cv_folds, "threshold": threshold, "ensemble": ensemble,
            "project_to": project_to}
    spec = {k: v for k, v in spec.items() if v is not None}
    try:
        rid, fut = _submit(spec)
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    return get_sdm_result(rid, wait_seconds=INLINE_WAIT)


def get_sdm_result(run_id: str, wait_seconds: float = 30) -> dict:
    """Get the result of a fit_sdm run. Long-polls up to wait_seconds (max 120)
    while the run is in progress. Returns the same structure as fit_sdm."""
    s = outputs.load_summary(run_id)
    if s:
        return {"status": "done", **_compact(s)}
    j = _jobs.get(run_id)
    if not j:
        return {"status": "unknown", "run_id": run_id,
                "error": "no such run on this server (it may have been lost on restart); re-submit fit_sdm"}
    try:
        summary = j["future"].result(timeout=min(float(wait_seconds), 120))
        return {"status": "done", **_compact(summary)}
    except concurrent.futures.TimeoutError:
        return {"status": "running", "run_id": run_id,
                "elapsed_s": round(time.time() - j["started"], 1),
                "next_step": f"call get_sdm_result(run_id='{run_id}', wait_seconds=60)"}
    except Exception as e:
        return {"status": "failed", "run_id": run_id, "error": str(e)}


# Served tools: async wrappers run the sync cores in a worker thread so the event
# loop (and /healthz) stays free; they reuse the sync docstrings as descriptions.
async def _fit_sdm_tool(
    occurrences: dict,
    algorithms: list[str] | None = None,
    covariates: list | None = None,
    domain: dict | None = None,
    resolution: int | None = None,
    background: dict | None = None,
    cv_folds: int | None = None,
    threshold: str | float | None = None,
    ensemble: bool | None = None,
    project_to: str | None = None,
) -> dict:
    return await anyio.to_thread.run_sync(lambda: fit_sdm(
        occurrences, algorithms, covariates, domain, resolution, background,
        cv_folds, threshold, ensemble, project_to))


async def _get_sdm_result_tool(run_id: str, wait_seconds: float = 30) -> dict:
    return await anyio.to_thread.run_sync(lambda: get_sdm_result(run_id, wait_seconds))


_fit_sdm_tool.__name__ = "fit_sdm"
_get_sdm_result_tool.__name__ = "get_sdm_result"
mcp.tool(name="fit_sdm", description=fit_sdm.__doc__)(_fit_sdm_tool)
mcp.tool(name="get_sdm_result", description=get_sdm_result.__doc__)(_get_sdm_result_tool)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

ALLOWED = {"predictions.parquet", "projection.parquet", "range.geojson", "range_future.geojson",
           "summary.json", "cells.json.gz", "cells_future.json.gz"}


async def serve_result(request):
    rid, name = request.path_params["rid"], request.path_params["name"]
    if name not in ALLOWED or not rid.isalnum():
        return Response(status_code=404)
    p = outputs.OUTPUT_DIR / rid / name
    if not p.exists():
        # Results written by another replica or before a restart live on S3.
        if outputs.OUTPUT_S3:
            return RedirectResponse(outputs.public_url(f"{rid}/{name}"), status_code=307)
        return Response(status_code=404)
    headers = {"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=86400"}
    if name.endswith(".json.gz"):
        headers["Content-Encoding"] = "gzip"
        return Response(p.read_bytes(), media_type="application/json", headers=headers)
    media = "application/geo+json" if name.endswith("geojson") else None
    return FileResponse(p, media_type=media, headers=headers)


async def serve_layer(request):
    """Local-mode copy of the S3 tiles/hex/<key>/ layout read by geo-agent."""
    key, name = request.path_params["key"], request.path_params["name"]
    if name not in ("data.geojson", "metadata.json") or not key.replace("-", "").isalnum():
        return Response(status_code=404)
    p = outputs.OUTPUT_DIR / "tiles" / "hex" / key / name
    if not p.exists():
        return Response(status_code=404)
    media = "application/geo+json" if name.endswith("geojson") else "application/json"
    return FileResponse(p, media_type=media, headers={"Access-Control-Allow-Origin": "*"})


VIEWER = (Path(__file__).parent / "viewer.html").read_text()


async def serve_map(request):
    rid = request.path_params["rid"]
    if not rid.isalnum() or not outputs.load_summary(rid):
        return HTMLResponse("<p>Unknown run id</p>", status_code=404)
    return HTMLResponse(VIEWER.replace("__RUN_ID__", rid))


async def healthz(_):
    return JSONResponse({"ok": True, "version": pipeline.VERSION})


def build_app():
    app = mcp.streamable_http_app()
    app.router.redirect_slashes = False
    app.add_route("/results/{rid}/{name}", serve_result, methods=["GET"])
    app.add_route("/map/{rid}", serve_map, methods=["GET"])
    app.add_route("/tiles/hex/{key}/{name}", serve_layer, methods=["GET"])
    app.add_route("/healthz", healthz, methods=["GET"])
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
