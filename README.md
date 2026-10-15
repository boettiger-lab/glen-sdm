# hexsdm: species distribution models over MCP

An MCP server that fits species distribution models (SDMs) on demand from occurrence records and the habitat, climate and terrain layers in the [boettiger-lab STAC catalog](https://s3-west.nrp-nautilus.io/public-data/stac/catalog.json). It returns a suitability surface on the H3 grid, a range polygon, cross-validated performance, and optionally a projection under future climate. It is a companion to [mcp-data-server](https://github.com/boettiger-lab/mcp-data-server) and reads the same data the same way: DuckDB streaming H3-indexed parquet from S3.

![California tiger salamander: ensemble suitability on H3 res 8, range polygon, GBIF presences and spatial-CV metrics](docs/img/tiger-salamander.png)

**Status:** first pass for review by SDM experts. See [Open questions](#open-questions-for-reviewers).

## Design

The model unit is an H3 cell. Every covariate layer in the catalog is already indexed at H3 resolution 8 or 9, so assembling a model matrix is a set of DuckDB joins with no raster processing.

1. **Occurrences.** These are either GBIF records (2026-09 snapshot, quality-filtered, isolated outliers removed) or user points as GeoJSON or a list of coordinates. They are reduced to one presence per occupied cell, which is also the spatial thinning step.
2. **Study area.** The default is a buffer around the records, built from rings of res-4 cells. The alternatives are an ISO region or country from Overture, or a bounding box.
3. **Resolution.** The model resolution is the finest of res 5–8 that keeps the domain under 2 million cells.
4. **Covariates.** These are read for every domain cell:
   - CHELSA 2.1 bioclim: 7 variables;
   - Copernicus DEM: elevation and roughness;
   - CGLS-LC100 land-cover fractions: 8 groups;
   - Global Human Modification;
   - any H3 parquet the user names.
5. **Background.** 20,000 random domain cells by default. The `target_group` option instead samples them in proportion to GBIF records of the same taxonomic class, a standard correction for observer bias.
6. **Models and evaluation.**
   - The models are fit and then evaluated by spatial block cross-validation, with H3 parent cells as the blocks.
   - The default members are combined into an ensemble weighted by cross-validated AUC.
   - Cell predictions are thresholded (maxSSS by default) into a binary range, which is dissolved into a polygon.
7. **Projection.** Optionally the fitted models are re-run with a CHELSA future scenario, which reports gained and lost area and flags cells outside the training range.

## Algorithms

| name | type | notes |
|---|---|---|
| `maxent` | presence-background | maxnet reimplementation: l/q/p/h features, default regularization, cloglog. Matches R maxnet to r = 0.996, about 7× faster |
| `brt` | presence-background | boosted trees (LightGBM), tree count by internal CV |
| `rf` | presence-background | down-sampled random forest (Valavi et al. 2021) |
| `glm` | presence-background | logistic, linear + quadratic, down-weighted background |
| `gam` | presence-background | penalized cubic splines |
| `bioclim` | presence-only | envelope baseline |
| `mahalanobis` | presence-only | distance-to-centroid baseline |

The defaults (maxent, brt, rf, glm) follow the top group of the Valavi et al. 2022 benchmark.

- **Literature.** The algorithms, references and implementations are reviewed in [docs/algorithms.md](docs/algorithms.md). That review includes whether Bayesian MCMC models are the gold standard; the benchmarks say no for prediction from presence-only data.
- **Speed and validation.** These are in [docs/benchmarks.md](docs/benchmarks.md).

## MCP tools

- **`list_sdm_algorithms`**, **`list_covariate_layers`**: list what can be fit and the predictors available.
- **`fit_sdm(occurrences, algorithms, covariates, domain, resolution, background, cv_folds, threshold, ensemble, project_to)`**: runs a model.
  - It returns `status: done` with metrics and artifact URLs, or `status: running` with a `run_id` to poll.
  - Identical requests are cached by a hash of the full specification.
- **`get_sdm_result(run_id, wait_seconds)`**: long-polls a running fit.

Example arguments:

```json
{"occurrences": {"gbif": {"species": "Ambystoma californiense"}},
 "background": {"bias": "target_group"},
 "project_to": "chelsa-2-1-ssp585-2041-2070"}
```

## Outputs

Each run writes to `SDM_OUTPUT_DIR/<run_id>/` and serves the files over HTTP:

- `/map/<run_id>`: a map viewer (MapLibre and deck.gl) with the suitability hexes, range polygon, presences, CV metrics, and a current/future toggle;
- `predictions.parquet`: one row per cell, with the H3 index, ensemble suitability, binary range, per-model predictions and an extrapolation count;
- `range.geojson`: the range polygon;
- `summary.json`: the full specification, metrics, variable importance and timings.

When `SDM_OUTPUT_S3` is set, the predictions are also copied to S3, and the result includes a `hex_tiles` block. Its SQL can be passed to mcp-data-server's `register_hex_tiles` to build a vector tile pyramid.

## Running

```bash
uv venv .venv && uv pip install -r requirements.txt pytest
python -m pytest -q
OMP_NUM_THREADS=16 SDM_MODEL_THREADS=16 python server.py      # http://localhost:8001/mcp
claude mcp add --transport http hex-sdm http://localhost:8001/mcp
```

Set `SDM_MODEL_THREADS`, `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` explicitly on shared nodes (8, 8 and 4 on NRP). Without it, LightGBM and BLAS see every core on the node, and the oversubscription made fits about 10× slower in testing.

`k8s/` holds a single-replica Deployment, Service and Ingress for NRP, using the same burstable resource pattern as mcp-data-server.

## Open questions for reviewers

- **Defaults.** Are these right for a general-purpose tool?
  - 20,000 background cells, a 100 km buffer domain, and maxSSS thresholds;
  - the CoordinateCleaner-style outlier rule;
  - the four-model ensemble weighted by AUC.
- **MaxEnt tuning.** The regularization multiplier and feature classes are not tuned (ENMeval-style). Should tuning be on by default, given that it multiplies cost by the size of the grid?
- **GAM smoothness.** It uses a fixed ridge penalty. Is REML selection worth adding?
- **Collinearity.** There is no screening of correlated covariates. Should there be a VIF or correlation filter, or a curated default set?
- **Spatial random effects.** Which formulation would be most useful: an ICAR prior on H3 adjacency, or an INLA-style log-Gaussian Cox process?
- **Absences and other data.** How should presence-absence and survey data (eBird, iNaturalist research grade with effort) enter?
- **Reporting.** What should a reviewable run carry, for example an ODMAP report?

## Known limitations

- Results and job state live on one pod's disk. Multiple replicas need S3 results and job markers, as in mcp-data-server.
- Continental runs are dominated by reading res-8 covariates in order to average them up. Precomputed coarse covariate stacks would remove most of that cost.
- The extrapolation flag counts out-of-range covariates and is not a full MESS.
- TSS and AUC are computed against background, not true absences.
