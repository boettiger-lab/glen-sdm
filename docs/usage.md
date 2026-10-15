---
title: Using fit_sdm
nav_order: 3
---

# Using fit_sdm

{: .no_toc }

1. TOC
{:toc}

## Tools

| Tool | Purpose |
|---|---|
| `list_sdm_algorithms` | algorithms, their data requirements, relative speed and references, and the default set |
| `list_covariate_layers` | covariate groups, defaults and future scenario names |
| `fit_sdm` | fit a model; returns the result, or a `run_id` while it runs |
| `get_sdm_result(run_id, wait_seconds)` | wait for a running fit, up to 120 s per call |

## Running a fit

`fit_sdm` waits up to 45 seconds for the fit to finish.

- **Fast fits.** Most regional fits finish in that time and return `status: "done"` with the full result.
- **Longer fits.** Anything slower returns `status: "running"` and a `run_id`. Call `get_sdm_result(run_id, wait_seconds=60)` until the status is `"done"` or `"failed"`.
- **Typical times.** Regional species usually take 30–60 s. Continent-wide ranges can take several minutes, because covariates are read at H3 resolution 8 before being averaged up.

Identical requests are cached by a hash of the full specification and the server version, so repeating a request returns the stored result immediately.

Minimal request:

```json
{"occurrences": {"gbif": {"species": "Ambystoma californiense"}}}
```

A request with bias correction, a state study area and a 2050 projection:

```json
{
  "occurrences": {"gbif": {"species": "Ambystoma californiense", "min_year": 1990}},
  "domain": {"type": "region", "region": "US-CA"},
  "background": {"bias": "target_group"},
  "project_to": "chelsa-2-1-ssp585-2041-2070"
}
```

## Parameters

| Parameter | Default | Values |
|---|---|---|
| `occurrences` | required | `{"gbif": {...}}`, `{"geojson": ...}`, or `{"points": [...], "absences": [...]}`. See [Data layers: occurrences](data-layers.md#occurrences). |
| `algorithms` | `["maxent", "brt", "rf", "glm"]` | any of `maxent`, `brt`, `rf`, `glm`, `gam`, `bioclim`, `mahalanobis`. See [Algorithms](algorithms.md). |
| `ensemble` | `true` | combine the algorithms, weighted by cross-validated AUC − 0.5. With `false`, the first listed algorithm is the final model and the others are reported in `cv_metrics` only. |
| `covariates` | `["climate", "elevation", "landcover", "human_modification"]` | group names and custom layer objects. See [Data layers: covariates](data-layers.md#covariates). |
| `domain` | `{"type": "buffer", "buffer_km": 100}` | `buffer`, `region`, `country` or `bbox`. See [Data layers: study area](data-layers.md#study-area). |
| `resolution` | finest of 5–8 under 2 million cells | H3 resolution 5–8 |
| `background` | `{"n": 20000, "bias": null}` | `n`; `bias: "target_group"` with optional `target_rank` and `target_name` |
| `cv_folds` | `5` | number of spatial cross-validation folds; `0` skips cross-validation |
| `threshold` | `"maxSSS"` | `"maxSSS"` (maximises sensitivity + specificity), `"p10"` (10th percentile of presence suitability), `"mtp"` (minimum training presence), or a number |
| `project_to` | none | a CHELSA scenario name. See [Data layers: future climate](data-layers.md#future-climate). |

### GBIF options

| Key | Default | Meaning |
|---|---|---|
| `species` or `specieskey` | required | scientific name, or GBIF taxon key |
| `min_year`, `max_year` | none | limit the record years |
| `max_uncertainty_m` | 1000 | maximum coordinate uncertainty; records without one are kept |
| `basis_of_record` | all except fossil and living specimens | explicit list, e.g. `["HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"]` |
| `outlier_filter` | `true` | drop spatially isolated records |

## Cross-validation

Models are evaluated by spatial block cross-validation.

- **Blocks.** Cells are grouped into blocks by their H3 parent. hexsdm picks the coarsest parent resolution (2–7) that still gives at least 3 × `cv_folds` blocks containing presences.
- **Folds.** The blocks are split into folds, so test cells are spatially separated from training cells.
- **Metrics.** Each model reports AUC, TSS and the continuous Boyce index on the held-out folds. The ensemble reports the same metrics for the combined prediction.

## Results

| Field | Contents |
|---|---|
| `status`, `run_id` | `done`, `running` or `failed`, and the cache key |
| `mode` | `presence-background` or `presence-absence` |
| `occurrences` | record and cell counts, the species' class and order, outlier cells removed |
| `domain`, `resolution`, `background_cells`, `background_bias` | the study area and sampling actually used |
| `covariates`, `top_covariates` | predictors used, and the most important by permutation |
| `final_model`, `ensemble_weights` | `ensemble` or the single algorithm, and member weights |
| `cv_metrics` | AUC, Boyce and TSS for each model and the ensemble |
| `threshold`, `suitable_area_km2` | the rule and value used for the range, and the range's area |
| `range_by_region` | the 8 first-level admin regions holding most of the range, with `suitable_km2` and `share_of_range` |
| `projection` | for `project_to`: future suitable area, `gained_km2`, `lost_km2`, `cells_extrapolated`, and its own `range_by_region` |
| `notes` | caveats; the first note states the background used |
| `next_step` | instructions for displaying the result in a geo-agent map |
| `elapsed_s` | fit time on the server |

### Artifacts

Every run writes its outputs to `s3://public-output/sdm/<run_id>/`, served at `https://s3-west.nrp-nautilus.io/public-output/sdm/<run_id>/`.

| Artifact | Contents |
|---|---|
| `map_layers.suitability`, `map_layers.range` | arguments for geo-agent's `add_hex_tile_layer`. The suitability surface is coarsened to at most 40,000 hexes for display. |
| `map_layers_future` | the same pair for the projection |
| `predictions_parquet` | one row per cell: `h<res>`, `suitability`, `suitable` (in range), `n_extrapolated`, `presence`, and `pred_<algorithm>` for each member model |
| `projection_parquet` | the same for the future scenario |
| `range_geojson` | the range as a polygon |
| `summary` | `summary.json`: the full specification, metrics, variable importance and timings |
| `map` | a standalone viewer with full-resolution hexes, range, presences, metrics and a current/future toggle |
| `hex_tiles` | SQL to pass to mcp-data-server's `register_hex_tiles` for a full-resolution tile pyramid |

`predictions_parquet` can be joined to other H3 data in the catalog, for example protected areas, with mcp-data-server's `query` tool or DuckDB directly.

## Known limitations

- The server runs as a single replica with two concurrent fits.
- Continental-scale fits are slow, because covariates are read at resolution 8 and averaged up.
- MaxEnt's regularization multiplier and feature classes are not tuned per species.
- The extrapolation count is not a full MESS analysis.
- AUC and TSS are computed against background cells, not true absences, except in presence-absence mode.
