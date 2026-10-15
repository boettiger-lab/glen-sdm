# Benchmarks and validation

All numbers are from a shared JupyterLab pod on NRP (2026-10-15), reading data from outside the cluster over the public S3 endpoint. In-cluster reads use the internal Ceph endpoint and should be faster.

## Test species

*Ambystoma californiense* (California tiger salamander):
- GBIF 2026-09: 527 quality-filtered records in 205 H3 res-8 cells.
- 18 covariates: 7 CHELSA bioclim, elevation, roughness, 8 land-cover fractions, gHM.
- Default domain: a 100 km buffer, giving 255–425k res-8 cells depending on the outlier filter.

## MaxEnt agrees with R maxnet

Both were fit to the same training set: 204 presence cells, 10,000 background cells and 18 covariates, with feature classes `lqph` and default regularization. Predictions were compared on 50,000 random domain cells:

| | R maxnet 0.1.4 (glmnet) | hexsdm `maxent` |
|---|---|---|
| fit time | 12.1 s | 1.7 s |
| nonzero coefficients | 80 | 76 |
| Pearson r of cloglog predictions | – | 0.996 |
| Spearman ρ | – | 0.992 |
| mean absolute difference | – | 0.003 |

The small differences come from the solver: glmnet follows a 200-step regularization path, while hexsdm solves at the final penalty with a working-set lasso.

## Single-model fit times

Same training set. R defaults follow Valavi et al. 2022 and the dismo vignette.

| Algorithm | R implementation | R fit | hexsdm fit (1 thread) | hexsdm fit (16 threads) |
|---|---|---|---|---|
| MaxEnt | `maxnet` | 12.1 s | 1.7 s | 1.5 s |
| BRT | `dismo::gbm.step` (tc 5, lr 0.01, bag 0.75) | 88 s | 1.4 s (LightGBM, 8 leaves, lr 0.02) | 0.9 s |
| Down-sampled RF | `randomForest(sampsize = c(n, n))`, 500 trees | 0.5 s | 0.6 s | 0.7 s |
| GLM (quadratic, weighted) | `glm` | 0.16 s | 0.05 s | 0.03 s |

- **MaxEnt and BRT** are the slow methods in R, and the reimplementations are 7× and about 60× faster. The BRT settings are not identical, so treat that ratio as approximate.
- **RF and GLM** are already fast in R.
- **Prediction** of MaxEnt over 50,000 cells takes 0.01 s, because only the features with nonzero coefficients are evaluated.

## Down-sampled RF: an approximation that failed

A first version used LightGBM's random-forest mode with class-specific bagging fractions. Its spatial-CV AUC was 0.877 and its Boyce index 0.02, against 0.945 and 0.975 for an exact implementation (bootstrap of n presences and n background points, full-depth scikit-learn trees). The exact version replaced it.

## End to end through MCP

Defaults throughout: MaxEnt + BRT + RF + GLM ensemble, 5-fold spatial block CV, 20,000 background cells, timings from `summary.json`.

| Run | Cells (res) | Data I/O | CV | Fit + predict | Total |
|---|---|---|---|---|---|
| Tiger salamander, target-group background, SSP5-8.5 2041–2070 projection | 255k (8) | 26 s | 20 s | 4 s | 57 s |
| Valley oak (user GeoJSON, 400 points), California domain, 4 models incl. GAM | 521k (8) | 13 s | 13 s | 6 s | 37 s |
| Bobcat, GBIF records since 2000 (72k records), North America | 276k (6) | 172 s | 52 s | 14 s | 245 s |

The bobcat run is dominated by data I/O.
- **Occurrences:** 72k GBIF records across many h0 partitions took 73 s.
- **Covariates:** 16 million res-8 covariate cells were read in order to average them up to res 6, which took 99 s.

A covariate stack precomputed at res 4–7 would cut the second number by one to two orders of magnitude. That stack would belong in `data-workflows`.

## Map display: hex tiles or GeoJSON

The tiger salamander predictions (255,074 res-8 cells) were displayed three ways.

**Option 1: vector tiles from mcp-data-server.** `register_hex_tiles` built a res 8–2 pyramid in 4.6 s on the server (production log: copy 2.9 s, stats 1.3 s, bounds 0.4 s). Tiles are then cut on request.

**Option 2: GeoJSON hex polygons,** generated in Python.

| H3 res | cells | generate | size | gzip |
|---|---|---|---|---|
| 8 | 255,074 | 3.3 s | 61.8 MB | 11.9 MB |
| 7 | 36,699 | 0.5 s | 8.9 MB | 1.8 MB |
| 6 | 5,337 | 0.1 s | 1.3 MB | 0.3 MB |

**Option 3: the dissolved range polygon,** 0.16 MB (0.06 MB gzip).

Render times were measured in MapLibre GL 4.7.1 in headless Chromium (software WebGL), from adding the source to the map's first `idle` event, over 3 repeats. Each run also counted rendered features (`queryRenderedFeatures`) to confirm the layer drew.

| Source | Time to idle | Features rendered |
|---|---|---|
| vector tiles, z6 view of California (serves res 7) | 1.3–1.6 s | 37,036 |
| vector tiles, z9 | 0.53–0.60 s | 18,993 |
| GeoJSON res 6 | 0.39–0.41 s | 5,483 |
| GeoJSON res 7 | 1.29–1.31 s | 37,070 |
| GeoJSON res 8 | 6.8–6.9 s | 256,026 |
| range polygon | 0.34–0.38 s | 2 |

The GeoJSON files were served from localhost, so these times exclude download; at 1.8 MB gzip the res-7 file adds well under a second on a normal connection. The vector tiles came over the network from the production server.

Up to about 40,000 hexes, GeoJSON renders as fast as the tile pyramid. It also needs no tile build, no S3 copy and no second tool call by the agent, and the agent round trip costs more than any of these. Above that size GeoJSON grows linearly and loses badly: 7–8 s plus a 12 MB download at res 8.

hexsdm therefore returns GeoJSON layers by default. The suitability surface is coarsened (by mean) to the finest resolution with at most 40,000 cells, and the range polygon is included. The full-resolution data remain available as hex tiles through `register_hex_tiles` when someone needs to zoom in at res 8.

Reproduce with `benchmarks/display_geojson_sizes.py` and `benchmarks/display_render_bench.py`.

An earlier run of this benchmark wrote hex polygons without the outer ring array. MapLibre dropped them silently, which made the GeoJSON timings meaningless. The feature counts above guard against that.

## Reproducing

```bash
python -m pytest -q                      # unit tests, no network
python -m pytest -q -m network           # end-to-end against the public catalog
```

The R and Python comparison scripts are in [`benchmarks/`](../benchmarks/).
