---
title: Benchmarks
nav_order: 5
---

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

## Agreement with expert range maps

We compared each model against CDFW's State Wildlife Action Plan (SWAP 2025) range maps for seven California species of greatest conservation need (SGCN):
- *Ambystoma californiense*
- *Gambelia sila*
- *Phrynosoma blainvillii*
- *Spea hammondii*
- *Dipodomys ingens*
- *Thamnophis gigas*
- *Actinemys marmorata*

Each species was fit with `domain: {"type": "region", "region": "US-CA"}` at H3 resolution 8. Each model's suitability was then scored by Mann-Whitney AUC over all California cells, with membership of the SGCN range as the label.

**Runs.** These were fit with v0.3.4, and the outputs are at `s3://public-output/sdm/<run_id>/`. The request for each species was `{"occurrences": {"gbif": {"species": <name>}}, "domain": {"type": "region", "region": "US-CA"}, "resolution": 8}`. The RF-only runs added `"algorithms": ["rf"], "ensemble": false`. The four-model runs carry every model's prediction in `pred_*` columns.

| Species | four-model run | RF-only run |
|---|---|---|
| *Actinemys marmorata* | `5d920a73888ed1ff` | `f2fcdbc369a9d3fe` |
| *Ambystoma californiense* | `bc66bc784324e17e` | `7c82d21fc622fdb6` |
| *Dipodomys ingens* | `fae4dae4b2c11f73` | `d9b587c4bc3ee372` |
| *Gambelia sila* | `54de421b88bb0a5c` | `d224966a88ee3edf` |
| *Phrynosoma blainvillii* | `6dfde77b8a9c8d59` | `7070a74ccda524f4` |
| *Spea hammondii* | `cd5823ca4161ead5` | `3f4eb9a1c9716a14` |
| *Thamnophis gigas* | `21b23e8a5a3fcafa` | `bc5f841612e71c6d` |

**Expert ranges.** These are from `s3://public-swap/california/swap-2025/sgcn-ranges/hex/h0=*/data_0.parquet`, matched on `SWAP_Scientific LIKE '<species>%'`. That match merges the three *A. californiense* distinct-population-segment polygons and both *G. sila* features.

**AUC query.** Run it once per `pred_*` column. The combinations (rf + glm, maxent + rf + glm) are equal-weight means of those columns, not CV-weighted. Tied scores get their average rank. RF votes come in steps of 1/500, so ties are common, but in our check average and minimum ranks differed by at most 0.003.

```sql
WITH p AS (SELECT h8, pred_rf AS s FROM read_parquet('s3://public-output/sdm/<run_id>/predictions.parquet')),
g AS (SELECT DISTINCT h8 FROM read_parquet('s3://public-swap/california/swap-2025/sgcn-ranges/hex/h0=*/data_0.parquet', hive_partitioning = true)
      WHERE h0 IN (SELECT DISTINCT h3_cell_to_parent(h8, 0)::BIGINT FROM p) AND SWAP_Scientific LIKE '<species>%'),
j AS (SELECT p.s, g.h8 IS NOT NULL AS y FROM p LEFT JOIN g USING (h8)),
r AS (SELECT y, rank() OVER (ORDER BY s) + (count(*) OVER (PARTITION BY s) - 1) / 2.0 AS rk FROM j)
SELECT (sum(rk) FILTER (y) - count(*) FILTER (y) * (count(*) FILTER (y) + 1) / 2.0)
       / (count(*) FILTER (y) * count(*) FILTER (NOT y)) AS auc
FROM r;
```

| Model | Mean AUC against SGCN range |
|---|---|
| maxent + brt + rf + glm ensemble | 0.962 |
| maxent + rf + glm | 0.962 |
| rf + glm (equal weights) | 0.961 |
| rf | 0.955 |
| glm | 0.953 |
| maxent | 0.952 |
| brt | 0.910 |

- **BRT** was weakest on the species with fewest presences: 0.80 for *D. ingens* (34 presence cells) and 0.85 for *T. gigas* (54).
- **RF + GLM matched the four-model ensemble** at about half the compute. California fits took 20–25 s, against 28–54 s for the four-model ensemble.
- **New default.** From v0.4.0, RF + GLM is the default (`algorithms: ["rf", "glm"]`, CV-weighted). MaxEnt and BRT remain available.
- **Caveat.** SGCN ranges are expert-drawn envelopes, not surveyed distributions. High AUC against them measures agreement with expert opinion, not predictive accuracy for occupancy.

## Compute profile on the NRP deployment

This profile is for the default run on the production pod (8 model threads, OpenBLAS 4): tiger salamander, 255,074 res-8 cells, 201 presence cells, 20,000 background cells, 18 covariates. Script: `benchmarks/profile_pipeline.py`.

| Algorithm | fit | predict all cells | 5-fold spatial CV | permutation importance |
|---|---|---|---|---|
| maxent | 2.89 s | 0.22 s | 9.33 s | 0.30 s |
| brt | 1.10 s | 0.10 s | 0.58 s | 0.22 s |
| rf | 0.59 s | 0.44 s | 3.08 s | 2.26 s |
| glm | 0.05 s | 0.08 s | 0.19 s | 0.08 s |

- **Data assembly** took 6.5 s in the cluster: occurrences plus four covariate groups.
- **Model compute** is about 21 s, and MaxEnt is 12 s of that, mostly its five CV refits. BRT CV is cheap because it reuses the full-data tree count.
- **Inside a MaxEnt fit** (`benchmarks/profile_maxent.py`, 3.1 s), glum's solves take 2.05 s over 5 working-set rounds with at most 255 active columns. Feature construction is 0.54 s and the KKT checks 0.5 s.
- **The hot loops are already compiled.** LightGBM is C++, the scikit-learn trees are Cython, and the dense algebra is BLAS. A Rust (or numba) rewrite would pay off mainly as a dedicated MaxEnt coordinate-descent solver in place of glum's per-round overhead [to test].
- **GPUs are unlikely to help at these sizes** (about 20k training rows, 18 covariates), because transfer and launch overhead would dominate. They would matter for deep or multi-species models or for predicting over tens of millions of cells [to test].

### Concurrent cross-validation

The 20 CV fits (5 folds × 4 algorithms) are independent. Run as concurrent threads on the pod (`benchmarks/parallel_cv.py`):

| Configuration | Time for 20 fits |
|---|---|
| sequential, 8 threads per model | 14.6 s |
| 5 concurrent jobs × 2 threads | 5.9 s |
| 10 × 2, 20 × 1, 10 × 4 | 6.6–6.7 s |

The gain plateaus at about 2.5×, consistent with the Python interpreter lock in MaxEnt's Python code. A process pool forked from the threaded server deadlocked (OpenMP after fork). Any process-based version would need a spawn context and shared-memory inputs. Concurrent CV is not yet in the pipeline.

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
