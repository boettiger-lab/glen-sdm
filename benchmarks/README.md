# Benchmark scripts

Results are in [../docs/benchmarks.md](../docs/benchmarks.md). Run from the repo root.

```bash
mkdir -p /tmp/sdmbench
PYTHONPATH=. python benchmarks/make_dataset.py /tmp/sdmbench/cats.npz        # pulls GBIF + covariates (~30 s)
PYTHONPATH=. python benchmarks/export_training_set.py /tmp/sdmbench/cats.npz /tmp/sdmbench   # train.csv, eval.csv, py_pred.txt
Rscript benchmarks/r_reference.R /tmp/sdmbench                                # maxnet, gbm.step, randomForest, glm
Rscript benchmarks/r_maxnet_predict.R /tmp/sdmbench  # writes r_pred.txt to compare with py_pred.txt
PYTHONPATH=. python benchmarks/hexsdm_timing.py /tmp/sdmbench
PYTHONPATH=. python benchmarks/spatial_cv_compare.py /tmp/sdmbench/cats.npz rf,brt,glm,gam,maxent
```

R packages: maxnet, gbm, randomForest, dismo.

Map display benchmark (hex tiles vs GeoJSON):

```bash
mkdir -p /tmp/gj && cp <predictions.parquet> /tmp/gj/pred.parquet && cp <range.geojson> /tmp/gj/range.geojson
python benchmarks/display_geojson_sizes.py /tmp/gj
cp benchmarks/display_render_bench.html /tmp/gj/bench.html && (cd /tmp/gj && python -m http.server 8765 &)
python benchmarks/display_render_bench.py      # needs playwright + chromium; edit the tile URL to a current register_hex_tiles hash
```
