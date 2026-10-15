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
