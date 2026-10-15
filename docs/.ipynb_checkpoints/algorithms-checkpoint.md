# Species distribution model algorithms: literature and implementations

This is a summary of the common algorithms for fitting species distribution models (SDMs) from occurrence records and environmental layers, with the canonical papers and the main R and Python implementations. DOIs were checked against Crossref and package pages against CRAN, PyPI and GitHub on 2026-10-15. Items marked [verify] could not be confirmed from the source.

Data types: **PO** presence-only; **PB** presence-background (presences plus random "background" cells, still presence-only data); **PA** presence-absence.

The last column of each table says what `hexsdm` implements.

## Envelope and profile methods (PO)

These use the presences alone. They were among the weakest methods in the NCEAS benchmark (Elith et al. 2006) and are best used as baselines.

| Method | Description | Reference | Code | hexsdm |
|---|---|---|---|---|
| BIOCLIM | Percentile envelope of each covariate at the presences | Booth et al. 2014, [10.1111/ddi.12144](https://doi.org/10.1111/ddi.12144) | [dismo](https://cran.r-project.org/package=dismo) `bioclim`, [predicts](https://cran.r-project.org/package=predicts) `envelope` | `bioclim` |
| DOMAIN | Gower similarity to the nearest presence | Carpenter et al. 1993, [10.1007/BF00051966](https://doi.org/10.1007/BF00051966) | dismo `domain` | not yet |
| Mahalanobis | Distance from the presence centroid, accounting for covariance | Farber & Kadmon 2003, [10.1016/S0304-3800(02)00327-7](https://doi.org/10.1016/S0304-3800(02)00327-7) | dismo `mahal` | `mahalanobis` |
| Convex and alpha hulls | Geometric range polygon with no covariates | Burgman & Fox 2003, [10.1017/S1367943003003044](https://doi.org/10.1017/S1367943003003044) | [alphahull](https://cran.r-project.org/package=alphahull), [rangeBuilder](https://cran.r-project.org/package=rangeBuilder), [red](https://cran.r-project.org/package=red) | not yet |

## Regression (PB or PA)

| Method | Description | Reference | Code | hexsdm |
|---|---|---|---|---|
| GLM | Logistic regression; with PB data the background is down-weighted | Guisan, Edwards & Hastie 2002, [10.1016/S0304-3800(02)00204-1](https://doi.org/10.1016/S0304-3800(02)00204-1) | `glm`, [glmnet](https://cran.r-project.org/package=glmnet); Python [glum](https://pypi.org/project/glum/), statsmodels | `glm` (linear + quadratic, light ridge) |
| IWLR | Infinitely weighted logistic regression, which converges to the Poisson point process (PPM) | Fithian & Hastie 2013, [10.1214/13-AOAS667](https://doi.org/10.1214/13-AOAS667) | | used for GLM/GAM weighting |
| GAM | Smooth splines for each covariate | (as GLM) | [mgcv](https://cran.r-project.org/package=mgcv); Python [pygam](https://pypi.org/project/pygam/) | `gam` (penalized B-splines; smoothness not yet selected by REML) |
| MARS | Piecewise-linear hinge basis | Friedman 1991, [10.1214/aos/1176347963](https://doi.org/10.1214/aos/1176347963); Leathwick et al. 2006, [10.1016/j.ecolmodel.2006.05.022](https://doi.org/10.1016/j.ecolmodel.2006.05.022) | [earth](https://cran.r-project.org/package=earth) | not yet |

Valavi et al. 2022 weight each presence 1 and each background point n_presence / n_background, and use 50,000 background points ([appendix](https://rvalavi.github.io/Presence-Only-SDM/)). hexsdm uses the same weights with 20,000 background cells by default.

## MaxEnt and point processes (PB)

MaxEnt estimates the ratio of covariate density at presences to density over the background, using regularized feature expansions. It is equivalent to a lasso-penalized inhomogeneous Poisson point process, so with enough background points a weighted GLM fits the same model.

- **Original papers.** Phillips, Anderson & Schapire 2006, [10.1016/j.ecolmodel.2005.03.026](https://doi.org/10.1016/j.ecolmodel.2005.03.026); Phillips & Dudík 2008, [10.1111/j.0906-7590.2008.5203.x](https://doi.org/10.1111/j.0906-7590.2008.5203.x).
- **Explanations and the open-source version.** Elith et al. 2011, [10.1111/j.1472-4642.2010.00725.x](https://doi.org/10.1111/j.1472-4642.2010.00725.x); Phillips et al. 2017 (maxnet, cloglog output), [10.1111/ecog.03049](https://doi.org/10.1111/ecog.03049).
- **Point-process equivalence.** Warton & Shepherd 2010, [10.1214/10-AOAS331](https://doi.org/10.1214/10-AOAS331); Renner & Warton 2013, [10.1111/j.1541-0420.2012.01824.x](https://doi.org/10.1111/j.1541-0420.2012.01824.x).
- **R code.** [maxnet](https://cran.r-project.org/package=maxnet) ([GitHub](https://github.com/mrmaxent/maxnet)); Java [Maxent](https://github.com/mrmaxent/Maxent); [ppmlasso](https://cran.r-project.org/package=ppmlasso); tuning with [ENMeval](https://cran.r-project.org/package=ENMeval) (Kass et al. 2021, [10.1111/2041-210X.13628](https://doi.org/10.1111/2041-210X.13628)) and [SDMtune](https://cran.r-project.org/package=SDMtune) (Vignali et al. 2020, [10.1002/ece3.6786](https://doi.org/10.1002/ece3.6786)).
- **Python code.** [elapid](https://github.com/earth-chris/elapid) (Anderson 2023, [10.21105/joss.04930](https://doi.org/10.21105/joss.04930)).
- **hexsdm.** `maxent` reimplements maxnet: linear, quadratic, product and hinge features chosen by sample size, maxnet's regularization tables, presences added to the background, cloglog output and clamping. The lasso is solved with a working-set method. It agrees with R maxnet to r = 0.996 on the same data and runs about 6× faster (see [benchmarks.md](benchmarks.md)). Tuning of the regularization multiplier and feature classes is not yet automated.

## Machine learning (PB or PA)

| Method | Reference | Code | hexsdm |
|---|---|---|---|
| Boosted regression trees (BRT/GBM) | Elith, Leathwick & Hastie 2008, [10.1111/j.1365-2656.2008.01390.x](https://doi.org/10.1111/j.1365-2656.2008.01390.x) | [gbm](https://cran.r-project.org/package=gbm) + `dismo::gbm.step`; [LightGBM](https://pypi.org/project/lightgbm/), [XGBoost](https://pypi.org/project/xgboost/) | `brt` (LightGBM, 8-leaf trees, learning rate 0.02, bagging 0.75, trees chosen by internal 5-fold CV) |
| Random forest | Breiman 2001, [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | [randomForest](https://cran.r-project.org/package=randomForest), [ranger](https://cran.r-project.org/package=ranger); scikit-learn | |
| Down-sampled (balanced) RF | Valavi et al. 2021, [10.1111/ecog.05615](https://doi.org/10.1111/ecog.05615); code [rvalavi/SDMwithRFs](https://github.com/rvalavi/SDMwithRFs) | randomForest `sampsize = c(n, n)`; [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) | `rf` (500 full-depth trees, each on a bootstrap of n presences and n background cells, mtry = √p) |
| XGBoost | Chen & Guestrin 2016, [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | xgboost | not separately (BRT uses LightGBM) |
| SVM | Drake et al. 2006, [10.1111/j.1365-2664.2006.01141.x](https://doi.org/10.1111/j.1365-2664.2006.01141.x) | e1071, scikit-learn | not yet |
| Neural networks | Pearson et al. 2002, [10.1016/S0304-3800(02)00056-X](https://doi.org/10.1016/S0304-3800(02)00056-X); Olden et al. 2008, [10.1086/587826](https://doi.org/10.1086/587826) | nnet, scikit-learn, PyTorch | not yet |

Plain random forests do poorly on PB data because presences are rare and overlap the background. Down-sampling each tree fixes this and puts RF in the top group of Valavi et al. 2022.

## Bayesian and spatial models

| Method | Data | Reference | Code |
|---|---|---|---|
| BART | PB, PA | Chipman et al. 2010, [10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285); Carlson 2020, [10.1111/2041-210X.13389](https://doi.org/10.1111/2041-210X.13389) | [embarcadero](https://github.com/cjcarlson/embarcadero) (wraps [dbarts](https://cran.r-project.org/package=dbarts)); [pymc-bart](https://pypi.org/project/pymc-bart/) |
| INLA / inlabru (log-Gaussian Cox process, SPDE) | PO, PA, integrated | Lindgren et al. 2011, [10.1111/j.1467-9868.2011.00777.x](https://doi.org/10.1111/j.1467-9868.2011.00777.x); Bachl et al. 2019, [10.1111/2041-210X.13168](https://doi.org/10.1111/2041-210X.13168) | [inlabru](https://cran.r-project.org/package=inlabru), [r-inla.org](https://www.r-inla.org/), [PointedSDMs](https://cran.r-project.org/package=PointedSDMs), [intSDM](https://cran.r-project.org/package=intSDM) |
| hSDM | PA, detection | none (package only) | [hSDM](https://cran.r-project.org/package=hSDM) |
| spOccupancy | detection / non-detection | Doser et al. 2022, [10.1111/2041-210X.13897](https://doi.org/10.1111/2041-210X.13897) | [spOccupancy](https://cran.r-project.org/package=spOccupancy) |
| Hmsc (joint SDM) | PA, counts | Ovaskainen et al. 2017, [10.1111/ele.12757](https://doi.org/10.1111/ele.12757); Tikhonov et al. 2020, [10.1111/2041-210X.13345](https://doi.org/10.1111/2041-210X.13345) | [Hmsc](https://cran.r-project.org/package=Hmsc), [hmsc-hpc](https://github.com/hmsc-r/hmsc-hpc) |

None of hexsdm's algorithms are Bayesian yet.

### Are the MCMC Bayesian methods the gold standard?

The benchmarks do not show them winning on prediction from PO data.

- **NCEAS benchmark.** Elith et al. 2006, [10.1111/j.2006.0906-7590.04596.x](https://doi.org/10.1111/j.2006.0906-7590.04596.x): 16 methods, 226 species, evaluated on independent PA data. The newer methods (BRT, MaxEnt, GDM, MARS-COMM) did best [verify per-method ranking]. No Bayesian MCMC methods were included.
- **Valavi et al. 2022 benchmark.** *Ecological Monographs*, [10.1002/ecm.1486](https://doi.org/10.1002/ecm.1486); code [rvalavi/Presence-Only-SDM](https://github.com/rvalavi/Presence-Only-SDM). Same data, 21 model variants.
  - The top group was BRT, down-sampled RF, MaxEnt (tuned and default) and an ensemble [verify exact membership].
  - Default biomod ensembles were no better than a single good model.
  - No Bayesian, INLA or BART models were tested.
- **Joint SDMs.** Norberg et al. 2019, [10.1002/ecm.1370](https://doi.org/10.1002/ecm.1370): 33 variants of joint and stacked SDMs on PA data. No model performed best for all tasks, and the most precise models were often overconfident.
- **SAGE benchmark.** Arens et al. 2026, preprint [arXiv:2609.31082](https://arxiv.org/abs/2609.31082). Random forests and deep SDMs were best, with neither dominating.

Hierarchical Bayesian models are the right tool for uncertainty, imperfect detection, spatial autocorrelation, joint community models and combining data types. For single-species suitability from PO data, the fast regularized and tree-based methods are as accurate. A spatial random effect on the H3 grid (an ICAR prior on hex adjacency) is a natural next addition. INLA would be the fast route to one, since its runtime is closer to seconds than to MCMC hours.

## Ensembles

- **biomod2.** Thuiller et al. 2009, [10.1111/j.1600-0587.2008.05742.x](https://doi.org/10.1111/j.1600-0587.2008.05742.x); [CRAN](https://cran.r-project.org/package=biomod2). The 2026 extension to more data types is Guéguen et al., [10.1002/ecog.08588](https://doi.org/10.1002/ecog.08588).
- **hexsdm.** It averages its members weighted by cross-validated AUC − 0.5. By default the members are MaxEnt, BRT, down-sampled RF and GLM.

## Workflow components

| Component | Reference | Code | hexsdm |
|---|---|---|---|
| Background / pseudo-absence choice | Barbet-Massin et al. 2012, [10.1111/j.2041-210X.2011.00172.x](https://doi.org/10.1111/j.2041-210X.2011.00172.x) | | 20,000 random cells in the domain |
| Target-group background (sampling bias) | Phillips et al. 2009, [10.1890/07-2153.1](https://doi.org/10.1890/07-2153.1) | | `background.bias = "target_group"`: cells sampled in proportion to GBIF records of the same class |
| Spatial thinning | Aiello-Lammens et al. 2015, [10.1111/ecog.01132](https://doi.org/10.1111/ecog.01132) | [spThin](https://cran.r-project.org/package=spThin) | one presence per H3 cell at the model resolution |
| Outlier records | CoordinateCleaner `cc_outl` | [CoordinateCleaner](https://cran.r-project.org/package=CoordinateCleaner) | quantile rule on mean distance, on by default |
| Spatial block cross-validation | Roberts et al. 2017, [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881); Valavi et al. 2019, [10.1111/2041-210X.13107](https://doi.org/10.1111/2041-210X.13107) | [blockCV](https://cran.r-project.org/package=blockCV) | blocks are H3 parent cells, assigned to folds at random |
| Boyce index | Hirzel et al. 2006, [10.1016/j.ecolmodel.2006.05.017](https://doi.org/10.1016/j.ecolmodel.2006.05.017) | ecospat `ecospat.boyce` | continuous Boyce index |
| TSS | Allouche et al. 2006, [10.1111/j.1365-2664.2006.01214.x](https://doi.org/10.1111/j.1365-2664.2006.01214.x) | | reported against background (only a true TSS with PA data) |
| Thresholds | Liu et al. 2013, [10.1111/jbi.12058](https://doi.org/10.1111/jbi.12058); Liu et al. 2016, [10.1002/ece3.1878](https://doi.org/10.1002/ece3.1878) | | maxSSS (default), p10, minimum training presence |
| Extrapolation (MESS) | Elith, Kearney & Phillips 2010, [10.1111/j.2041-210X.2010.00036.x](https://doi.org/10.1111/j.2041-210X.2010.00036.x) | `predicts::mess` | count of covariates outside the training range per cell (simpler than MESS) |
| Reporting standard (ODMAP) | Zurell et al. 2020, [10.1111/ecog.04960](https://doi.org/10.1111/ecog.04960) | | not yet |

## Packages that bundle methods

**R:**
- **tidysdm.** Leonardi et al. 2024, *Methods Ecol. Evol.*, [10.1111/2041-210X.14406](https://doi.org/10.1111/2041-210X.14406); [CRAN](https://cran.r-project.org/package=tidysdm), [GitHub](https://github.com/EvolEcolGroup/tidysdm). This is the recent tidymodels-based bundle: GLM, GAM, maxnet, RF and BRT, with ensembles, spatial CV, thinning, and Boyce and TSS metrics.
- **biomod2.** Ensemble modelling; details above.
- **sdm.** Naimi & Araújo 2016, [10.1111/ecog.01881](https://doi.org/10.1111/ecog.01881).
- **flexsdm.** Velazco et al. 2022, [10.1111/2041-210X.13874](https://doi.org/10.1111/2041-210X.13874).
- **ENMTML.** Andrade et al. 2020, [10.1016/j.envsoft.2019.104615](https://doi.org/10.1016/j.envsoft.2019.104615).
- **wallace.** Kass et al. 2018, [10.1111/2041-210X.12945](https://doi.org/10.1111/2041-210X.12945); a Shiny interface.
- **Others.** [enmSdmX](https://github.com/adamlilith/enmSdmX); [SSDM](https://cran.r-project.org/package=SSDM); the catalogue [sdmverse](https://github.com/sylvainschmitt/sdmverse).

**Python:**
- **elapid.** The main dedicated SDM package.
- **scikit-learn plus helpers.** Used with imbalanced-learn, [pyspatialml](https://pypi.org/project/pyspatialml/) or [malpolon](https://github.com/plantnet/malpolon). malpolon is for deep-learning SDMs (Larcher et al. 2024, [arXiv:2409.18102](https://arxiv.org/abs/2409.18102)).
- The PyPI package named `sdm` is unrelated (sparse distributed memory).

## Deep-learning SDMs

These pool information across many species. They are relevant for rare species and global-scale mapping, but less so for single-species models with tabular covariates.

- **SINR.** Cole et al., ICML 2023, [arXiv:2306.02564](https://arxiv.org/abs/2306.02564), [code](https://github.com/elijahcole/sinr). Trained on iNaturalist presences for about 47k species.
- **DeepMaxent.** [arXiv:2412.19217](https://arxiv.org/abs/2412.19217).
- **GeoLifeCLEF.** Challenges built on these models [verify challenge pages].
