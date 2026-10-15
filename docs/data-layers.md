---
title: Data layers
nav_order: 2
---

# Data layers

A model draws on four kinds of data: occurrence records, the study area, environmental covariates, and (for projections) future climate. All of them are cloud-native H3 parquet in the [boettiger-lab STAC catalog](https://s3-west.nrp-nautilus.io/public-data/stac/catalog.json), read from S3 by DuckDB. This page lists each source, how it enters the model, and which parts a user can change.

{: .no_toc }

1. TOC
{:toc}

## Summary of what can be configured

| Input | Default | Can the user change it? |
|---|---|---|
| Occurrences | GBIF 2026-09 snapshot, quality-filtered | Yes: filters on the GBIF query, or your own points or GeoJSON |
| Study area | 100 km buffer around the records | Yes: buffer distance, region, country or bounding box |
| Covariate groups | climate, elevation, landcover, human_modification | Yes: choose any subset of the four groups, and add your own H3 parquet layers |
| Variables inside a built-in group | fixed | No: a group is included or left out as a whole |
| Climate baseline | CHELSA 2.1, 1981–2010 | No |
| Future climate | none | Yes: one of nine CHELSA 2.1 scenarios through `project_to` |
| Background sampling | 20,000 uniform cells | Yes: number of cells, and target-group bias correction |
| Model resolution | finest of H3 5–8 under 2 million cells | Yes: `resolution` 5–8 |

Parameter syntax is in [Using fit_sdm](usage.md). `list_covariate_layers` returns the current groups, defaults and scenario names from the server.

## Occurrences

**GBIF.** The default source is the GBIF occurrence snapshot of 2026-09-01 ([AWS Open Data](https://registry.opendata.aws/gbif/), [GBIF terms](https://www.gbif.org/terms)), indexed to H3 cells in the catalog. A request names a species (`"species": "Ambystoma californiense"`) or a GBIF `specieskey`. Records are kept when:

- none of the GBIF issue flags `ZERO_COORDINATE`, `COORDINATE_INVALID`, `COORDINATE_OUT_OF_RANGE` or `COUNTRY_COORDINATE_MISMATCH` is set;
- the occurrence status is not `ABSENT`;
- the coordinate uncertainty is at most 1,000 m, or is not recorded (`max_uncertainty_m` changes the limit);
- the basis of record is not a fossil or living specimen (`basis_of_record` replaces this with an explicit list);
- the year is within `min_year` and `max_year`, when given.

Spatially isolated records are then removed with the quantile rule of CoordinateCleaner's `cc_outl`: a cell is dropped when its mean distance to all other occupied cells exceeds Q3 + 5 × IQR. Distances are computed between H3 resolution-4 parents. Set `outlier_filter: false` to keep every record.

Records are reduced to one presence per occupied H3 cell at the model resolution, which also thins clustered records. A fit needs at least 5 occupied cells.

### Species names

From the 2026-09 snapshot, GBIF matches names against the Catalogue of Life, which places some species in different genera from older usage. For example, the California red-legged frog is filed as *Amerana draytonii*, so a request for *Rana draytonii* finds no records. If a fit fails with too few occupied cells, look the name up in the GBIF species index (`s3://public-gbif/2026-09/species-h0-index.parquet`), for example by searching on the species epithet.

### Threatened and sensitive species

Data providers often generalise the coordinates of threatened species, for example to a box of about 0.2° (a coordinate uncertainty of about 28 km). The default `max_uncertainty_m` of 1,000 m drops these records. In the 2026-09 snapshot, US records of six California species of conservation concern had 48–95% of records above 1 km uncertainty (median about 28 km for four of them). Two common California species had 17–22% (median about 30 m).

| Species | US records | Uncertainty > 1 km | Median uncertainty |
|---|---|---|---|
| *Phrynosoma blainvillii* | 6,196 | 94.5% | 28.8 km |
| *Spea hammondii* | 2,760 | 68.0% | 28.5 km |
| *Ambystoma californiense* | 1,581 | 65.8% | 28.3 km |
| *Gambelia sila* | 956 | 59.9% | 2.1 km |
| *Thamnophis gigas* | 171 | 50.9% | 28.1 km |
| *Dipodomys ingens* | 288 | 47.6% | 3.1 km |
| *Pseudacris regilla* (common) | 68,023 | 22.3% | 38 m |
| *Sceloporus occidentalis* (common) | 156,956 | 16.6% | 27 m |

A model for such a species is fit to the minority of records with precise coordinates. That is the right choice at resolution 8, but it can leave too few records, or a sample biased towards the datasets that publish exact locations. Raising `max_uncertainty_m` keeps more records at the cost of spatial precision; if you do, also fit at a coarser `resolution` (5 or 6). Agency survey data supplied as your own points avoids the problem.

**Your own points.** Occurrences can instead be a GeoJSON FeatureCollection of points (inline or by URL) or a list of `[lon, lat]` pairs. A GeoJSON feature with property `presence: 0`, or a pair in the `absences` list, is a true absence. With 5 or more absence cells, the model uses them in place of random background (presence-absence mode).

## Study area

The study area (the "accessible area" in SDM terms) is the set of H3 cells the model is trained and predicted on.

| `domain.type` | Cells | Source |
|---|---|---|
| `buffer` (default) | rings of H3 resolution-4 cells around the occupied cells, out to `buffer_km` (default 100) | computed from the records |
| `region` | one or more ISO 3166-2 regions, e.g. `"US-CA"` | [Overture Maps divisions](https://s3-west.nrp-nautilus.io/public-overturemaps/stac-collection.json), release 2026-02-18 (CDLA-Permissive-2.0) |
| `country` | one or more ISO 3166-1 alpha-2 codes, e.g. `"MX"` | Overture Maps divisions, release 2026-02-18 |
| `bbox` | cells inside `[west, south, east, north]` | given by the user |

The same Overture regions are used to report `range_by_region`, the suitable area by first-level admin region.

## Covariates

Each covariate group is read at H3 resolution 8 for every cell in the study area. When the model runs at a coarser resolution, values are averaged up to the parent cell.

### climate

[CHELSA 2.1](https://s3-west.nrp-nautilus.io/public-bioclimate/chelsa-2-1/baseline-1981-2010/stac-collection.json) bioclimatic variables for the observed 1981–2010 baseline. Karger et al. 2017, [10.1038/sdata.2017.122](https://doi.org/10.1038/sdata.2017.122); [CHELSA terms of use](https://chelsa-climate.org/terms-of-use/).

| Column | Variable |
|---|---|
| `bio1` | annual mean temperature (°C) |
| `bio4` | temperature seasonality (standard deviation × 100) |
| `bio5` | maximum temperature of the warmest month (°C) |
| `bio6` | minimum temperature of the coldest month (°C) |
| `bio12` | annual precipitation (mm) |
| `bio15` | precipitation seasonality (coefficient of variation, %) |
| `bio17` | precipitation of the driest quarter (mm) |

These seven are a subset of the 19 standard bioclim variables, chosen to cover temperature and precipitation means, extremes and seasonality. The CHELSA source encodes annual precipitation with a ceiling of 6,553.5 mm; cells at that value are excluded.

### elevation

[Copernicus DEM GLO-90](https://s3-west.nrp-nautilus.io/public-dem/copernicus-glo90/stac-collection.json), a global 90 m digital surface model ([10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65); [license](https://spacedata.copernicus.eu/documents/20126/0/CSCDA_ESA_Mission-specific+Annex.pdf)).

| Column | Variable |
|---|---|
| `elevation` | mean elevation of the cell (m) |
| `roughness` | standard deviation of the H3 resolution-9 cell means within each resolution-8 cell (m), a measure of terrain ruggedness |

### landcover

[Copernicus Global Land Cover 100 m, 2019 (CGLS-LC100)](https://s3-west.nrp-nautilus.io/public-land-cover/cgls-lc100-2019/stac-collection.json), CC-BY-4.0. Each column is the fraction of the cell's classified area in a group of land-cover classes; no-data classes (0 and 255) are left out of the denominator.

| Column | CGLS-LC100 classes |
|---|---|
| `lc_forest` | 111–126 (closed and open forest types) |
| `lc_shrub` | 20 |
| `lc_herbaceous` | 30 |
| `lc_cropland` | 40 |
| `lc_urban` | 50 |
| `lc_bare` | 60, 70, 100 (bare or sparse vegetation, snow and ice, moss and lichen) |
| `lc_wetland` | 90 |
| `lc_water` | 80, 200 (permanent water, open sea) |

### human_modification

[Global Human Modification (gHM)](https://s3-west.nrp-nautilus.io/public-human-modification/stac-collection.json): a 0–1 index of cumulative human modification of land at about 1 km, circa 2016, combining settlement, agriculture, transport, mining and energy, and electrical infrastructure. Kennedy, Oakleaf, Theobald et al. 2019 ([data, 10.6084/m9.figshare.7283087](https://doi.org/10.6084/m9.figshare.7283087)), CC-BY-4.0. Column: `ghm`.

### How the groups are combined

- The first group in the list defines which cells are in the model. With the default order this is `climate`, which covers land cells only.
- The other groups are joined to those cells. A missing land-cover fraction is set to 0; any other missing value is set to the column's median over the study area.
- Cells still missing a value after that are dropped.

### Choosing groups

Pass `covariates` as a list of group names to use a subset, for example `["climate", "elevation"]` for a climate-and-terrain model. Leaving out `landcover` and `human_modification` gives a model that responds to climate and terrain alone, which is often preferred for projections, since those two groups are held at present-day values in the future scenario.

There is no screening for correlated covariates. Several of the climate variables are correlated, which matters more for interpreting individual responses than for predictive accuracy.

### Your own covariates

Any H3-indexed parquet that the server can read from the catalog's object store can be added as a covariate. Add an object to the `covariates` list:

```json
{
  "name": "soil",
  "path": "s3://public-example/soil/hex/h0=*/data_0.parquet",
  "columns": ["clay", "ph"],
  "h3_column": "h9",
  "agg": "avg"
}
```

| Field | Meaning |
|---|---|
| `path` | parquet path or glob. A path containing `h0=` is read as hive-partitioned. |
| `columns` | numeric columns to use as predictors |
| `h3_column` | the H3 index column, at resolution 8 or finer (default `h8`). Values are aggregated to resolution 8 and then to the model resolution. |
| `agg` | how finer cells are combined: `avg` (default), `max`, `min` or `sum` |
| `name` | a label for the group in the results (default `custom`) |

The server reads S3 anonymously, so the layer must be public.

## Future climate

`project_to` re-runs the fitted models with future CHELSA 2.1 climate in place of the 1981–2010 baseline. Nine scenarios are available:

| | 2011–2040 | 2041–2070 | 2071–2100 |
|---|---|---|---|
| SSP1-2.6 | `chelsa-2-1-ssp126-2011-2040` | `chelsa-2-1-ssp126-2041-2070` | `chelsa-2-1-ssp126-2071-2100` |
| SSP3-7.0 | `chelsa-2-1-ssp370-2011-2040` | `chelsa-2-1-ssp370-2041-2070` | `chelsa-2-1-ssp370-2071-2100` |
| SSP5-8.5 | `chelsa-2-1-ssp585-2011-2040` | `chelsa-2-1-ssp585-2041-2070` | `chelsa-2-1-ssp585-2071-2100` |

- **Which values are used.** Each [future collection](https://s3-west.nrp-nautilus.io/public-bioclimate/stac-collection.json) holds the values from the five global climate models CHELSA provides, plus their median, minimum and maximum. hexsdm uses the median of the five models.
- **What is held fixed.** Only the climate group changes. Elevation, land cover and human modification keep their present-day values, so a projection shows the effect of climate change alone.
- **Extrapolation.** The projection reports `cells_extrapolated`, the number of cells where at least one covariate falls outside the range seen in training. Predictions in those cells are extrapolations and are less reliable. This is a count of out-of-range covariates, not a full MESS analysis.

## Background and sampling bias

Presence-only models compare presences with background cells drawn from the study area. By default 20,000 cells are drawn uniformly (`background.n` changes the number).

GBIF records are concentrated where people look, near roads, cities and protected areas. Setting `background.bias` to `"target_group"` draws the background in proportion to the number of GBIF records of the whole taxonomic group in each cell, so the background shares the sampling bias of the presences. The group is the species' class by default (for example Amphibia); `target_rank` and `target_name` choose a different group. The first note in each result states which background was used.
