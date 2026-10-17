---
title: Marine models (prototype)
nav_order: 4
---

# Marine models (prototype)

Setting `realm: "marine"` fits a model with marine predictors and, by default, OBIS occurrence records. The algorithms, cross-validation, thresholds and outputs are the same as for terrestrial models. This mode is a prototype: its predictors are a first set from the catalog, and the gaps are listed below.

Marine species distribution models are an established practice. AquaMaps (Kaschner et al. 2006) publishes models for tens of thousands of marine species, and Bio-ORACLE (Tyberghein et al. 2012; Assis et al. 2024) was built for them.

## Data

**Occurrences.** `{"obis": {"species": "Posidonia oceanica"}}` reads the [OBIS](https://s3-west.nrp-nautilus.io/public-obis/stac-collection.json) records in the catalog: 228 million records that pass OBIS quality control, indexed to H3 resolution 8, with absences excluded. Names follow the World Register of Marine Species (WoRMS), so they can differ from GBIF names; an `aphiaid` can be given instead. Options:

| Key | Meaning |
|---|---|
| `min_year`, `max_year` | limit the record years |
| `max_uncertainty_m` | maximum coordinate uncertainty (default 1000; records without one are kept) |
| `depth_range` | `[min_m, max_m]`: keep records whose reported depth interval overlaps it, or that report no depth |
| `outlier_filter` | drop spatially isolated records (default true) |

GBIF records and uploaded points also work in marine mode. OBIS data are CC-BY-NC.

**Predictors.** [Bio-ORACLE v3.0](https://s3-west.nrp-nautilus.io/public-bio-oracle/stac-collection.json), 2010s baseline, at H3 resolution 6, which matches Bio-ORACLE's 0.05° (about 5.5 km) grid. Models run at resolution 6, or 5 for very large study areas.

| Group | Columns | Default |
|---|---|---|
| `ocean` | `thetao` temperature, `so` salinity, `o2` dissolved oxygen, `phyc` phytoplankton carbon, `ph`, `no3` nitrate (depth-mean) | yes |
| `currents` | `sws` current speed, `swd_sin`, `swd_cos` current direction (depth-mean) | yes |
| `seafloor` | `depth` (m, positive down), `slope`, `aspect_sin`, `aspect_cos`, `tpi` topographic position, `tri` ruggedness | yes |
| `nutrients_extra` | `po4` phosphate, `si` silicate, `dfe` dissolved iron (depth-mean) | no |

The ocean cells of these layers define the model's cells, so land cells are excluded.

**Study area.** `buffer` (default) or `bbox`; country and region boundaries are land, so they are not offered. `domain.depth_range: [min_m, max_m]` keeps only cells whose mean depth lies in the range, which also limits the background to plausible depths. For example, `[0, 50]` suits a shallow seagrass. It needs the `seafloor` group.

**Background.** Uniform, or `target_group`, which uses OBIS records of the same class, or of the same order when WoRMS gives no class (as for sea turtles).

## Checks

A marine fit is refused when under half of the occupied cells in the study area have marine predictors: a terrestrial or freshwater species, or a `depth_range` that excludes most records. `allow_onshore: true` overrides this. Between 50% and 80%, the fit runs with a note. The share is reported as `occurrences.ocean_share`.

## Not yet available

- **Future projections.** The catalog has no Bio-ORACLE future layers yet ([data-workflows#755](https://github.com/boettiger-lab/data-workflows/issues/755)), so `project_to` is refused.
- **Surface and bottom predictors.** Only the depth-mean realm (the water-column average) and seafloor terrain are in the catalog. Most marine models use surface layers for pelagic species and bottom layers for benthic ones, together with seasonal minima and maxima; these are requested in [data-workflows#798](https://github.com/boettiger-lab/data-workflows/issues/798).
- **Range by region.** `range_by_region` is empty for marine models, because the admin regions it uses are land. Marine ecoregions are a candidate.

## Interpreting results

- **Depth.** One 2-D set of predictors fits a 3-D habitat only approximately, and depth-mean values suit surface and bottom dwellers least well.
- **Mobile species.** For highly mobile species such as sea turtles, cetaceans and tunas, a climatological model shows broad habitat, not where animals are at a given time. Dynamic models matched to the date of each record are the tool for that.
- **Sampling bias.** Records concentrate along coasts, ship tracks and survey cruises. Target-group background corrects for this only when the target group is much broader than the species. Mediterranean OBIS turtle records are almost all loggerheads, so a turtle target group samples background where the loggerhead itself was recorded, and its AUC falls to about 0.55.

## Tests

| Species | Request | Result |
|---|---|---|
| *Posidonia oceanica* (seagrass) | OBIS, bbox [-6, 30, 37, 46], depth 0–50 m | 8,526 cells, 775 presence cells, AUC 0.84, 64% of occupied cells with marine predictors in the depth range; top predictors salinity, ruggedness, pH, slope |
| *Caretta caretta* (loggerhead turtle) | OBIS, same bbox, uniform background | 76,766 cells, 10,210 presence cells, AUC 0.77; top predictors salinity, nitrate, temperature |
| *Caretta caretta* | same, target-group background (order Testudines) | AUC 0.55 (see sampling bias above) |
| *Lynx pardinus* (Iberian lynx) | GBIF, `realm: marine` | refused: 5% of occupied cells have marine predictors |

## References

- Assis, J. et al. (2024). Bio-ORACLE v3.0. Pushing marine data layers to the CMIP6 Earth system models of climate change research. *Global Ecology and Biogeography*. [doi:10.1111/geb.13813](https://doi.org/10.1111/geb.13813)
- Tyberghein, L. et al. (2012). Bio-ORACLE: a global environmental dataset for marine species distribution modelling. *Global Ecology and Biogeography*, 21, 272–281. [doi:10.1111/j.1466-8238.2011.00656.x](https://doi.org/10.1111/j.1466-8238.2011.00656.x)
- Kaschner, K. et al. (2006). Mapping world-wide distributions of marine mammal species using a relative environmental suitability (RES) model. *Marine Ecology Progress Series*, 316, 285–310.
