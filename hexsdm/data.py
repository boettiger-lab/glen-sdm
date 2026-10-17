"""Data assembly: occurrences, study domain, background, and covariates on an H3 grid.

Every model in this package works on H3 cells. An occurrence becomes "presence in
the cell that contains it", a covariate becomes "the cell's mean value", and the
study area is a set of cells. All catalog layers are already H3-indexed parquet
(res 8 or 9), so the heavy lifting is DuckDB joins streamed from S3.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field

import duckdb
import h3
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

S3_ENDPOINT = os.environ.get("S3_DEFAULT_ENDPOINT", "s3-west.nrp-nautilus.io")
THREADS = int(os.environ.get("SDM_DUCKDB_THREADS", "32"))


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    for ext in ("httpfs", "h3"):
        try:
            con.sql(f"LOAD {ext}")
        except Exception:
            con.sql(f"INSTALL {ext}" + (" FROM community" if ext == "h3" else ""))
            con.sql(f"LOAD {ext}")
    con.sql(f"SET THREADS={THREADS}")
    con.sql("SET enable_progress_bar=false")
    if os.environ.get("DUCKDB_MEMORY_LIMIT"):
        con.sql(f"SET memory_limit='{os.environ['DUCKDB_MEMORY_LIMIT']}'")
    con.sql("SET temp_directory='/tmp'")
    con.sql("SET preserve_insertion_order=false")
    con.sql("SET enable_object_cache=true")
    con.sql("SET enable_http_metadata_cache=true")
    use_ssl = "false" if S3_ENDPOINT.startswith("rook") else "true"
    con.sql(
        f"CREATE SECRET (TYPE S3, ENDPOINT '{S3_ENDPOINT}', URL_STYLE 'path', "
        f"USE_SSL {use_ssl}, KEY_ID '', SECRET '')"
    )
    return con


def q(s: str) -> str:
    """Quote a string literal for SQL."""
    return "'" + str(s).replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Catalog paths (from the STAC catalog served by mcp-data-server)
# ---------------------------------------------------------------------------

GBIF_HEX = "s3://public-gbif/2026-09/hex/h0=*/data_0.parquet"
GBIF_SPECIES_INDEX = "s3://public-gbif/2026-09/species-h0-index.parquet"
CHELSA = "s3://public-bioclimate/chelsa-2-1/{period}/hex/h0=*/data_0.parquet"
DEM = "s3://public-dem/copernicus-glo90/hex/h0=*/data_0.parquet"
LANDCOVER = "s3://public-land-cover/cgls-lc100-2019/hex-fractions/h0=*/data_0.parquet"
GHM = "s3://public-human-modification/ghm/hex/h0=*/data_0.parquet"
REGIONS = "s3://public-overturemaps/2026-02-18.0/regions/hex/h0=*/data_0.parquet"
COUNTRIES = "s3://public-overturemaps/2026-02-18.0/countries/hex/h0=*/data_0.parquet"

OBIS_HEX = "s3://public-obis/2026-09-09/hex/h0=*/data_0.parquet"
BIO_ORACLE = "s3://public-bio-oracle/{layer}/hex/h0=*/data_0.parquet"

BIO_VARS = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio15", "bio17"]
BIO_DESC = {
    "bio1": "annual mean temperature (C)",
    "bio4": "temperature seasonality (sd x100)",
    "bio5": "max temperature of warmest month (C)",
    "bio6": "min temperature of coldest month (C)",
    "bio12": "annual precipitation (mm)",
    "bio15": "precipitation seasonality (CV %)",
    "bio17": "precipitation of driest quarter (mm)",
}
FUTURE_SCENARIOS = [
    f"{ssp}-{per}"
    for ssp in ("ssp126", "ssp370", "ssp585")
    for per in ("2011-2040", "2041-2070", "2071-2100")
]
LC_GROUPS = {
    "lc_forest": "lc_class BETWEEN 111 AND 126",
    "lc_shrub": "lc_class = 20",
    "lc_herbaceous": "lc_class = 30",
    "lc_cropland": "lc_class = 40",
    "lc_urban": "lc_class = 50",
    "lc_bare": "lc_class IN (60, 70, 100)",
    "lc_wetland": "lc_class = 90",
    "lc_water": "lc_class IN (80, 200)",
}

# Each layer group is SQL producing (h8, <columns>) restricted to the domain.
# `{dom}` is a filter expression over an h8 column; `{h0}` the partition filter.
COVARIATE_LAYERS = {
    "climate": {
        "description": "CHELSA 2.1 bioclim 1981-2010 (" + ", ".join(BIO_VARS) + "); "
        "the only group that can be swapped for a future scenario with project_to.",
        "columns": BIO_VARS,
        "source": "chelsa-2-1-baseline-1981-2010",
    },
    "elevation": {
        "description": "Copernicus GLO-90 DEM: mean elevation (m) and roughness "
        "(sd of res-9 cell means within the res-8 cell).",
        "columns": ["elevation", "roughness"],
        "source": "copernicus-glo90",
    },
    "landcover": {
        "description": "Copernicus CGLS-LC100 2019 class fractions per cell: "
        + ", ".join(LC_GROUPS) + ".",
        "columns": list(LC_GROUPS),
        "source": "cgls-lc100-2019",
    },
    "human_modification": {
        "description": "Global Human Modification index (Theobald et al.), 0-1.",
        "columns": ["ghm"],
        "source": "global-human-modification",
    },
}


# Marine predictors: Bio-ORACLE v3.0, 2010s baseline, native at H3 res 6 (one
# ~5.5 km source pixel per cell). Each variable is its own parquet; a group joins
# them on h6. (column alias, Bio-ORACLE layer path, source column, SQL transform)
MARINE_GROUPS = {
    "ocean": [
        ("thetao", "depthmean/thetao-mean", "thetao_mean", None),
        ("so", "depthmean/so-mean", "so_mean", None),
        ("o2", "depthmean/o2-mean", "o2_mean", None),
        ("phyc", "depthmean/phyc-mean", "phyc_mean", None),
        ("ph", "depthmean/ph-mean", "ph_mean", None),
        ("no3", "depthmean/no3-mean", "no3_mean", None),
    ],
    "nutrients_extra": [
        ("po4", "depthmean/po4-mean", "po4_mean", None),
        ("si", "depthmean/si-mean", "si_mean", None),
        ("dfe", "depthmean/dfe-mean", "dfe_mean", None),
    ],
    "currents": [
        ("sws", "depthmean/sws-mean", "sws_mean", None),
        ("swd_sin", "depthmean/swd-mean-sin", "swd_sin", None),
        ("swd_cos", "depthmean/swd-mean-cos", "swd_cos", None),
    ],
    "seafloor": [
        ("depth", "terrain/bathymetry-mean", "bathymetry_mean", "-{c}"),   # metres, positive down
        ("slope", "terrain/slope", "slope", None),
        ("aspect_sin", "terrain/aspect-sin", "aspect_sin", None),
        ("aspect_cos", "terrain/aspect-cos", "aspect_cos", None),
        ("tpi", "terrain/tpi", "topographic_position_index", None),
        ("tri", "terrain/tri", "terrain_ruggedness_index", None),
    ],
}
MARINE_DESC = {
    "ocean": "Bio-ORACLE v3.0 depth-mean temperature, salinity, dissolved oxygen, phytoplankton carbon, pH and nitrate (2010s baseline).",
    "nutrients_extra": "Bio-ORACLE v3.0 depth-mean phosphate, silicate and dissolved iron (2010s baseline).",
    "currents": "Bio-ORACLE v3.0 depth-mean current speed and direction (sine and cosine of the bearing).",
    "seafloor": "Bio-ORACLE v3.0 benthic terrain: depth (m, positive down), slope, aspect (sine, cosine), topographic position and ruggedness indices.",
}
MARINE_LAYERS = {
    g: {"description": MARINE_DESC[g], "columns": [v[0] for v in vs], "source": "bio-oracle-v3.0",
        "h3_column": "h6", "realm": "marine"}
    for g, vs in MARINE_GROUPS.items()
}
MARINE_RES = 6   # Bio-ORACLE's native resolution; finer cells would only repeat values


def _marine_layer_sql(name: str) -> str:
    """SQL for one Bio-ORACLE group: h6 + its columns, joined across variables,
    restricted to the domain (temp tables dom4, dom0)."""
    parts = MARINE_GROUPS[name]
    sel, joins = [], []
    for i, (alias, layer, col, expr) in enumerate(parts):
        t = f"t{i}"
        v = f"{t}.{col}"
        sel.append(f"({expr.format(c=v) if expr else v})::DOUBLE AS {alias}")
        src = (f"(SELECT h6, {col} FROM read_parquet('{BIO_ORACLE.format(layer=layer)}', hive_partitioning=true) "
               f"WHERE h0 IN (SELECT h0 FROM dom0) AND h3_cell_to_parent(h6, 4) IN (SELECT h4 FROM dom4)) {t}")
        joins.append(src if i == 0 else f"JOIN {src} USING (h6)")
    return f"SELECT h6, {', '.join(sel)} FROM {' '.join(joins)}"


def layer_info(name: str) -> dict:
    """Metadata for a built-in covariate group of either realm."""
    if name in COVARIATE_LAYERS:
        return {**COVARIATE_LAYERS[name], "h3_column": "h8", "realm": "terrestrial"}
    if name in MARINE_LAYERS:
        return MARINE_LAYERS[name]
    raise ValueError(f"unknown covariate layer {name!r}; choose from {list(COVARIATE_LAYERS) + list(MARINE_LAYERS)}")


def _layer_sql(name: str, period: str = "baseline-1981-2010") -> str:
    """SQL for one layer group, returning h8 + columns, filtered to the domain.

    Uses temp tables `dom4` (res-4 domain cells) and `dom0` (h0 partitions)."""
    h0f = "h0 IN (SELECT h0 FROM dom0)"
    if name == "climate":
        path = CHELSA.format(period=period)
        suffix = "" if period.startswith("baseline") else "_median"
        cols = ", ".join(f"{v}{suffix} AS {v}" for v in BIO_VARS)
        return (
            f"SELECT h8, {cols} FROM read_parquet('{path}', hive_partitioning=true) "
            f"WHERE {h0f} AND h4 IN (SELECT h4 FROM dom4) AND bio12{suffix} < 6553.5"
        )
    if name == "elevation":
        return (
            f"SELECT h8, avg(elevation) AS elevation, coalesce(stddev_pop(elevation), 0) AS roughness "
            f"FROM read_parquet('{DEM}', hive_partitioning=true) "
            f"WHERE {h0f} AND h3_cell_to_parent(h8, 4) IN (SELECT h4 FROM dom4) GROUP BY h8"
        )
    if name == "landcover":
        cols = ", ".join(
            f"coalesce(sum(frac) FILTER ({cond}), 0) / nullif(sum(frac) FILTER (lc_class NOT IN (0, 255)), 0) AS {k}"
            for k, cond in LC_GROUPS.items()
        )
        return (
            f"SELECT h8, {cols} FROM read_parquet('{LANDCOVER}', hive_partitioning=true) "
            f"WHERE {h0f} AND h3_cell_to_parent(h8, 4) IN (SELECT h4 FROM dom4) GROUP BY h8"
        )
    if name == "human_modification":
        return (
            f"SELECT h8, ghm FROM read_parquet('{GHM}', hive_partitioning=true) "
            f"WHERE {h0f} AND h3_cell_to_parent(h8, 4) IN (SELECT h4 FROM dom4)"
        )
    raise ValueError(f"unknown covariate layer {name!r}; choose from {list(COVARIATE_LAYERS)}")


def _custom_layer_sql(spec: dict) -> tuple[str, list[str]]:
    """A user-supplied hex parquet: {"path", "columns", "h3_column" (default h8), "agg" (avg)}.

    The H3 column may be at any resolution >= 8; values are averaged up to h8."""
    path = spec["path"]
    cols = spec["columns"]
    hcol = spec.get("h3_column", "h8")
    agg = spec.get("agg", "avg")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", hcol) or agg not in ("avg", "max", "min", "sum"):
        raise ValueError("bad custom covariate spec")
    for c in cols:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c):
            raise ValueError(f"bad column name {c!r}")
    sel = ", ".join(f"{agg}({c})::DOUBLE AS {c}" for c in cols)
    hive = "true" if "h0=" in path else "false"
    sql = (
        f"SELECT h3_cell_to_parent({hcol}, 8) AS h8, {sel} "
        f"FROM read_parquet({q(path)}, hive_partitioning={hive}) "
        f"WHERE h3_cell_to_parent({hcol}, 4) IN (SELECT h4 FROM dom4) GROUP BY 1"
    )
    return sql, cols


# ---------------------------------------------------------------------------
# Occurrences
# ---------------------------------------------------------------------------

GBIF_BAD_ISSUES = "['ZERO_COORDINATE','COORDINATE_INVALID','COORDINATE_OUT_OF_RANGE','COUNTRY_COORDINATE_MISMATCH']"


def _species_suggestions(con, name: str, limit: int = 5) -> list[str]:
    """Names in the GBIF species index sharing the epithet or the genus of `name`."""
    parts = name.split()
    if len(parts) < 2:
        return []
    genus, epithet = parts[0], parts[-1]
    rows = con.sql(f"""
      SELECT DISTINCT species FROM read_parquet('{GBIF_SPECIES_INDEX}')
      WHERE species LIKE {q('% ' + epithet)} OR species LIKE {q(genus + ' %')}
      ORDER BY (species LIKE {q('% ' + epithet)}) DESC, species LIMIT {int(limit)}
    """).fetchall()
    return [r[0] for r in rows]


def gbif_occurrences(con, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Occurrence counts per h8 cell for a GBIF species, with standard quality filters.

    spec: {"species": "Genus species"} or {"specieskey": "..."}; optional
    min_year, max_year, max_uncertainty_m (default 1000), basis_of_record (list).
    Returns (per-cell counts, info) where info reports how many records the
    coordinate-uncertainty filter removed.
    """
    if "specieskey" in spec:
        key = f"specieskey = {q(spec['specieskey'])}"
    elif "species" in spec:
        key = f"species = {q(spec['species'])}"
    else:
        raise ValueError("gbif occurrences need 'species' or 'specieskey'")
    hit = con.sql(f"SELECT count(*) FROM read_parquet('{GBIF_SPECIES_INDEX}') WHERE {key}").fetchone()[0]
    if not hit:
        name = spec.get("species") or spec.get("specieskey")
        sugg = _species_suggestions(con, spec["species"]) if "species" in spec else []
        msg = (f"{name!r} is not a species name in the GBIF 2026-09 snapshot, which uses the "
               "Catalogue of Life backbone (some species have new genus names).")
        if sugg:
            msg += " Similar names: " + ", ".join(sugg) + "."
        raise ValueError(msg)
    where = [key, f"NOT list_has_any(issue, {GBIF_BAD_ISSUES})",
             "coalesce(occurrencestatus, 'PRESENT') <> 'ABSENT'"]
    unc = float(spec.get("max_uncertainty_m", 1000))
    unc_ok = f"(coordinateuncertaintyinmeters IS NULL OR coordinateuncertaintyinmeters <= {unc})"
    if spec.get("min_year"):
        where.append(f"year >= {int(spec['min_year'])}")
    if spec.get("max_year"):
        where.append(f"year <= {int(spec['max_year'])}")
    bor = spec.get("basis_of_record")
    if bor:
        where.append("basisofrecord IN (" + ", ".join(q(b) for b in bor) + ")")
    else:
        where.append("basisofrecord NOT IN ('FOSSIL_SPECIMEN', 'LIVING_SPECIMEN')")
    sql = f"""
      SELECT h8, (count(*) FILTER ({unc_ok}))::INT AS n, count(*)::INT AS n_all,
             any_value(class) AS class, any_value("order") AS "order"
      FROM read_parquet('{GBIF_HEX}', hive_partitioning=true)
      WHERE h0 IN (SELECT h0 FROM read_parquet('{GBIF_SPECIES_INDEX}') WHERE {key})
        AND {' AND '.join(where)}
      GROUP BY h8
    """
    df = con.sql(sql).df()
    n_all, n_kept = int(df.n_all.sum()), int(df.n.sum())
    info = {"records_before_uncertainty_filter": n_all,
            "dropped_by_uncertainty_filter": n_all - n_kept,
            "max_uncertainty_m": unc}
    if n_all and (n_all - n_kept) / n_all > 0.25:
        info["uncertainty_note"] = (
            f"{100 * (n_all - n_kept) / n_all:.0f}% of quality-filtered records have coordinate "
            f"uncertainty above {unc:g} m and were dropped. This is common for sensitive species, "
            "whose public coordinates are generalised (e.g. iNaturalist obscured locations, ~28 km). "
            "Raising max_uncertainty_m keeps them, but only at a coarser resolution (5 or 6) do "
            "such records place a species in the right cell.")
    return df[df.n > 0].reset_index(drop=True), info


def _points_from_geojson(gj) -> list[tuple[float, float, int]]:
    if isinstance(gj, str):
        if gj.startswith("http"):
            with urllib.request.urlopen(gj, timeout=60) as r:
                gj = json.load(r)
        else:
            gj = json.loads(gj)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    out = []
    for f in feats:
        g = f.get("geometry") or {}
        props = f.get("properties") or {}
        pres = int(props.get("presence", 1))
        coords = [g["coordinates"]] if g.get("type") == "Point" else (
            g["coordinates"] if g.get("type") == "MultiPoint" else [])
        for lon, lat, *_ in coords:
            out.append((lon, lat, pres))
    return out


def user_occurrences(spec: dict) -> pd.DataFrame:
    """User-supplied points -> per-h8 counts. Accepts:
      {"geojson": FeatureCollection | URL}  (property presence=0 marks an absence)
      {"points": [[lon, lat], ...], "absences": [[lon, lat], ...]}
    """
    pts: list[tuple[float, float, int]] = []
    if "geojson" in spec:
        pts += _points_from_geojson(spec["geojson"])
    for lon, lat, *_ in spec.get("points", []):
        pts.append((lon, lat, 1))
    for lon, lat, *_ in spec.get("absences", []):
        pts.append((lon, lat, 0))
    if not pts:
        raise ValueError("no points found in user occurrences")
    rows = [(h3.str_to_int(h3.latlng_to_cell(lat, lon, 8)), p) for lon, lat, p in pts]
    df = pd.DataFrame(rows, columns=["h8", "presence"])
    df["h8"] = df["h8"].astype("uint64")
    pres = df[df.presence == 1].groupby("h8").size().rename("n").reset_index()
    absn = df[df.presence == 0].groupby("h8").size().rename("n").reset_index()
    pres.attrs["absences"] = absn
    return pres


def remove_spatial_outliers(pres: pd.DataFrame, multiplier: float = 5.0, max_points: int = 3000,
                            seed: int = 1) -> tuple[pd.DataFrame, int]:
    """Drop isolated records, as CoordinateCleaner::cc_outl(method = "quantile"):
    a cell is an outlier if its mean great-circle distance to all other occupied
    cells exceeds Q3 + multiplier * IQR of those mean distances. Distances are
    computed between res-4 parents (and a sample of them for large ranges)."""
    if len(pres) < 8:
        return pres, 0
    p4 = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), 4)) for c in pres.h8], dtype="uint64")
    u = np.unique(p4)
    ref = u if len(u) <= max_points else np.random.default_rng(seed).choice(u, max_points, replace=False)
    ll = np.radians([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in u])
    lr = np.radians([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in ref])
    dlat = ll[:, None, 0] - lr[None, :, 0]
    dlng = ll[:, None, 1] - lr[None, :, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(ll[:, None, 0]) * np.cos(lr[None, :, 0]) * np.sin(dlng / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    md = d.sum(1) / max(len(ref) - 1, 1)
    q1, q3 = np.quantile(md, [0.25, 0.75])
    bad = set(u[md > q3 + multiplier * (q3 - q1)].tolist())
    keep = np.array([c not in bad for c in p4.tolist()])
    return pres[keep].reset_index(drop=True), int((~keep).sum())


def _mode(col: pd.Series):
    """Most common non-null value, or None (WoRMS leaves some ranks empty, e.g. no class
    for sea turtles)."""
    m = col.dropna().mode()
    return m.iat[0] if len(m) else None


def obis_occurrences(con, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Occurrence counts per h8 cell for an OBIS species (records already pass OBIS
    quality control, and absences are excluded at ingest).

    spec: {"species": "Genus species"} or {"aphiaid": 137205}; optional min_year,
    max_year, max_uncertainty_m (default 1000), depth_range [min_m, max_m]
    (keeps records whose reported depth interval overlaps it, or that report none).
    """
    if "aphiaid" in spec:
        key = f"aphiaid = {int(spec['aphiaid'])}"
    elif "species" in spec:
        key = f"species = {q(spec['species'])}"
    else:
        raise ValueError("obis occurrences need 'species' or 'aphiaid'")
    where = [key]
    unc = float(spec.get("max_uncertainty_m", 1000))
    unc_ok = f"coordinateuncertaintyinmeters IS NULL OR coordinateuncertaintyinmeters <= {unc}"
    if spec.get("min_year"):
        where.append(f"date_year >= {int(spec['min_year'])}")
    if spec.get("max_year"):
        where.append(f"date_year <= {int(spec['max_year'])}")
    dr = spec.get("depth_range")
    if dr:
        lo, hi = float(dr[0]), float(dr[1])
        where.append(f"(maximumdepthinmeters IS NULL OR maximumdepthinmeters >= {lo})")
        where.append(f"(minimumdepthinmeters IS NULL OR minimumdepthinmeters <= {hi})")
    df = con.sql(f"""
      SELECT h8, (count(*) FILTER ({unc_ok}))::INT AS n, count(*)::INT AS n_all,
             any_value(class) AS class, any_value("order") AS "order"
      FROM read_parquet('{OBIS_HEX}', hive_partitioning=true)
      WHERE {' AND '.join(where)}
      GROUP BY h8
    """).df()
    if len(df) == 0 and "species" in spec:
        raise ValueError(f"no OBIS records for {spec['species']!r}. OBIS names follow WoRMS, "
                         "which can differ from GBIF; try the WoRMS accepted name or an aphiaid.")
    n_all, n_kept = int(df.n_all.sum()), int(df.n.sum())
    info = {"records_before_uncertainty_filter": n_all,
            "dropped_by_uncertainty_filter": n_all - n_kept, "max_uncertainty_m": unc}
    return df[df.n > 0].reset_index(drop=True), info


def load_occurrences(con, occ: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Return (presences per h8, absences per h8 or None, info)."""
    if "obis" in occ:
        pres, unc_info = obis_occurrences(con, occ["obis"])
        info = {"source": "obis", "records": int(pres.n.sum()) if len(pres) else 0, **unc_info,
                "class": _mode(pres["class"]), "order": _mode(pres["order"])}
        pres = pres[["h8", "n"]]
        if occ["obis"].get("outlier_filter", True):
            pres, dropped = remove_spatial_outliers(pres)
            info["outlier_cells_removed"] = dropped
        return pres, None, info
    if "gbif" in occ:
        pres, unc_info = gbif_occurrences(con, occ["gbif"])
        info = {"source": "gbif", "records": int(pres.n.sum()) if len(pres) else 0, **unc_info,
                "class": pres["class"].mode().iat[0] if len(pres) else None,
                "order": pres["order"].mode().iat[0] if len(pres) else None}
        pres = pres[["h8", "n"]]
        if occ["gbif"].get("outlier_filter", True):
            pres, dropped = remove_spatial_outliers(pres)
            info["outlier_cells_removed"] = dropped
        return pres, None, info
    pres = user_occurrences(occ)
    absn = pres.attrs.get("absences")
    info = {"source": "user", "records": int(pres.n.sum()),
            "absence_records": int(absn.n.sum()) if absn is not None and len(absn) else 0}
    return pres, (absn if absn is not None and len(absn) else None), info


# ---------------------------------------------------------------------------
# Domain (the "accessible area" M) and resolution
# ---------------------------------------------------------------------------

def _res4_from_h8(h8s) -> set[int]:
    return {h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), 4)) for c in h8s}


def build_domain(con, pres: pd.DataFrame, domain: dict | None) -> dict:
    """Create temp tables dom4 (res-4 cells) + dom0 (partitions) + optional dom8 mask.

    domain options:
      {"type": "buffer", "buffer_km": 100}     (default) k-ring buffer around occurrences
      {"type": "region", "region": "US-CA"}    Overture ISO 3166-2 region(s)
      {"type": "country", "country": "MX"}     Overture ISO country code(s)
      {"type": "bbox", "bbox": [w, s, e, n]}
    """
    domain = domain or {"type": "buffer"}
    kind = domain.get("type", "buffer")
    mask_sql = None
    if kind == "buffer":
        km = float(domain.get("buffer_km", 100))
        edge = h3.average_hexagon_edge_length(4, unit="km")  # ~26 km
        k = max(1, math.ceil(km / (edge * math.sqrt(3))))
        base = _res4_from_h8(pres.h8)
        cells = set()
        for c in base:
            cells.update(h3.grid_disk(h3.int_to_str(c), k))
        dom4 = [h3.str_to_int(c) for c in cells]
    elif kind in ("region", "country"):
        vals = domain.get(kind)
        vals = [vals] if isinstance(vals, str) else list(vals)
        path, col = (REGIONS, "region") if kind == "region" else (COUNTRIES, "country")
        inlist = ", ".join(q(v) for v in vals)
        con.sql(f"""CREATE OR REPLACE TABLE dom8 AS SELECT DISTINCT h8
                   FROM read_parquet('{path}', hive_partitioning=true) WHERE {col} IN ({inlist})""")
        dom4 = [int(r[0]) for r in con.sql(
            "SELECT DISTINCT h3_cell_to_parent(h8, 4) FROM dom8").fetchall()]
        mask_sql = "SELECT h8 FROM dom8"
    elif kind == "bbox":
        w, s, e, n = domain["bbox"]
        poly = h3.LatLngPoly([(s, w), (s, e), (n, e), (n, w)])
        dom4 = [h3.str_to_int(c) for c in h3.h3shape_to_cells_experimental(poly, 4, "overlap")]
        lo, hi = sorted([w, e])
        mask_sql = None
        domain = {**domain, "_bbox": (w, s, e, n)}
    else:
        raise ValueError(f"unknown domain type {kind!r}")
    if not dom4:
        raise ValueError("empty domain")
    con.execute("CREATE OR REPLACE TABLE dom4 AS SELECT unnest(?::UBIGINT[]) AS h4", [dom4])
    con.sql("CREATE OR REPLACE TABLE dom0 AS SELECT DISTINCT h3_cell_to_parent(h4, 0)::BIGINT AS h0 FROM dom4")
    approx_h8 = len(dom4) * 7 ** 4
    return {"type": kind, "n_res4": len(dom4), "approx_h8_cells": approx_h8,
            "mask_sql": mask_sql, "bbox": domain.get("_bbox")}


def choose_resolution(approx_h8_cells: int, requested: int | None, max_cells: int, finest: int = 8) -> int:
    if requested:
        if not 5 <= int(requested) <= finest:
            raise ValueError(f"resolution must be between 5 and {finest}")
        return int(requested)
    res = finest
    while res > 5 and approx_h8_cells / 7 ** (8 - res) > max_cells:
        res -= 1
    return res


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------

def load_covariates(con, layers: list, res: int, dom: dict, period: str = "baseline-1981-2010") -> tuple[pd.DataFrame, list[str]]:
    """Covariate matrix for every domain cell at resolution `res`.

    The climate group (if requested) defines the land cell universe; other groups
    are left-joined (land-cover fractions missing -> 0, others -> column median)."""
    names, sqls, hcols = [], [], []
    for layer in layers:
        if isinstance(layer, dict):
            sql, cols = _custom_layer_sql(layer)
            names.append((layer.get("name", "custom"), cols))
            hcols.append("h8")
        elif layer in MARINE_LAYERS:
            sql = _marine_layer_sql(layer)
            names.append((layer, MARINE_LAYERS[layer]["columns"]))
            hcols.append("h6")
        else:
            sql = _layer_sql(layer, period)
            names.append((layer, COVARIATE_LAYERS[layer]["columns"]))
            hcols.append("h8")
        sqls.append(sql)
    cols_all = [c for _, cols in names for c in cols]

    def mask_for(h):
        if dom.get("mask_sql"):
            return f"WHERE {h} IN ({dom['mask_sql']})"
        if dom.get("bbox"):
            w, s, e, n = dom["bbox"]
            return (f"WHERE h3_cell_to_lat({h}) BETWEEN {s} AND {n} "
                    f"AND h3_cell_to_lng({h}) BETWEEN {w} AND {e}")
        return ""

    def one(args):
        sql, cols, h = args
        sel = ", ".join(f"avg({c})::FLOAT AS {c}" for c in cols)
        cur = con.cursor()  # one connection per layer: S3 reads overlap across layers
        try:
            return cur.sql(f"SELECT h3_cell_to_parent({h}, {res}) AS cell, {sel} "
                           f"FROM ({sql}) {mask_for(h)} GROUP BY 1").df().set_index("cell")
        finally:
            cur.close()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(sqls)) as ex:
        parts = list(ex.map(one, [(sq, cols, h) for sq, (_, cols), h in zip(sqls, names, hcols)]))
    # The first layer defines the cell universe (climate by default), others left join.
    df = parts[0].join(parts[1:], how="left") if len(parts) > 1 else parts[0]
    df = df.reset_index().sort_values("cell", kind="stable")
    for (lname, cols) in names[1:]:
        for c in cols:
            if c.startswith("lc_"):
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna(df[c].median())
    df = df.dropna()
    df["cell"] = df["cell"].astype("uint64")
    return df.reset_index(drop=True), cols_all


def target_group_effort(con, rank: str, name: str, res: int, source: str = "gbif") -> pd.DataFrame:
    """Record counts for a whole taxonomic group per domain cell (sampling effort),
    from GBIF or OBIS."""
    if rank not in ("class", "order", "family", "phylum", "kingdom", "genus"):
        raise ValueError("target_group rank must be a Linnaean rank column")
    if source == "obis":
        df = con.sql(f"""
          SELECT h3_cell_to_parent(h8, {res}) AS cell, count(*)::DOUBLE AS effort
          FROM read_parquet('{OBIS_HEX}', hive_partitioning=true)
          WHERE h0 IN (SELECT h0 FROM dom0) AND h3_cell_to_parent(h8, 4) IN (SELECT h4 FROM dom4)
            AND "{rank}" = {q(name)}
          GROUP BY 1
        """).df()
        df["cell"] = df["cell"].astype("uint64")
        return df
    sql = f"""
      SELECT h3_cell_to_parent(h8, {res}) AS cell, count(*)::DOUBLE AS effort
      FROM read_parquet('{GBIF_HEX}', hive_partitioning=true)
      WHERE h0 IN (SELECT h0 FROM dom0) AND h4 IN (SELECT h4 FROM dom4)
        AND "{rank}" = {q(name)}
        AND NOT list_has_any(issue, {GBIF_BAD_ISSUES})
      GROUP BY 1
    """
    df = con.sql(sql).df()
    df["cell"] = df["cell"].astype("uint64")
    return df


def to_res(h8: pd.Series, res: int) -> np.ndarray:
    if res == 8:
        return h8.astype("uint64").to_numpy()
    return np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), res)) for c in h8],
                    dtype="uint64")


def region_summary(con, cells: np.ndarray, res: int, top: int = 8) -> list[dict]:
    """Area of a cell set by first-level admin region (Overture), largest first.

    Gives the agent named geography taken from the model output, so it does not
    describe where a range lies from prior knowledge."""
    if len(cells) == 0:
        return []
    con.execute("CREATE OR REPLACE TABLE _sel AS SELECT unnest(?::UBIGINT[]) AS cell", [cells.tolist()])
    df = con.sql(f"""
      WITH reg AS (
        SELECT DISTINCT h3_cell_to_parent(h8, {res}) AS cell, name_en, country
        FROM read_parquet('{REGIONS}', hive_partitioning=true)
        WHERE h0 IN (SELECT h0 FROM dom0) AND h3_cell_to_parent(h8, 4) IN (SELECT h4 FROM dom4)
          AND name_en IS NOT NULL
      )
      SELECT name_en AS region, country, count(*) AS n
      FROM _sel JOIN reg USING (cell) GROUP BY ALL ORDER BY n DESC LIMIT {int(top)}
    """).df()
    a = h3.average_hexagon_area(res, unit="km^2")
    tot = len(cells)
    return [{"region": r.region, "country": r.country, "suitable_km2": round(r.n * a),
             "share_of_range": round(r.n / tot, 3)} for r in df.itertuples()]
