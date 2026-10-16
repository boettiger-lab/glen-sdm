"""End-to-end SDM run: occurrences + covariates -> fitted models -> per-cell predictions + polygons."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import data, evaluate, outputs
from .models import ALGORITHMS

VERSION = "0.5.0"  # part of the cache key: bump when results would change
DEFAULTS = {
    "algorithms": ["rf", "glm"],
    "covariates": ["climate", "elevation", "landcover", "human_modification"],
    "domain": {"type": "buffer", "buffer_km": 100},
    "resolution": None,
    "background": {"n": 20000, "bias": None},
    "cv_folds": 5,
    "threshold": "maxSSS",
    "ensemble": True,
    "project_to": None,
    "max_cells": 2_000_000,
    "seed": 1,
}


# Every predictor is terrestrial (CHELSA, which covers land only, defines the cell
# universe), so presences at sea have no covariates and are dropped. Below this
# share of in-domain presence cells on land, the model would describe only a
# coastal fringe, and the fit is refused unless the caller passes allow_offshore.
MIN_LAND_SHARE = 0.5
WARN_LAND_SHARE = 0.8


def land_share(con, pres8, env_cells, res: int, dom: dict) -> tuple[int, int]:
    """(presence cells at `res` inside the study area, how many of them have covariates)."""
    con.register("_pres8", pres8)
    if dom.get("mask_sql"):
        inside = f"h8 IN ({dom['mask_sql']})"
    elif dom.get("bbox"):
        w, s_, e, n = dom["bbox"]
        inside = (f"h3_cell_to_lat(h8) BETWEEN {s_} AND {n} "
                  f"AND h3_cell_to_lng(h8) BETWEEN {w} AND {e}")
    else:
        inside = "h3_cell_to_parent(h8, 4) IN (SELECT h4 FROM dom4)"
    cells = con.sql(f"SELECT DISTINCT h3_cell_to_parent(h8, {res})::UBIGINT AS cell "
                    f"FROM _pres8 WHERE {inside}").df()["cell"].to_numpy()
    return len(cells), int(np.isin(cells, env_cells).sum())


def normalize(spec: dict) -> dict:
    s = {**DEFAULTS, **{k: v for k, v in spec.items() if v is not None}}
    s["background"] = {**DEFAULTS["background"], **(spec.get("background") or {})}
    unknown = [a for a in s["algorithms"] if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"unknown algorithms {unknown}; choose from {list(ALGORITHMS)}")
    if "occurrences" not in s:
        raise ValueError("spec needs 'occurrences'")
    return s


def run_id(spec: dict) -> str:
    blob = json.dumps({"v": VERSION, **spec}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _sample_background(env: pd.DataFrame, n: int, effort: pd.DataFrame | None, rng) -> np.ndarray:
    """Indices into env for background cells (uniform, or proportional to sampling effort)."""
    if effort is None:
        if n >= len(env):
            return np.arange(len(env))
        return rng.choice(len(env), n, replace=False)
    e = env[["cell"]].merge(effort, on="cell", how="left")["effort"].fillna(0).to_numpy()
    cand = np.flatnonzero(e > 0)
    if len(cand) == 0:
        raise ValueError("target-group background is empty in this domain")
    p = e[cand] / e[cand].sum()
    return rng.choice(cand, min(n, len(cand)), replace=False, p=p)


def _perm_importance(model, X, y, cols, rng, base_auc):
    out = {}
    pres = y == 1
    for j, c in enumerate(cols):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        p = model.predict(Xp)
        out[c] = round(max(0.0, base_auc - evaluate.auc(p[pres], p[~pres])), 4)
    return out


def run(spec: dict, log=print) -> dict:
    spec = normalize(spec)
    rid = run_id(spec)
    cached = outputs.load_summary(rid)
    if cached:
        cached["cached"] = True
        return cached

    t0 = time.perf_counter()
    timings = {}
    rng = np.random.default_rng(spec["seed"])
    con = data.connect()

    def tick(label):
        timings[label] = round(time.perf_counter() - t0 - sum(timings.values()), 2)
        log(f"[sdm {rid}] {label}: {timings[label]}s")

    # 1. occurrences
    pres8, abs8, occ_info = data.load_occurrences(con, spec["occurrences"])
    if len(pres8) < 5:
        msg = f"only {len(pres8)} occupied cells after filtering; need at least 5."
        if occ_info.get("uncertainty_note"):
            msg += " " + occ_info["uncertainty_note"]
        raise ValueError(msg)
    tick("occurrences")

    # 2. domain + resolution
    dom = data.build_domain(con, pres8, spec["domain"])
    res = data.choose_resolution(dom["approx_h8_cells"], spec["resolution"], spec["max_cells"])
    tick("domain")

    # 3. covariates
    env, cols = data.load_covariates(con, spec["covariates"], res, dom)
    if len(env) == 0:
        raise ValueError("no covariate cells in the domain")
    tick("covariates")

    # 4. presences (and absences) at the model resolution, inside the covariate grid
    con.register("pres8", pres8)
    pres = con.sql(f"SELECT DISTINCT h3_cell_to_parent(h8, {res})::UBIGINT AS cell FROM pres8").df()
    pres_idx = env.reset_index().merge(pres, on="cell")["index"].to_numpy()
    n_inside, n_land = land_share(con, pres8, env["cell"].to_numpy(), res, dom)
    share = n_land / n_inside if n_inside else 0.0
    occ_info["presence_cells_in_domain"] = n_inside
    occ_info["land_share"] = round(share, 3)
    if n_inside and share < MIN_LAND_SHARE and not spec.get("allow_offshore"):
        raise ValueError(
            f"only {share:.0%} of the {n_inside} occupied cells in the study area are on land. "
            "This server's predictors (CHELSA climate, terrain, land cover, human modification) "
            "cover land only, so it cannot model marine or largely offshore species; the fit would "
            "describe only a coastal fringe. Pass allow_offshore=true to fit the land cells anyway.")
    if n_inside and share < WARN_LAND_SHARE:
        occ_info["land_note"] = (
            f"{1 - share:.0%} of the occupied cells in the study area are offshore or otherwise "
            "lack land predictors and were excluded; the model describes only the land part of the "
            "species' distribution.")
    if len(pres_idx) < 5:
        raise ValueError(f"only {len(pres_idx)} presence cells have covariate data")
    abs_idx = None
    if abs8 is not None:
        con.register("abs8", abs8)
        ab = con.sql(f"SELECT DISTINCT h3_cell_to_parent(h8, {res})::UBIGINT AS cell FROM abs8").df()
        abs_idx = np.setdiff1d(env.reset_index().merge(ab, on="cell")["index"].to_numpy(), pres_idx)

    # 5. background
    effort = None
    bias = spec["background"].get("bias")
    tg = None
    if bias == "target_group":
        rank = spec["background"].get("target_rank", "class")
        name = spec["background"].get("target_name") or occ_info.get(rank)
        if not name:
            raise ValueError("target_group background needs target_name (or GBIF occurrences)")
        effort = data.target_group_effort(con, rank, name, res)
        tg = {"rank": rank, "name": name, "cells_with_effort": int(len(effort))}
        tick("target_group_effort")
    if abs_idx is not None and len(abs_idx) >= 5 and not spec["background"].get("add_background"):
        bg_idx, mode = abs_idx, "presence-absence"
    else:
        bg_idx, mode = _sample_background(env, int(spec["background"]["n"]), effort, rng), "presence-background"

    X_all = env[cols].to_numpy(dtype=np.float64)
    tr_idx = np.concatenate([pres_idx, bg_idx])
    X = X_all[tr_idx]
    y = np.concatenate([np.ones(len(pres_idx)), np.zeros(len(bg_idx))])
    cells_tr = env["cell"].to_numpy()[tr_idx]
    tick("training_set")

    # 6. final fits + predictions over the domain
    algos = list(spec["algorithms"])
    preds, models, importance = {}, {}, {}
    for a in algos:
        m = ALGORITHMS[a]().fit(X, y)
        models[a] = m
        preds[a] = m.predict(X_all).astype(np.float32)
    tick("fit_predict")

    # 7. spatial block CV (BRT reuses the full-data tree count instead of nesting a CV)
    hints = {"brt": {"n_trees": getattr(models.get("brt"), "n_trees", None)}}
    cv = {}
    k = int(spec["cv_folds"] or 0)
    if k >= 2:
        folds, block_res = evaluate.spatial_blocks(cells_tr, env["cell"].to_numpy()[pres_idx], k, spec["seed"])
        oof = {a: np.full(len(y), np.nan) for a in algos}
        for f in range(k):
            te = folds == f
            if y[te].sum() == 0 or y[~te].sum() < 5:
                continue
            for a in algos:
                oof[a][te] = ALGORITHMS[a](**hints.get(a, {})).fit(X[~te], y[~te]).predict(X[te])
        for a in algos:
            ok = ~np.isnan(oof[a])
            cv[a] = {kk: round(v, 4) for kk, v in evaluate.metrics(oof[a][ok & (y == 1)], oof[a][ok & (y == 0)]).items()}
        cv["_blocks"] = {"folds": k, "block_h3_res": block_res}
        tick("cross_validation")

    weights = None
    if spec["ensemble"] and len(algos) > 1:
        if cv:
            weights = {a: max(cv[a]["auc"] - 0.5, 0.0) for a in algos}
        else:
            weights = {a: 1.0 for a in algos}
        tot = sum(weights.values()) or 1.0
        weights = {a: w / tot for a, w in weights.items()}
        final = sum(weights[a] * preds[a] for a in algos)
        final_name = "ensemble"
        if cv:
            oof_ens = sum(weights[a] * oof[a] for a in algos)
            ok = ~np.isnan(oof_ens)
            cv["ensemble"] = {kk: round(v, 4) for kk, v in evaluate.metrics(oof_ens[ok & (y == 1)], oof_ens[ok & (y == 0)]).items()}
    else:
        final, final_name = preds[algos[0]], algos[0]

    # threshold from training predictions of the final surface
    f_tr = final[tr_idx]
    thr = evaluate.threshold(spec["threshold"], f_tr[y == 1], f_tr[y == 0])
    train_auc = {}
    for a in algos:
        p = preds[a][tr_idx]
        train_auc[a] = round(evaluate.auc(p[y == 1], p[y == 0]), 4)
        importance[a] = _perm_importance(models[a], X, y, cols, rng, train_auc[a])
    tick("threshold_importance")

    # extrapolation: number of covariates outside the training range
    lo, hi = X.min(0), X.max(0)
    extrap = ((X_all < lo) | (X_all > hi)).sum(1).astype(np.int8)

    pred_df = pd.DataFrame({"cell": env["cell"].to_numpy(), "suitability": final.astype(np.float32),
                            "suitable": (final >= thr), "n_extrapolated": extrap})
    pred_df["presence"] = False
    pred_df.loc[pres_idx, "presence"] = True
    for a in algos:
        pred_df[f"pred_{a}"] = preds[a]

    # 8. projection to a future climate
    proj_info = None
    proj_df = None
    if spec["project_to"]:
        period = spec["project_to"].replace("chelsa-2-1-", "")
        if period not in data.FUTURE_SCENARIOS:
            raise ValueError(f"project_to must be one of {data.FUTURE_SCENARIOS}")
        envf, colsf = data.load_covariates(con, spec["covariates"], res, dom, period=period)
        Xf = envf[cols].to_numpy(dtype=np.float64)
        pf = {a: models[a].predict(Xf).astype(np.float32) for a in algos}
        ff = sum(weights[a] * pf[a] for a in algos) if weights else pf[algos[0]]
        proj_df = pd.DataFrame({"cell": envf["cell"].to_numpy(), "suitability": ff.astype(np.float32),
                                "suitable": ff >= thr,
                                "n_extrapolated": ((Xf < lo) | (Xf > hi)).sum(1).astype(np.int8)})
        tick("projection")

    # 9. write outputs
    area = outputs.cell_area_km2(res)
    try:
        regions_now = data.region_summary(con, pred_df.cell[pred_df.suitable].to_numpy(), res)
        regions_future = (data.region_summary(con, proj_df.cell[proj_df.suitable].to_numpy(), res)
                          if proj_df is not None else None)
    except Exception as e:  # a summary must never fail the fit
        log(f"[sdm {rid}] region summary skipped: {e}")
        regions_now, regions_future = None, None
    tick("region_summary")
    summary = {
        "run_id": rid,
        "version": VERSION,
        "spec": spec,
        "mode": mode,
        "occurrences": {**occ_info, "occupied_h8_cells": int(len(pres8)),
                        "presence_cells_at_res": int(len(pres_idx))},
        "background_cells": int(len(bg_idx)),
        "background_bias": tg,
        "resolution": res,
        "domain": {k: v for k, v in dom.items() if k not in ("mask_sql",)} | {"cells": int(len(env))},
        "covariates": cols,
        "final_model": final_name,
        "ensemble_weights": weights and {a: round(w, 3) for a, w in weights.items()},
        "cv_metrics": cv or None,
        "train_auc": train_auc,
        "variable_importance": importance,
        "threshold": {"rule": spec["threshold"], "value": round(thr, 4)},
        "suitable_area_km2": round(float(pred_df.suitable.sum()) * area, 1),
        "range_by_region": regions_now,
        "notes": [
            (f"Background: {len(bg_idx)} cells sampled in proportion to GBIF records of "
             f"{tg['rank']} {tg['name']} (target-group correction for sampling bias)." if tg else
             f"Background: {len(bg_idx)} uniformly random cells; no correction for sampling "
             "bias was applied." if mode == "presence-background" else
             "Absences: user-supplied absence records were used instead of background cells."),
            "TSS and AUC are presence vs background unless mode is presence-absence.",
            "Suitability is relative (cloglog-scale for maxent); not an occupancy probability.",
            "n_extrapolated counts covariates outside the training range in each cell.",
        ],
    }
    if occ_info.get("land_note"):
        summary["notes"].insert(1, occ_info["land_note"])
    if proj_df is not None:
        both = pred_df[["cell", "suitable"]].merge(proj_df[["cell", "suitable"]], on="cell", suffixes=("_now", "_future"))
        summary["projection"] = {
            "scenario": spec["project_to"],
            "suitable_area_km2": round(float(proj_df.suitable.sum()) * area, 1),
            "range_by_region": regions_future,
            "gained_km2": round(float((~both.suitable_now & both.suitable_future).sum()) * area, 1),
            "lost_km2": round(float((both.suitable_now & ~both.suitable_future).sum()) * area, 1),
            "cells_extrapolated": int((proj_df.n_extrapolated > 0).sum()),
        }
    summary["artifacts"] = outputs.write_all(rid, pred_df, proj_df, res, summary)
    tick("write")
    summary["timings_s"] = timings
    summary["elapsed_s"] = round(time.perf_counter() - t0, 2)
    outputs.save_summary(rid, summary)
    con.close()
    return summary
