"""Evaluation: spatial block cross-validation on H3 parents, AUC, Boyce, TSS, thresholds."""
from __future__ import annotations

import h3
import numpy as np
from scipy import stats


def auc(pres: np.ndarray, bg: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney), presence vs background/absence."""
    if len(pres) == 0 or len(bg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pres, bg]))
    return float((r[: len(pres)].sum() - len(pres) * (len(pres) + 1) / 2) / (len(pres) * len(bg)))


def boyce(pres: np.ndarray, bg: np.ndarray, n_windows: int = 101, width: float = 0.1) -> float:
    """Continuous Boyce index (Hirzel et al. 2006): Spearman correlation between
    suitability and the predicted/expected ratio over moving windows."""
    lo, hi = min(pres.min(), bg.min()), max(pres.max(), bg.max())
    w = (hi - lo) * width
    centers = np.linspace(lo + w / 2, hi - w / 2, n_windows)
    pe = []
    for c in centers:
        p = np.mean((pres >= c - w / 2) & (pres <= c + w / 2))
        e = np.mean((bg >= c - w / 2) & (bg <= c + w / 2))
        pe.append(p / e if e > 0 else np.nan)
    pe = np.array(pe)
    ok = ~np.isnan(pe)
    if ok.sum() < 3 or np.ptp(pe[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(centers[ok], pe[ok]).statistic)


def max_sss_threshold(pres: np.ndarray, bg: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing sensitivity + specificity (Liu et al. 2013); returns (thr, TSS)."""
    cand = np.unique(np.quantile(np.concatenate([pres, bg]), np.linspace(0, 1, 201)))
    ps, bs = np.sort(pres), np.sort(bg)
    sens = 1 - np.searchsorted(ps, cand, side="left") / len(ps)
    spec = np.searchsorted(bs, cand, side="left") / len(bs)
    i = int(np.argmax(sens + spec))
    return float(cand[i]), float(sens[i] + spec[i] - 1)


def threshold(rule, pres: np.ndarray, bg: np.ndarray) -> float:
    if isinstance(rule, (int, float)):
        return float(rule)
    rule = (rule or "maxSSS").lower()
    if rule == "maxsss":
        return max_sss_threshold(pres, bg)[0]
    if rule == "p10":
        return float(np.quantile(pres, 0.10))
    if rule in ("mtp", "min_presence"):
        return float(pres.min())
    raise ValueError("threshold must be a number or one of maxSSS, p10, mtp")


def spatial_blocks(cells: np.ndarray, pres_cells: np.ndarray, k: int, seed: int = 1) -> tuple[np.ndarray, int]:
    """Assign each cell to one of k folds by its H3 parent block.

    Picks the coarsest block resolution (2..7) that gives at least 3*k presence
    blocks, so folds are spatially separated but each holds presences."""
    res_cell = h3.get_resolution(h3.int_to_str(int(cells[0])))
    chosen = None
    for br in range(2, res_cell):
        pb = {h3.cell_to_parent(h3.int_to_str(int(c)), br) for c in pres_cells}
        if len(pb) >= 3 * k:
            chosen = br
            break
    if chosen is None:
        chosen = max(2, res_cell - 1)
    parents = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), chosen)) for c in cells],
                       dtype="uint64")
    uniq = np.unique(parents)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, k, len(uniq))))
    folds = np.array([fold_of[p] for p in parents])
    return folds, chosen


def metrics(pred_pres: np.ndarray, pred_bg: np.ndarray) -> dict:
    thr, tss = max_sss_threshold(pred_pres, pred_bg)
    return {"auc": auc(pred_pres, pred_bg), "boyce": boyce(pred_pres, pred_bg), "tss": tss}
