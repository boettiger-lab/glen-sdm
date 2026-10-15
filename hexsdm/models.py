"""SDM algorithms with one interface: fit(X, y, w) -> self; predict(X) -> suitability in [0, 1].

X: float array (n, p) of raw covariates; y: 1 = presence cell, 0 = background (or
absence) cell; w: optional case weights. Profile methods (bioclim, mahalanobis)
use the presence rows only.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from scipy import stats



def _default_threads() -> int:
    """SDM_MODEL_THREADS, else the cgroup CPU quota, else min(cpu_count, 16).

    os.cpu_count() reports the node's cores, not the pod's; oversubscribing
    LightGBM on a shared node is ~10x slower than using a few threads."""
    if os.environ.get("SDM_MODEL_THREADS"):
        return int(os.environ["SDM_MODEL_THREADS"])
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    return min(os.cpu_count() or 1, 16)


N_THREADS = _default_threads()


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """Presence weight 1; background weights summing to the number of presences
    (the 'down-weighted background' used by Valavi et al. 2022 for GLM/GAM/BRT)."""
    npres, nbg = y.sum(), (1 - y).sum()
    return np.where(y == 1, 1.0, npres / max(nbg, 1))


class Scaler:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        self.lo, self.hi = X.min(0), X.max(0)
        return self

    def transform(self, X, clamp=True):
        if clamp:
            X = np.clip(X, self.lo, self.hi)
        return (X - self.mu) / self.sd


# ---------------------------------------------------------------------------
# Profile (presence-only) methods
# ---------------------------------------------------------------------------

class Bioclim:
    """BIOCLIM climate envelope (Nix 1986; Booth et al. 2014), as in dismo::bioclim:
    each variable's percentile among presences, folded so 0.5 -> 1, combined by
    the minimum across variables."""
    name = "bioclim"

    def fit(self, X, y, w=None):
        self.sorted = np.sort(X[y == 1], axis=0)
        return self

    def predict(self, X):
        n = self.sorted.shape[0]
        score = np.ones(X.shape[0])
        for j in range(X.shape[1]):
            lo = np.searchsorted(self.sorted[:, j], X[:, j], side="left")
            hi = np.searchsorted(self.sorted[:, j], X[:, j], side="right")
            f = (lo + hi) / (2.0 * n)
            score = np.minimum(score, 2 * np.minimum(f, 1 - f))
        return score


class Mahalanobis:
    """Mahalanobis distance to the presence centroid (Farber & Kadmon 2003),
    converted to suitability with the chi-square survival function."""
    name = "mahalanobis"

    def fit(self, X, y, w=None):
        P = X[y == 1]
        self.mu = P.mean(0)
        cov = np.cov(P, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self.icov = np.linalg.pinv(cov)
        self.df = X.shape[1]
        return self

    def predict(self, X):
        d = X - self.mu
        d2 = np.einsum("ij,jk,ik->i", d, self.icov, d)
        return stats.chi2.sf(d2, self.df)


# ---------------------------------------------------------------------------
# Regression (presence-background) methods, fit with glum
# ---------------------------------------------------------------------------

def _glum_logit(F, y, w, alpha, l1_ratio=0.0, P1=None):
    from glum import GeneralizedLinearRegressor
    kw = dict(family="binomial", alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
              max_iter=500, gradient_tol=1e-6)
    if P1 is not None:
        kw["P1"] = P1
    m = GeneralizedLinearRegressor(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(np.asarray(F, dtype=np.float64), np.asarray(y, dtype=np.float64), sample_weight=np.asarray(w, dtype=np.float64))
    return m


class GLM:
    """Logistic GLM with linear + quadratic terms and down-weighted background
    (IWLR-style; Fithian & Hastie 2013), light ridge penalty for stability."""
    name = "glm"

    def _features(self, X):
        Z = self.scaler.transform(X)
        return np.hstack([Z, Z ** 2])

    def fit(self, X, y, w=None):
        self.scaler = Scaler().fit(X)
        w = _balanced_weights(y) if w is None else w
        self.m = _glum_logit(self._features(X), y, w, alpha=1e-4)
        return self

    def predict(self, X):
        return self.m.predict(self._features(X))


class GAM:
    """Additive logistic model: cubic B-spline basis per covariate with a ridge
    penalty (a penalized-regression-spline approximation to mgcv::gam)."""
    name = "gam"

    def __init__(self, n_knots=6, alpha=1e-3):
        self.n_knots, self.alpha = n_knots, alpha

    def fit(self, X, y, w=None):
        from sklearn.preprocessing import SplineTransformer
        self.scaler = Scaler().fit(X)
        self.spl = SplineTransformer(n_knots=self.n_knots, degree=3, extrapolation="constant")
        F = self.spl.fit_transform(self.scaler.transform(X))
        w = _balanced_weights(y) if w is None else w
        self.m = _glum_logit(F, y, w, alpha=self.alpha)
        return self

    def predict(self, X):
        return self.m.predict(self.spl.transform(self.scaler.transform(X)))


class MaxEnt:
    """MaxEnt via its equivalence to an L1-penalized (infinitely weighted) logistic
    regression, following maxnet (Phillips et al. 2017): linear, quadratic,
    product and hinge features chosen by sample size, maxnet's default
    feature-specific regularization, cloglog output, clamping."""
    name = "maxent"

    def __init__(self, regmult=1.0, features=None, nknots=50):
        self.regmult, self.features, self.nknots = regmult, features, nknots

    def _feat_classes(self, npres):
        if self.features:
            return self.features
        return "l" if npres < 10 else "lq" if npres < 15 else "lqh" if npres < 80 else "lqph"

    def _specs(self, p):
        """Feature list as (kind, i, j, knot): l/q use i; p uses i, j; hf/hr use i, knot."""
        sp = [("l", i, -1, 0.0) for i in range(p)]
        if "q" in self.fc:
            sp += [("q", i, -1, 0.0) for i in range(p)]
        if "p" in self.fc:
            sp += [("p", i, j, 0.0) for i, j in zip(*np.triu_indices(p, 1))]
        if "h" in self.fc:
            k = np.linspace(0, 1, self.nknots)
            sp += [("hf", i, -1, kk) for i in range(p) for kk in k[:-1]]  # hingeval(x, knot, max)
            sp += [("hr", i, -1, kk) for i in range(p) for kk in k[1:]]   # hingeval(x, min, knot)
        return sp

    def _features(self, X, specs=None):
        Z = (np.clip(X, self.lo, self.hi) - self.lo) / self.rng
        specs = self.specs if specs is None else specs
        F = np.empty((len(Z), len(specs)))
        for c, (kind, i, j, kk) in enumerate(specs):
            z = Z[:, i]
            if kind == "l":
                F[:, c] = z
            elif kind == "q":
                F[:, c] = z * z
            elif kind == "p":
                F[:, c] = z * Z[:, j]
            elif kind == "hf":
                F[:, c] = np.clip((z - kk) / (1 - kk), 0, 1)
            else:
                F[:, c] = np.clip(z / kk, 0, 1)
        return F

    def _regularization(self, F, kinds, y):
        P = F[y == 1]
        npres = P.shape[0]
        tables = {"l": ([0, 10, 30, 100], [1, 1, 0.2, 0.05]),
                  "q": ([0, 10, 17, 30, 100], [1.3, 0.8, 0.5, 0.25, 0.05]),
                  "p": ([0, 10, 17, 30, 100], [2.6, 1.6, 0.9, 0.55, 0.05]),
                  "h": ([0, 1], [0.5, 0.5])}
        lqp = "p" if "p" in kinds else "q" if "q" in kinds else "l"
        beta = np.array([np.interp(npres, *tables["h" if k == "h" else lqp]) for k in kinds])
        sd = P.std(0, ddof=1) if npres > 1 else np.zeros(F.shape[1])
        reg = sd * beta / np.sqrt(npres)
        hmin = np.where(kinds == "h", np.maximum(sd, 1 / np.sqrt(npres)) * 0.5 / np.sqrt(npres), 0)
        reg = np.maximum.reduce([0.001 * (F.max(0) - F.min(0)), reg, hmin])
        return reg * self.regmult

    def fit(self, X, y, w=None):
        # maxnet adds presences to the background ("addsamplestobackground")
        Xb = np.vstack([X, X[y == 1]])
        yb = np.concatenate([y, np.zeros(int(y.sum()))])
        self.lo, self.hi = Xb.min(0), Xb.max(0)
        self.rng = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.fc = self._feat_classes(int(y.sum()))
        specs = self._specs(X.shape[1])
        F = self._features(Xb, specs)
        keep = F.std(0) > 0
        F = F[:, keep]
        self.specs = [sp for sp, k in zip(specs, keep) if k]
        kinds = np.array(["h" if sp[0] in ("hf", "hr") else sp[0] for sp in self.specs])
        reg = self._regularization(F, kinds, yb)
        wts = np.where(yb == 1, 1.0, 100.0)
        # glmnet objective, rescaled: -(1/sum w) sum w*loglik + sum_j reg_j*npres/sum(w) |b_j|.
        # glum normalizes weights to sum 1, so P1 is the per-coefficient L1 weight.
        P1 = reg * yb.sum() / wts.sum()
        coef, icpt = _lasso_logit_working_set(F, yb, wts, P1)
        nz = np.flatnonzero(coef)
        self.specs = [self.specs[i] for i in nz]
        self.coef_ = coef[nz]
        link_bg = F[yb == 0][:, nz] @ self.coef_
        mx = link_bg.max()
        self.alpha_ = -(mx + np.log(np.exp(link_bg - mx).sum()))
        raw = np.exp(link_bg + self.alpha_)
        self.entropy_ = -np.sum(raw * np.log(raw + 1e-300))
        self.n_features_used = len(nz)
        return self

    def predict(self, X):
        link = self._features(X) @ self.coef_
        return 1 - np.exp(-np.exp(np.minimum(self.entropy_ + link + self.alpha_, 50)))


def _lasso_logit_working_set(F, y, w, P1, init=64, grow=64, tol=1e-4, max_rounds=30):
    """Weighted L1 logistic regression by working sets.

    Solve on a small set of columns, check the KKT condition |grad_j| <= P1_j for
    every column with one matrix-vector product, add the worst violators, repeat.
    MaxEnt solutions are sparse (tens of nonzeros among ~2000 features), so this
    beats solving the full problem by roughly an order of magnitude."""
    from glum import GeneralizedLinearRegressor
    wn = w / w.sum()
    p0 = float((wn * y).sum())
    mu = np.full(len(y), p0)
    active = np.zeros(F.shape[1], bool)
    coef_full, icpt = np.zeros(F.shape[1]), np.log(p0 / (1 - p0))
    for _ in range(max_rounds):
        score = np.abs(F.T @ (wn * (y - mu))) / P1
        viol = (score > 1 + tol) & ~active
        if not viol.any():
            break
        n_add = init if not active.any() else grow
        cand = np.flatnonzero(viol)
        add = cand[np.argsort(-score[cand])[:n_add]]
        active[add] = True
        idx = np.flatnonzero(active)
        m = GeneralizedLinearRegressor(family="binomial", alpha=1.0, l1_ratio=1.0, P1=P1[idx],
                                       fit_intercept=True, max_iter=500, gradient_tol=1e-7,
                                       start_params=np.concatenate([[icpt], coef_full[idx]]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(F[:, idx], y, sample_weight=w)
        coef_full[:] = 0
        coef_full[idx] = m.coef_
        icpt = float(m.intercept_)
        mu = 1 / (1 + np.exp(-(icpt + F[:, idx] @ m.coef_)))
    return coef_full, icpt


# ---------------------------------------------------------------------------
# Tree ensembles
# ---------------------------------------------------------------------------

class BRT:
    """Boosted regression trees (Elith et al. 2008) via LightGBM: small trees, slow
    learning rate, bagging 0.75, number of trees chosen by internal 5-fold CV
    (the gbm.step recipe), down-weighted background."""
    name = "brt"

    def __init__(self, learning_rate=0.02, num_leaves=8, max_rounds=3000, n_trees=None):
        # n_trees fixes the number of trees and skips the internal CV (used inside
        # the outer spatial CV, where the full-data choice is reused).
        self.lr, self.leaves, self.max_rounds, self.n_trees = learning_rate, num_leaves, max_rounds, n_trees

    def fit(self, X, y, w=None):
        import lightgbm as lgb
        w = _balanced_weights(y) if w is None else w
        params = dict(objective="binary", learning_rate=self.lr, num_leaves=self.leaves,
                      min_data_in_leaf=10, bagging_fraction=0.75, bagging_freq=1,
                      feature_fraction=1.0, verbose=-1, num_threads=N_THREADS, seed=1)
        ds = lgb.Dataset(X, y, weight=w, free_raw_data=False)
        if self.n_trees is None:
            cv = lgb.cv(params, ds, num_boost_round=self.max_rounds, nfold=5, stratified=True,
                        callbacks=[lgb.early_stopping(100, verbose=False)], seed=1)
            self.n_trees = len(next(iter(cv.values())))
        self.m = lgb.train(params, ds, num_boost_round=max(self.n_trees, 50))
        return self

    def predict(self, X):
        return self.m.predict(X, num_threads=N_THREADS)


class RFDownsampled:
    """Down-sampled ("balanced") random forest (Valavi et al. 2021), as in
    randomForest(sampsize = c(n_pres, n_pres)): every tree is grown to full depth
    on a bootstrap of n_pres presences and n_pres background cells, mtry = sqrt(p).
    Suitability is the mean of the trees' leaf presence shares."""
    name = "rf"

    def __init__(self, n_trees=500, seed=1):
        self.n_trees, self.seed = n_trees, seed

    def fit(self, X, y, w=None):
        from concurrent.futures import ThreadPoolExecutor
        from sklearn.tree import DecisionTreeClassifier
        P, B = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n = len(P)
        mtry = max(1, int(np.sqrt(X.shape[1])))

        def grow(i):
            r = np.random.default_rng(self.seed * 100003 + i)
            idx = np.concatenate([r.choice(P, n), r.choice(B, n)])
            return DecisionTreeClassifier(max_features=mtry, random_state=i).fit(X[idx], y[idx])

        with ThreadPoolExecutor(N_THREADS) as ex:  # sklearn trees release the GIL
            self.trees = list(ex.map(grow, range(self.n_trees)))
        return self

    def predict(self, X, chunk=200_000):
        from concurrent.futures import ThreadPoolExecutor
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros(len(X))
        with ThreadPoolExecutor(N_THREADS) as ex:
            for s in range(0, len(X), chunk):
                Xc = X[s:s + chunk]
                out[s:s + chunk] = sum(ex.map(lambda t: t.predict_proba(Xc)[:, 1], self.trees)) / len(self.trees)
        return out


ALGORITHMS = {
    "bioclim": Bioclim,
    "mahalanobis": Mahalanobis,
    "glm": GLM,
    "gam": GAM,
    "maxent": MaxEnt,
    "brt": BRT,
    "rf": RFDownsampled,
}

ALGORITHM_INFO = {
    "bioclim": {"data": "presence-only", "speed": "instant",
                "summary": "Climate envelope: percentile of each covariate among presences, min across covariates.",
                "reference": "Booth et al. 2014, https://doi.org/10.1111/ddi.12144"},
    "mahalanobis": {"data": "presence-only", "speed": "instant",
                    "summary": "Mahalanobis distance from the presence centroid, chi-square p-value as suitability.",
                    "reference": "Farber & Kadmon 2003, https://doi.org/10.1016/S0304-3800(02)00327-7"},
    "glm": {"data": "presence-background", "speed": "fast",
            "summary": "Logistic regression with linear+quadratic terms, background down-weighted.",
            "reference": "Fithian & Hastie 2013, https://doi.org/10.1214/13-AOAS667"},
    "gam": {"data": "presence-background", "speed": "fast",
            "summary": "Additive logistic model with penalized cubic splines per covariate.",
            "reference": "Hastie & Tibshirani 1986; Valavi et al. 2022, https://doi.org/10.1002/ecm.1486"},
    "maxent": {"data": "presence-background", "speed": "fast",
               "summary": "MaxEnt as L1-penalized logistic regression (maxnet): l/q/p/h features, default regularization, cloglog output.",
               "reference": "Phillips et al. 2017, https://doi.org/10.1111/ecog.03049"},
    "brt": {"data": "presence-background", "speed": "moderate",
            "summary": "Boosted regression trees (LightGBM), trees chosen by internal CV.",
            "reference": "Elith, Leathwick & Hastie 2008, https://doi.org/10.1111/j.1365-2656.2008.01390.x"},
    "rf": {"data": "presence-background", "speed": "fast",
           "summary": "Down-sampled (balanced) random forest: each tree sees equal presences and background.",
           "reference": "Valavi et al. 2021, https://doi.org/10.1111/ecog.05615"},
    "ensemble": {"data": "presence-background", "speed": "sum of members",
                 "summary": "Weighted mean of member predictions, weights = cross-validated AUC - 0.5.",
                 "reference": "Araújo & New 2007, https://doi.org/10.1016/j.tree.2006.09.010"},
}
