"""Algorithm tests on a simulated species with a known niche (no network)."""
import numpy as np
import pytest

from hexsdm import evaluate
from hexsdm.models import ALGORITHMS, MaxEnt


@pytest.fixture(scope="module")
def sim():
    """Landscape of 30k cells, 4 covariates; true intensity is Gaussian in x0, x1."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30_000, 4))
    eta = -1.5 * (X[:, 0] - 0.5) ** 2 - 1.0 * (X[:, 1] + 0.3) ** 2
    p = np.exp(eta)
    pres = rng.choice(len(X), 300, replace=False, p=p / p.sum())
    bg = rng.choice(len(X), 5000, replace=False)
    Xtr = np.vstack([X[pres], X[bg]])
    ytr = np.r_[np.ones(len(pres)), np.zeros(len(bg))]
    return X, eta, Xtr, ytr


@pytest.mark.parametrize("algo", list(ALGORITHMS))
def test_recovers_niche(sim, algo):
    X, eta, Xtr, ytr = sim
    m = ALGORITHMS[algo]().fit(Xtr, ytr)
    pred = m.predict(X)
    assert pred.shape == (len(X),)
    assert np.all((pred >= 0) & (pred <= 1))
    # rank agreement with the true intensity
    from scipy.stats import spearmanr
    rho = spearmanr(pred, eta).statistic
    floor = 0.5 if algo == "bioclim" else 0.7
    assert rho > floor, f"{algo}: spearman {rho:.2f}"


def test_maxent_sparse_and_ignores_noise(sim):
    X, eta, Xtr, ytr = sim
    m = MaxEnt().fit(Xtr, ytr)
    assert 0 < m.n_features_used < 200
    used = {i for _, i, j, _ in m.specs} | {j for _, _, j, _ in m.specs if j >= 0}
    assert {0, 1} <= used


def test_maxent_feature_classes_by_sample_size():
    m = MaxEnt()
    assert m._feat_classes(5) == "l"
    assert m._feat_classes(12) == "lq"
    assert m._feat_classes(50) == "lqh"
    assert m._feat_classes(500) == "lqph"


def test_metrics_basic():
    pres = np.array([0.9, 0.8, 0.7])
    bg = np.array([0.1, 0.2, 0.75, 0.3])
    assert evaluate.auc(pres, bg) == pytest.approx(11 / 12)
    thr, tss = evaluate.max_sss_threshold(pres, bg)
    assert tss > 0.5
    assert evaluate.threshold("p10", pres, bg) == pytest.approx(np.quantile(pres, 0.1))
    assert evaluate.threshold(0.4, pres, bg) == 0.4


def test_boyce_monotone():
    rng = np.random.default_rng(0)
    bg = rng.uniform(size=20_000)
    pres = rng.choice(bg, 2000, p=bg / bg.sum())  # selection proportional to suitability
    assert evaluate.boyce(pres, bg) > 0.9
