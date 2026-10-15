"""Where MaxEnt fit time goes: feature construction, KKT checks, glum solves."""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import hexsdm.models as M
d = np.load(sys.argv[1]); X_all = d["X_all"].astype(float); pi = d["pres_idx"]
rng = np.random.default_rng(1); bg = rng.choice(len(X_all), 20000, replace=False)
X = np.vstack([X_all[pi], X_all[bg]]); y = np.r_[np.ones(len(pi)), np.zeros(len(bg))]
orig = M._lasso_logit_working_set
stats = {"rounds": 0, "glum": 0.0, "kkt": 0.0, "active": 0}
from glum import GeneralizedLinearRegressor as G
_fit = G.fit
def tfit(self, *a, **k):
    t = time.perf_counter(); r = _fit(self, *a, **k); stats["glum"] += time.perf_counter() - t; stats["rounds"] += 1; stats["active"] = a[0].shape[1]; return r
G.fit = tfit
_feat = M.MaxEnt._features
ft = [0.0]
def tfeat(self, *a, **k):
    t = time.perf_counter(); r = _feat(self, *a, **k); ft[0] += time.perf_counter() - t; return r
M.MaxEnt._features = tfeat
t = time.perf_counter(); m = M.MaxEnt().fit(X, y); tot = time.perf_counter() - t
print(f"total {tot:.2f}s  features {ft[0]:.2f}s  glum {stats['glum']:.2f}s in {stats['rounds']} rounds (final active {stats['active']})  other {tot-ft[0]-stats['glum']:.2f}s  nnz {m.n_features_used}")
