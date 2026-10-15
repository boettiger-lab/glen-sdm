"""Per-stage, per-algorithm compute profile of one SDM run (no caching).

Usage: python benchmarks/profile_pipeline.py "Ambystoma californiense"
"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from hexsdm import data, evaluate, pipeline
from hexsdm.models import ALGORITHMS, N_THREADS

sp = sys.argv[1] if len(sys.argv) > 1 else "Ambystoma californiense"
T = {}
def tic(): return time.perf_counter()
t = tic(); con = data.connect()
pres8, _, info = data.load_occurrences(con, {"gbif": {"species": sp}})
dom = data.build_domain(con, pres8, None)
res = data.choose_resolution(dom["approx_h8_cells"], None, 2_000_000)
env, cols = data.load_covariates(con, pipeline.DEFAULTS["covariates"], res, dom)
T["data"] = tic() - t
con.register("p8", pres8)
pres = con.sql(f"SELECT DISTINCT h3_cell_to_parent(h8,{res})::UBIGINT cell FROM p8").df()
pi = env.reset_index().merge(pres, on="cell")["index"].to_numpy()
rng = np.random.default_rng(1)
bg = rng.choice(len(env), 20000, replace=False)
X_all = env[cols].to_numpy(float); tr = np.r_[pi, bg]; X = X_all[tr]; y = np.r_[np.ones(len(pi)), np.zeros(len(bg))]
print(f"threads={N_THREADS} cells={len(env)} pres={len(pi)} p={len(cols)} data={T['data']:.1f}s", flush=True)
t = tic(); folds, br = evaluate.spatial_blocks(env.cell.to_numpy()[tr], env.cell.to_numpy()[pi], 5); print(f"blocks {tic()-t:.2f}s")
for a in ["maxent", "brt", "rf", "glm"]:
    t = tic(); m = ALGORITHMS[a]().fit(X, y); tf = tic() - t
    t = tic(); m.predict(X_all); tp = tic() - t
    hint = {"n_trees": m.n_trees} if a == "brt" else {}
    t = tic()
    for f in range(5):
        te = folds == f
        ALGORITHMS[a](**hint).fit(X[~te], y[~te]).predict(X[te])
    tcv = tic() - t
    t = tic()
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); m.predict(Xp)
    tim = tic() - t
    print(f"{a:7s} fit {tf:5.2f}  predict_all {tp:5.2f}  cv5 {tcv:5.2f}  perm_imp {tim:5.2f}", flush=True)
