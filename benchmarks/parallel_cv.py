"""Sequential vs concurrent spatial CV (5 folds x 4 algorithms) on one training set."""
import sys, time, warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
import hexsdm.models as M
from hexsdm import evaluate
d = np.load(sys.argv[1]); X_all = d["X_all"].astype(float); pi = d["pres_idx"]; cells = d["cells"]
rng = np.random.default_rng(1); bg = rng.choice(len(X_all), 20000, replace=False)
tr = np.r_[pi, bg]; X = X_all[tr]; y = np.r_[np.ones(len(pi)), np.zeros(len(bg))]
folds, _ = evaluate.spatial_blocks(cells[tr], cells[pi], 5)
ntrees = M.BRT().fit(X, y).n_trees
algos = ["maxent", "brt", "rf", "glm"]
def job(a, f):
    te = folds == f
    kw = {"n_trees": ntrees} if a == "brt" else {}
    return M.ALGORITHMS[a](**kw).fit(X[~te], y[~te]).predict(X[te])
base = M.N_THREADS
t = time.perf_counter(); [job(a, f) for a in algos for f in range(5)]; seq = time.perf_counter() - t
print(f"sequential, {base} threads per model: {seq:.2f}s", flush=True)
for workers, per in [(5, 2), (10, 2), (20, 1), (10, 4)]:
    M.N_THREADS = per
    t = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda af: job(*af), [(a, f) for a in algos for f in range(5)]))
    print(f"concurrent: {workers} jobs x {per} threads: {time.perf_counter() - t:.2f}s", flush=True)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
def pjob(af):
    M.N_THREADS = PER
    return job(*af)
for workers, per in [(10, 2), (20, 1), (20, 2)]:
    PER = per
    t = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        list(ex.map(pjob, [(a, f) for a in algos for f in range(5)]))
    print(f"processes: {workers} x {per} threads: {time.perf_counter() - t:.2f}s", flush=True)
