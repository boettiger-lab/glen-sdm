import numpy as np, pandas as pd, sys, time, warnings
warnings.filterwarnings("ignore")
from hexsdm.models import MaxEnt
d = np.load(sys.argv[1]); X_all = d["X_all"].astype(np.float64); pi = d["pres_idx"]; cols=list(d["cols"])
rng = np.random.default_rng(1); bg = rng.choice(len(X_all), 10000, replace=False)
X = np.vstack([X_all[pi], X_all[bg]]); y = np.r_[np.ones(len(pi)), np.zeros(len(bg))]
ev = rng.choice(len(X_all), 50000, replace=False)
pd.DataFrame(X, columns=cols).assign(p=y).to_csv(f"{sys.argv[2]}/train.csv", index=False)
pd.DataFrame(X_all[ev], columns=cols).to_csv(f"{sys.argv[2]}/eval.csv", index=False)
t=time.time(); m=MaxEnt().fit(X,y); print("py fit", round(time.time()-t,2), "nnz", m.n_features_used)
np.savetxt(f"{sys.argv[2]}/py_pred.txt", m.predict(X_all[ev]))
