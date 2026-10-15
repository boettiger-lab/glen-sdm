import numpy as np, time, sys, warnings
warnings.filterwarnings("ignore")
from hexsdm.models import ALGORITHMS
from hexsdm import evaluate
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ThreadPoolExecutor
class SkBalancedRF:
    def fit(self, X, y, w=None, n_trees=500):
        rng = np.random.default_rng(1); P=np.flatnonzero(y==1); B=np.flatnonzero(y==0); n=len(P)
        p = X.shape[1]; mtry = max(1,int(np.sqrt(p)))
        def one(i):
            r=np.random.default_rng(i); idx=np.r_[r.choice(P,n),r.choice(B,n)]
            return DecisionTreeClassifier(max_features=mtry, random_state=i).fit(X[idx],y[idx])
        with ThreadPoolExecutor(16) as ex: self.trees=list(ex.map(one, range(n_trees)))
        return self
    def predict(self, X):
        with ThreadPoolExecutor(16) as ex: ps=list(ex.map(lambda t:t.predict_proba(X)[:,1], self.trees))
        return np.mean(ps,0)
ALGORITHMS["rf_sklearn"]=SkBalancedRF
d = np.load(sys.argv[1]); X_all = d["X_all"].astype(np.float64); pi = d["pres_idx"]; cells=d["cells"]
rng = np.random.default_rng(1)
bg = rng.choice(len(X_all), 20000, replace=False)
tr = np.r_[pi,bg]; X = X_all[tr]; y = np.r_[np.ones(len(pi)), np.zeros(len(bg))]
folds, br = evaluate.spatial_blocks(cells[tr], cells[pi], 5)
for a in sys.argv[2].split(","):
    oof=np.full(len(y),np.nan); t=time.time()
    for f in range(5):
        te=folds==f
        oof[te]=ALGORITHMS[a]().fit(X[~te],y[~te]).predict(X[te])
    print(f"{a:12s} cv {time.time()-t:6.1f}s", {k:round(v,3) for k,v in evaluate.metrics(oof[y==1],oof[y==0]).items()})
