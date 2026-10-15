import numpy as np, pandas as pd, time, sys, warnings
warnings.filterwarnings("ignore")
from hexsdm.models import ALGORITHMS
d=sys.argv[1]; tr=pd.read_csv(f"{d}/train.csv"); ev=pd.read_csv(f"{d}/eval.csv").to_numpy(float)
y=tr.pop("p").to_numpy(float); X=tr.to_numpy(float)
for a in ["maxent","brt","rf","glm"]:
    t=time.time(); m=ALGORITHMS[a]().fit(X,y); tf=time.time()-t; t=time.time(); m.predict(ev); print(a, "fit", round(tf,2), "predict", round(time.time()-t,2), getattr(m,"n_trees",""))
