import numpy as np
from hexsdm import data, pipeline
import sys
con = data.connect()
pres8,_,info = data.load_occurrences(con, {"gbif":{"species":"Ambystoma californiense"}})
dom = data.build_domain(con, pres8, None)
env, cols = data.load_covariates(con, pipeline.DEFAULTS["covariates"], 8, dom)
pres_idx = env.reset_index().merge(pres8.rename(columns={"h8":"cell"}).astype({"cell":"uint64"}), on="cell")["index"].to_numpy()
np.savez(sys.argv[1], X_all=env[cols].to_numpy(), cells=env.cell.to_numpy(), pres_idx=pres_idx, cols=np.array(cols))
