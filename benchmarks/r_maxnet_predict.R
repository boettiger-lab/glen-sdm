args <- commandArgs(TRUE); library(maxnet)
tr <- read.csv(file.path(args[1], "train.csv")); ev <- read.csv(file.path(args[1], "eval.csv"))
p <- tr$p; tr$p <- NULL
t <- system.time(m <- maxnet(p, tr, maxnet.formula(p, tr, classes="lqph")))
cat("R fit", t[["elapsed"]], "nnz", sum(m$betas != 0), "\n")
write(predict(m, ev, type="cloglog"), file.path(args[1], "r_pred.txt"), ncolumns=1)
