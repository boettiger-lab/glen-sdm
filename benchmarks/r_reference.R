suppressMessages({library(maxnet); library(gbm); library(randomForest); library(dismo)})
d <- commandArgs(TRUE)[1]
tr <- read.csv(file.path(d, "train.csv")); ev <- read.csv(file.path(d, "eval.csv"))
p <- tr$p; X <- tr; X$p <- NULL
n <- sum(p); w <- ifelse(p == 1, 1, n / sum(1 - p))
tm <- function(expr) system.time(expr)[["elapsed"]]
cat("maxnet fit", tm(m1 <- maxnet(p, X, maxnet.formula(p, X, classes = "lqph"))), "predict", tm(predict(m1, ev, type = "cloglog")), "\n")
cat("gbm.step fit", tm(m2 <- gbm.step(data = tr, gbm.x = 1:ncol(X), gbm.y = ncol(tr), family = "bernoulli",
    tree.complexity = 5, learning.rate = 0.01, bag.fraction = 0.75, site.weights = w, silent = TRUE, plot.main = FALSE)),
    "trees", m2$n.trees, "predict", tm(predict(m2, ev, n.trees = m2$n.trees, type = "response")), "\n")
f <- as.factor(p)
cat("randomForest down-sampled fit", tm(m3 <- randomForest(X, f, ntree = 500, sampsize = c("0" = n, "1" = n), replace = TRUE)),
    "predict", tm(predict(m3, ev, type = "prob")), "\n")
fm <- as.formula(paste("p ~", paste(c(names(X), paste0("I(", names(X), "^2)")), collapse = "+")))
cat("glm fit", tm(m4 <- suppressWarnings(glm(fm, data = cbind(X, p = p), family = binomial, weights = w))), "\n")
