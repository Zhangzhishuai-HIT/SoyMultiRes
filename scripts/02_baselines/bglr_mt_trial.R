# BGLR Multitrait timing trial (week 1, plan step 5 fallback): five SCN races, N accessions, RKHS on the VanRaden kernel,
# unstructured genetic covariance, diagonal residual covariance (races were scored in separate experiments).
# usage: Rscript bglr_mt_trial.R <trial_dir> [nIter]
suppressMessages(library(BGLR))
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]; n_iter <- if (length(args) >= 2) as.integer(args[2]) else 200L
Y <- as.matrix(read.delim(file.path(d, "Y.tsv"), row.names = 1, check.names = FALSE))
n <- nrow(Y)
K <- matrix(readBin(file.path(d, "K.bin"), "double", n = n * n), n, n)
Y <- scale(Y)
t0 <- Sys.time()
# explicit priors: BGLR's default S0 comes from pairwise phenotypic covariances, which are not positive definite when
# traits share few accessions (e.g. 83 accessions for races 1 and 4). Prior means 0.5 on the scaled traits.
tt <- ncol(Y)
fm <- Multitrait(y = Y, ETA = list(list(K = K, model = "RKHS",
                                        Cov = list(type = "UN", df0 = tt + 2, S0 = diag(0.5, tt)))),
                 resCov = list(type = "DIAG", df0 = rep(5, tt), S0 = rep(1.5, tt)),
                 nIter = n_iter, burnIn = as.integer(n_iter / 4), thin = 5,
                 verbose = FALSE, saveAt = file.path(d, "bglrmt_"))
t1 <- Sys.time()
G <- fm$ETA[[1]]$Cov$Omega
cat(sprintf('{"n": %d, "traits": %d, "nIter": %d, "seconds": %.1f, "s_per_iter": %.3f}\n', n, ncol(Y), n_iter,
            as.numeric(difftime(t1, t0, units = "secs")), as.numeric(difftime(t1, t0, units = "secs")) / n_iter))
cat("genetic correlations (posterior mean):\n"); print(round(cov2cor(G), 2))
