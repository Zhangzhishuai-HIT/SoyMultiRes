# Step 5: BGLR Multitrait for one stress category and one fold (inputs: bglr_mt_prepare.py).
# RKHS on the VanRaden kernel (rows read straight from geno/grm.f32.npy), unstructured genetic covariance, diagonal
# residual covariance (traits were scored in separate experiments), explicit priors (BGLR's default S0 is built from
# pairwise phenotypic covariances, which are not positive definite when traits share few accessions; trial of 11-11).
# usage: Rscript bglr_mt_fit.R <dir> [nIter] [burnIn]
suppressMessages(library(BGLR))
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
n_iter <- if (length(args) >= 2) as.integer(args[2]) else 12000L
burn <- if (length(args) >= 3) as.integer(args[3]) else 2000L
m <- read.delim(file.path(d, "meta.txt"), header = FALSE, stringsAsFactors = FALSE)
meta <- setNames(as.list(m$V2), m$V1)
meta$offset <- as.numeric(meta$offset); meta$n_all <- as.numeric(meta$n_all)
Y <- as.matrix(read.delim(file.path(d, "Y.tsv"), row.names = 1, check.names = FALSE))
rows <- scan(file.path(d, "rows.txt"), quiet = TRUE)
n <- length(rows); N <- meta$n_all
con <- file(meta$grm, "rb")
K <- matrix(0, n, n)
o <- order(rows)
for (k in seq_along(o)) {                          # read rows in file order, keep the selected columns
  i <- o[k]
  seek(con, meta$offset + rows[i] * N * 4)
  K[i, ] <- readBin(con, "numeric", n = N, size = 4, endian = "little")[rows + 1]
}
close(con)
K <- (K + t(K)) / 2
mu <- colMeans(Y, na.rm = TRUE); sdv <- apply(Y, 2, sd, na.rm = TRUE)
Ys <- sweep(sweep(Y, 2, mu), 2, sdv, "/")
tt <- ncol(Ys)
t0 <- Sys.time()
if (tt == 1) {                                     # one-trait category: single-trait RKHS (same prior scale)
  fm <- BGLR(y = Ys[, 1], ETA = list(list(K = K, model = "RKHS")), nIter = n_iter, burnIn = burn, thin = 5,
             verbose = FALSE, saveAt = file.path(d, "bglr_"))
  P <- matrix(fm$yHat, ncol = 1)
} else {
  fm <- Multitrait(y = Ys, ETA = list(list(K = K, model = "RKHS",
                                           Cov = list(type = "UN", df0 = tt + 2, S0 = diag(0.5, tt)))),
                   resCov = list(type = "DIAG", df0 = rep(5, tt), S0 = rep(1.5, tt)),
                   nIter = n_iter, burnIn = burn, thin = 5, verbose = FALSE, saveAt = file.path(d, "bglrmt_"))
  P <- fm$ETAHat
}
P <- sweep(sweep(P, 2, sdv, "*"), 2, mu, "+")
colnames(P) <- colnames(Y); rownames(P) <- rownames(Y)
write.table(P, file.path(d, "pred.tsv"), sep = "\t", quote = FALSE, col.names = NA)
if (tt > 1) {
  G <- round(cov2cor(fm$ETA[[1]]$Cov$Omega), 3); dimnames(G) <- list(colnames(Y), colnames(Y))
  write.table(G, file.path(d, "genetic_cor.tsv"), sep = "\t", quote = FALSE, col.names = NA)
}
cat(sprintf('{"n": %d, "traits": %d, "nIter": %d, "burnIn": %d, "minutes": %.1f}\n', n, tt, n_iter, burn,
            as.numeric(difftime(Sys.time(), t0, units = "mins"))), file = file.path(d, "done.json"))
