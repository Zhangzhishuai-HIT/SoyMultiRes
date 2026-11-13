# BGLR marker regression (BayesB by default) for one fold (pattern of HapGAM's bglr_fit.R).
# usage: Rscript bglr_fit.R <fold_dir> [model] [nIter] [burnIn]
# Reads train/test genotype rows straight from data/processed/geno/dosage.npy (int8, row-major; header offset and shape
# in dosage.json), centres markers with the training means, writes <fold_dir>/<model>.pred.txt and <model>.done.json.
suppressMessages(library(BGLR))
args <- commandArgs(trailingOnly = TRUE)
fold_dir <- args[1]
model <- if (length(args) >= 2) args[2] else "BayesB"
n_iter <- if (length(args) >= 3) as.integer(args[3]) else 12000L
burn_in <- if (length(args) >= 4) as.integer(args[4]) else 2000L
root <- "/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/data/processed/geno"
meta <- paste(readLines(file.path(root, "dosage.json"), warn = FALSE), collapse = "")
p <- as.integer(sub('.*"shape": \\[[0-9]+, ([0-9]+)\\].*', "\\1", meta))
offset <- as.numeric(sub('.*"offset": ([0-9]+).*', "\\1", meta))

read_rows <- function(rows) {
  con <- file(file.path(root, "dosage.npy"), "rb"); on.exit(close(con))
  out <- matrix(0, length(rows), p)
  ord <- order(rows)                           # read in file order, put back in the requested order
  for (k in ord) {
    seek(con, where = offset + as.numeric(rows[k]) * p, origin = "start")
    out[k, ] <- readBin(con, "integer", n = p, size = 1, signed = TRUE)
  }
  out
}
X <- read_rows(scan(file.path(fold_dir, "train_rows.txt"), quiet = TRUE))
y <- scan(file.path(fold_dir, "y_train.txt"), quiet = TRUE)
mu_x <- colMeans(X)
X <- sweep(X, 2, mu_x)
set.seed(20261111)
t0 <- Sys.time()
fit <- BGLR(y = y, ETA = list(list(X = X, model = model, saveEffects = FALSE)),
            nIter = n_iter, burnIn = burn_in, thin = 10, verbose = FALSE,
            saveAt = file.path(fold_dir, paste0(model, "_")))
rm(X); invisible(gc())
Xt <- read_rows(scan(file.path(fold_dir, "test_rows.txt"), quiet = TRUE))
Xt <- sweep(Xt, 2, mu_x)
pred <- as.numeric(fit$mu + Xt %*% fit$ETA[[1]]$b)
write(pred, file.path(fold_dir, paste0(model, ".pred.txt")), ncolumns = 1)
mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
writeLines(sprintf('{"model": "%s", "n_train": %d, "n_test": %d, "p": %d, "nIter": %d, "burnIn": %d, "minutes": %.1f, "varE": %.5f}',
                   model, length(y), nrow(Xt), p, n_iter, burn_in, mins, fit$varE),
           file.path(fold_dir, paste0(model, ".done.json")))
cat(sprintf("done %s %s n=%d p=%d minutes=%.1f\n", model, fold_dir, length(y), p, mins))
