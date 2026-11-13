# MegaLMM timing trial (week 1, plan step 5): five SCN races, N accessions, ~ 1 + (1|ID) with the VanRaden kernel.
# usage: Rscript megalmm_trial.R <trial_dir> [K_factors] [n_iter]
suppressMessages(library(MegaLMM))
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]; Kf <- if (length(args) >= 2) as.integer(args[2]) else 5L; n_iter <- if (length(args) >= 3) as.integer(args[3]) else 100L
Y <- as.matrix(read.delim(file.path(d, "Y.tsv"), row.names = 1, check.names = FALSE))
ids <- readLines(file.path(d, "ids.txt")); n <- length(ids)
K <- matrix(readBin(file.path(d, "K.bin"), "double", n = n * n), n, n); dimnames(K) <- list(ids, ids)
dat <- data.frame(ID = ids)
t0 <- Sys.time()
rp <- MegaLMM_control(h2_divisions = 20, burn = 0, thin = 2, K = Kf, max_NA_groups = ncol(Y) + 1, scale_Y = TRUE)
st <- setup_model_MegaLMM(Y = Y, formula = ~ 1 + (1 | ID), data = dat, relmat = list(ID = K), run_parameters = rp,
                          run_ID = file.path(d, "run"))
pri <- MegaLMM_priors(tot_Y_var = list(V = 0.5, nu = 5), tot_F_var = list(V = 18/20, nu = 20),
                      h2_priors_resids_fun = function(h2s, n) 1, h2_priors_factors_fun = function(h2s, n) 1,
                      Lambda_prior = list(sampler = sample_Lambda_prec_ARD, Lambda_df = 3,
                                          delta_1 = list(shape = 2, rate = 1), delta_2 = list(shape = 3, rate = 1),
                                          delta_iterations_factor = 100))
st <- set_priors_MegaLMM(st, pri)
maps <- make_Missing_data_map(st, max_NA_groups = ncol(Y) + 1, verbose = FALSE)
st <- set_Missing_data_map(st, maps$Missing_data_map)
st <- initialize_variables_MegaLMM(st)
cat("memory estimate:\n"); print(estimate_memory_initialization_MegaLMM(st))
t1 <- Sys.time()
st <- initialize_MegaLMM(st, verbose = FALSE)
t2 <- Sys.time()
st$Posterior$posteriorSample_params <- c("Lambda", "F_h2")
st$Posterior$posteriorMean_params <- "Eta_mean"
st <- clear_Posterior(st)
st <- sample_MegaLMM(st, n_iter)
t3 <- Sys.time()
cat(sprintf('{"n": %d, "traits": %d, "factors": %d, "setup_s": %.1f, "init_s": %.1f, "sample_s": %.1f, "s_per_iter": %.3f}\n',
            n, ncol(Y), Kf, as.numeric(difftime(t1, t0, units = "secs")), as.numeric(difftime(t2, t1, units = "secs")),
            as.numeric(difftime(t3, t2, units = "secs")), as.numeric(difftime(t3, t2, units = "secs")) / n_iter))
