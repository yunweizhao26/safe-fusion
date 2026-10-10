#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
work <- normalizePath(args[1])
kcluster <- as.integer(args[2])
ncores <- as.integer(args[3])
seed <- as.integer(args[4])
drop_thre <- 0.5

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
  library(scImpute)
})

genes <- readLines(file.path(work, "genes.txt"))
cells <- readLines(file.path(work, "cells.txt"))
counts <- as.matrix(readMM(file.path(work, "counts.mtx")))
dimnames(counts) <- list(genes, cells)
count_path <- file.path(work, "counts.rds")
saveRDS(counts, count_path)
out_dir <- paste0(file.path(work, "scimpute"), "/")
dir.create(out_dir, showWarnings = FALSE)

set.seed(seed)
started <- proc.time()[["elapsed"]]
outliers <- scimpute(count_path = count_path, infile = "rds", outfile = "rds", type = "count",
                     out_dir = out_dir, labeled = FALSE, drop_thre = drop_thre,
                     Kcluster = kcluster, ncores = ncores)
elapsed <- proc.time()[["elapsed"]] - started
imputed <- readRDS(paste0(out_dir, "scimpute_count.rds"))
stopifnot(identical(rownames(imputed), genes), identical(colnames(imputed), cells))

lnorm_dir <- paste0(file.path(work, "lnorm"), "/")
dir.create(lnorm_dir, showWarnings = FALSE)
count_lnorm <- scImpute:::read_count(filetype = "rds", path = count_path, out_dir = lnorm_dir,
                                     type = "count", genelen = NULL)
clust <- readRDS(paste0(out_dir, "clust.rds"))
nclust <- sum(!is.na(unique(clust)))
dropout <- matrix(0, nrow = length(genes), ncol = length(cells))
valid_per_cluster <- integer(0)
for (cc in seq_len(nclust)) {
  members <- which(clust == cc)
  if (length(members) <= 1) next
  pars <- readRDS(paste0(out_dir, "pars", cc, ".rds"))
  valid <- scImpute:::find_va_genes(pars, subcount = count_lnorm[, members])
  valid_per_cluster[as.character(cc)] <- length(valid)
  if (length(valid) <= 10) next
  sub <- count_lnorm[valid, members, drop = FALSE]
  p <- pars[valid, , drop = FALSE]
  droprate <- t(sapply(seq_along(valid), function(i) scImpute:::calculate_weight(sub[i, ], p[i, ])[, 1]))
  mucheck <- sweep(sub, MARGIN = 1, p[, "mu"], FUN = ">")
  droprate[mucheck & droprate > drop_thre] <- 0
  dropout[valid, members] <- droprate
}

writeBin(as.vector(as.matrix(imputed)), file.path(work, "imputed.bin"))
writeBin(as.vector(dropout), file.path(work, "dropout.bin"))
cluster_sizes <- as.list(table(clust, useNA = "ifany"))
names(cluster_sizes) <- ifelse(is.na(names(cluster_sizes)), "outlier", names(cluster_sizes))
report <- list(
  method = "scImpute",
  package_version = as.character(packageVersion("scImpute")),
  source = "GitHub Vivianstats/scImpute commit 78556ee2dc0a9c830223b9e3e9f99387cd3e09a4",
  r_version = R.version.string,
  dependencies = lapply(c(penalized = "penalized", kernlab = "kernlab", rsvd = "rsvd", doParallel = "doParallel"),
                        function(name) as.character(packageVersion(name))),
  settings = list(infile = "rds", type = "count", labeled = FALSE, drop_thre = drop_thre, Kcluster = kcluster,
                  ncores = ncores, seed = seed, genelen = NULL),
  n_clusters = nclust,
  cluster_sizes = cluster_sizes,
  n_outlier_cells = length(outliers),
  valid_genes_per_cluster = as.list(valid_per_cluster),
  dropout_probability = paste("weight of the dropout (gamma) component of the scImpute mixture of each gene",
                              "in each cell cluster, from the saved parameters; 0 for genes that scImpute",
                              "does not model in a cluster and for outlier cells"),
  imputed_scale = "counts (scimpute_count)",
  elapsed_seconds = elapsed
)
write_json(report, file.path(work, "runner.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("scImpute done", elapsed, "seconds\n")
