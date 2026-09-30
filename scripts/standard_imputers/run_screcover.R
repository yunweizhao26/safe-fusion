#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
work <- normalizePath(args[1])
kcluster <- as.integer(args[2])
ncores <- as.integer(args[3])
seed <- as.integer(args[4])

parallel_namespace <- asNamespace("parallel")
unlockBinding("detectCores", parallel_namespace)
assign("detectCores", function(all.tests = FALSE, logical = TRUE) ncores + 2L, envir = parallel_namespace)
lockBinding("detectCores", parallel_namespace)

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
  library(BiocParallel)
  library(scRecover)
})

genes <- readLines(file.path(work, "genes.txt"))
cells <- readLines(file.path(work, "cells.txt"))
counts <- as.matrix(readMM(file.path(work, "counts.mtx")))

gene_names <- paste0("g", seq_along(genes))
cell_names <- paste0("c", seq_along(cells))
dimnames(counts) <- list(gene_names, cell_names)
output_dir <- paste0(file.path(work, "screcover"), "/")

run_screcover <- scRecover::scRecover
body(run_screcover) <- as.call(c(as.list(body(run_screcover)), quote(
  saveRDS(list(P_dropout_cc_list = P_dropout_cc_list, clust = clust, dropoutNum = dropoutNum),
          paste0(outputDir, "dropout_probability.rds")))))

set.seed(seed)
started <- proc.time()[["elapsed"]]
run_screcover(counts = counts, Kcluster = kcluster, outputDir = output_dir, depth = 20, SAVER = FALSE,
          MAGIC = FALSE, UMI = FALSE, parallel = TRUE, BPPARAM = MulticoreParam(workers = ncores),
          verbose = FALSE)
elapsed <- proc.time()[["elapsed"]] - started

imputed <- as.matrix(read.csv(paste0(output_dir, "scRecover+scImpute.csv"), header = TRUE, row.names = 1,
                              check.names = FALSE))
stopifnot(identical(rownames(imputed), gene_names), identical(colnames(imputed), cell_names))
state <- readRDS(paste0(output_dir, "dropout_probability.rds"))
dropout <- matrix(0, nrow = length(genes), ncol = length(cells), dimnames = list(gene_names, cell_names))
for (block in state$P_dropout_cc_list) {
  dropout[rownames(block), colnames(block)] <- as.matrix(block)
}

writeBin(as.vector(imputed), file.path(work, "imputed.bin"))
writeBin(as.vector(dropout), file.path(work, "dropout.bin"))
clust <- state$clust
cluster_sizes <- as.list(table(clust, useNA = "ifany"))
names(cluster_sizes) <- ifelse(is.na(names(cluster_sizes)), "outlier", names(cluster_sizes))
report <- list(
  method = "scRecover",
  package_version = as.character(packageVersion("scRecover")),
  source = "Bioconductor 3.22 release (bioconda bioconductor-screcover 1.26.0); R code identical to 1.28.0 (Bioconductor 3.23)",
  r_version = R.version.string,
  dependencies = lapply(c(SAVER = "SAVER", preseqR = "preseqR", gamlss = "gamlss", pscl = "pscl", bbmle = "bbmle",
                          penalized = "penalized", kernlab = "kernlab", rsvd = "rsvd", BiocParallel = "BiocParallel"),
                        function(name) as.character(packageVersion(name))),
  settings = list(Kcluster = kcluster, depth = 20, SAVER = FALSE, MAGIC = FALSE, UMI = FALSE, parallel = TRUE,
                  workers = ncores, scimpute_drop_thre = 0.5, seed = seed),
  umi_note = "UMI = TRUE needs per-cell histograms of reads per UMI, which the processed data do not contain",
  n_clusters = sum(!is.na(unique(clust))),
  cluster_sizes = cluster_sizes,
  dropout_number = as.list(summary(state$dropoutNum)),
  dropout_probability = paste("scRecover P_dropout: for each gene and cell cluster, the probability under the",
                              "fitted ZINB that a zero comes from the negative binomial component; 0 for",
                              "nonzero entries, genes with no counts and outlier cells"),
  imputed_scale = "counts (scRecover+scImpute.csv)",
  elapsed_seconds = elapsed
)
write_json(report, file.path(work, "runner.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("scRecover done", elapsed, "seconds\n")
