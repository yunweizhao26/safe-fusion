#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
work_dir <- normalizePath(args[1])
kcluster <- as.integer(args[2])
ncores <- as.integer(args[3])
seed <- as.integer(args[4])

suppressMessages({
  library(Matrix)
  library(jsonlite)
  library(EnImpute)
})

counts <- as.matrix(readMM(file.path(work_dir, "counts.mtx")))
rownames(counts) <- readLines(file.path(work_dir, "genes.txt"))
colnames(counts) <- readLines(file.path(work_dir, "cells.txt"))

components <- c("ALRA", "DCA", "DrImpute", "MAGIC", "SAVER", "scImpute", "scRMD", "Seurat")
timing <- new.env()
for (component in components) {
  local({
    name <- component
    trace(
      paste0(name, ".EnImpute"),
      tracer = bquote(assign(.(name), proc.time()[["elapsed"]], envir = .(timing))),
      exit = bquote(assign(.(name), proc.time()[["elapsed"]] - get(.(name), envir = .(timing)), envir = .(timing))),
      where = asNamespace("EnImpute"),
      print = FALSE
    )
  })
}

settings <- list(
  scImpute.Kcluster = kcluster,
  SAVER.ncores = ncores,
  scImpute.ncores = ncores,
  MAGIC.n.jobs = ncores
)
setwd(work_dir)
set.seed(seed)
started <- proc.time()[["elapsed"]]
expressed <- rowSums(counts) > 0
result <- do.call(EnImpute, c(list(count = counts[expressed, , drop = FALSE]), settings))
elapsed <- proc.time()[["elapsed"]] - started

stopifnot(identical(dim(result$count.EnImpute.exp), c(sum(expressed), ncol(counts))),
          all(is.finite(result$count.EnImpute.exp)))
imputed <- matrix(0, nrow(counts), ncol(counts))
imputed[expressed, ] <- result$count.EnImpute.exp
writeBin(as.vector(imputed), file.path(work_dir, "imputed.bin"), size = 8)

zeros <- counts == 0
expressed_zeros <- counts[expressed, , drop = FALSE] == 0
individual <- lapply(seq_along(result$Methods.used), function(k) {
  values <- result$count.imputed.individual[, , k]
  zeros <- expressed_zeros
  list(
    positive_share_of_input_zeros = mean(values[zeros] > 0),
    median_value_at_input_zeros = median(values[zeros])
  )
})
names(individual) <- result$Methods.used

packages <- c("EnImpute", "Seurat", "SAVER", "scImpute", "scRMD", "DrImpute", "Rmagic", "rsvd",
              "reticulate", "glmnet", "penalized", "kernlab", "RSpectra", "corpcor", "Matrix", "lars")
python_packages <- c("dca", "tensorflow", "keras", "magic-impute", "graphtools", "scprep", "scanpy", "anndata", "numpy", "scipy", "scikit-learn")
python_versions <- fromJSON(system2(
  Sys.which("python"),
  c("-c", shQuote(paste0("import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in ",
                         toJSON(python_packages), "}))"))),
  stdout = TRUE
))
record <- list(
  method = "EnImpute",
  r_version = R.version.string,
  packages = setNames(lapply(packages, function(p) as.character(packageVersion(p))), packages),
  python = list(
    reticulate_executable = reticulate::py_config()$python,
    reticulate_version = reticulate::py_config()$version,
    dca_command = unname(Sys.which("dca")),
    packages = python_versions
  ),
  settings = c(settings, list(seed = seed, other_arguments = "EnImpute 1.1 package defaults")),
  methods_used = result$Methods.used,
  n_genes = nrow(counts),
  n_cells = ncol(counts),
  n_genes_without_counts_set_to_zero = sum(!expressed),
  input_zero_share = mean(zeros),
  enimpute_positive_share_of_input_zeros = mean(imputed[zeros] > 0),
  individual = individual,
  elapsed_seconds = elapsed,
  component_elapsed_seconds = mget(components, envir = timing, ifnotfound = list(NA)),
  output = "count.EnImpute.exp (exp(trimmed mean of log(1 + library-normalized imputed values at scale factor 1e4)) - 1), genes x cells, float64 column-major"
)
write_json(record, file.path(work_dir, "runner.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat(toJSON(record[c("methods_used", "elapsed_seconds", "component_elapsed_seconds", "enimpute_positive_share_of_input_zeros")], auto_unbox = TRUE), "\n")
