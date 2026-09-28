#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[1]
estimate_path <- args[2]
se_path <- args[3]
ncores <- as.integer(args[4])

counts <- as.matrix(Matrix::readMM(counts_path))
x <- t(counts)
result <- SAVER::saver(x, estimates.only = FALSE, ncores = ncores)
Matrix::writeMM(Matrix::Matrix(t(result$estimate), sparse = TRUE), estimate_path)
Matrix::writeMM(Matrix::Matrix(t(result$se), sparse = TRUE), se_path)
