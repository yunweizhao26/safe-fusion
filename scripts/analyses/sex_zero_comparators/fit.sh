#!/usr/bin/env bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=artifacts/paper_evidence/review_round4/sex_zero_comparators/logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUT=${SEX_ZERO_OUTPUT:-artifacts/paper_evidence/review_round4/sex_zero_comparators}
units=(pancreas_0 pancreas_1 pancreas_2 colon_0 colon_1 colon_2)
unit=${units[SLURM_ARRAY_TASK_ID]}
base=artifacts/paper_evidence/review_round4/transductive_references/sex_zeros/deployment_rebuilt/${unit}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""
dest=${OUT}/fits/${METHOD}/${unit}
if [[ -f ${dest}/mean.npy ]]; then echo "Existing fit: ${dest}"; exit 0; fi
case ${METHOD} in
  alra)
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    .venv/bin/python scripts/run_alra_baseline.py --corrupted "${base}/hybrid.h5ad" --splits "${base}/splits.parquet" --output "${dest}" --seed 1729
    ;;
  saver)
    .venv/bin/python scripts/run_saver_baseline.py --corrupted "${base}/hybrid.h5ad" --splits "${base}/splits.parquet" --output "${dest}" --ncores "${SLURM_CPUS_PER_TASK}"
    ;;
  kcluster)
    .venv-scanpy/bin/python scripts/standard_imputers/r3_kcluster.py --corrupted "${base}/hybrid.h5ad" --output "${OUT}/kcluster/${unit}.json" --seed 1729
    ;;
  dca|scimpute|enimpute)
    .venv/bin/python scripts/standard_imputers/run_r3_comparator.py --method "${METHOD}" --corrupted "${base}/hybrid.h5ad" --coordinates "${base}/coordinates.parquet" --splits "${base}/splits.parquet" --output "${dest}" --kcluster-json "${OUT}/kcluster/${unit}.json" --ncores "${SLURM_CPUS_PER_TASK}" --seed 1729
    ;;
esac
