#!/bin/bash
#SBATCH --job-name=sf_saver_nm
#SBATCH --output=logs/sf_saver_nm_%j.log
#SBATCH --error=logs/sf_saver_nm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# SAVER on the Norman CRISPRa data, restricted to the same 1,500-gene subset
# used by the GEARS/CPA sweeps (full 5,045-gene SAVER exceeds the 4 h limit).

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
PY="uv run python"

$PY - <<'PY'
import anndata as ad
import numpy as np
from scipy import sparse

a = ad.read_h5ad('artifacts/paper_evidence/norman_crispra/corrupted.h5ad')
X = a.layers['corrupted_counts']
if sparse.issparse(X):
    dense = X.toarray().astype(np.float32)
else:
    dense = np.asarray(X).astype(np.float32)
variance = dense.var(axis=0)
keep = np.sort(np.argsort(-variance)[:1500])
subset = ad.AnnData(X=sparse.csr_matrix(dense[:, keep]), obs=a.obs.copy(), var=a.var.iloc[keep].copy())
subset.obs_names = a.obs_names
subset.var_names = a.var_names[keep]
subset.layers['corrupted_counts'] = sparse.csr_matrix(dense[:, keep])
subset.write_h5ad('artifacts/paper_evidence/baselines/saver/norman_mask_010_subset.h5ad')
print('subset written', subset.shape)
PY

$PY -u scripts/run_saver_baseline.py \
  --corrupted artifacts/paper_evidence/baselines/saver/norman_mask_010_subset.h5ad \
  --splits artifacts/paper_evidence/norman_crispra/splits.parquet \
  --output artifacts/paper_evidence/baselines/saver/norman_mask_010

echo SAVER_NORMAN_DONE
