KNOCKDOWN_ROOT="${KNOCKDOWN_ROOT:-artifacts/paper_evidence/review_round2/knockdown}"
NORMAN_REBUILT="${NORMAN_REBUILT:-artifacts/paper_evidence/review_round2/leakage_free/norman_crispra}"
NORMAN_KNOCKDOWN="${NORMAN_KNOCKDOWN:-${KNOCKDOWN_ROOT}/norman_rebuilt}"

knockdown_paths() {
  local screen="$1"
  deployment_paths "${screen}"
  label_root="${methods_root}"
  kd_screen="${screen}"
  if [[ "${screen}" == norman_crispra ]]; then
    input="${NORMAN_REBUILT}/corrupted.h5ad"
    coordinates="${NORMAN_REBUILT}/coordinates.parquet"
    splits="${NORMAN_REBUILT}/splits.parquet"
    truth="${NORMAN_REBUILT}/prepared.h5ad"
    methods_root="${NORMAN_REBUILT}/methods"
    label_root="${NORMAN_KNOCKDOWN}/masked"
    out="${NORMAN_KNOCKDOWN}/deployment"
    deploy_methods="${out}/methods"
  fi
}

kd_standard_output() {
  if [[ "${kd_screen}" == norman_crispra ]]; then
    echo "${NORMAN_KNOCKDOWN}/standard_imputers/$1"
  else
    echo "${KNOCKDOWN_ROOT}/standard_imputers/$1/${kd_screen}"
  fi
}
