DEPLOY_KEYS=(pancreas_0 pancreas_1 pancreas_2 colon norman_crispra adamson_crispri dixit_ko papalexi_eccite zebrafish)
DEPLOY_ROOT="${DEPLOY_ROOT:-artifacts/paper_evidence/downstream_deployment}"

deployment_paths() {
  local key="$1"
  unit_paths "${key}"
  case "${key}" in
    pancreas_*) out="${DEPLOY_ROOT}/pancreas/fold_${key##*_}" ;;
    *) out="${DEPLOY_ROOT}/${key}" ;;
  esac
  deploy_methods="${out}/methods"
  fit=(--fit-split development)
  if [[ "${key}" == colon ]]; then
    fit=(--fit-split validation --fit-cells 3852)
  fi
}
