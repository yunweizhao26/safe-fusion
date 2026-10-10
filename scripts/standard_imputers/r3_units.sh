R3_ROOT="${R3_ROOT:-artifacts/paper_evidence/review_round3/comparators}"
R3_MASKED=(pancreas_0 pancreas_1 pancreas_2 colon norman_crispra colon_cf_0 colon_cf_1 colon_cf_2)
R3_DEPLOY=(deploy_pancreas_0 deploy_pancreas_1 deploy_pancreas_2 deploy_colon deploy_zebrafish deploy_norman_crispra deploy_adamson_crispri deploy_papalexi_eccite)
R3_UNITS=("${R3_MASKED[@]}" "${R3_DEPLOY[@]}")

r3_unit_paths() {
  local unit="$1" root
  local lf=artifacts/paper_evidence/review_round2/leakage_free
  local deploy=artifacts/paper_evidence/downstream_deployment
  local colon=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial
  recorded=""
  case "${unit}" in
    pancreas_*) root="${lf}/pancreas_crossfit/fold_${unit##*_}" ;;
    norman_crispra) root="${lf}/norman_crispra" ;;
    colon_cf_*) root=artifacts/paper_evidence/review_round3/colon_crossfit/fold_${unit##*_} ;;
    colon)
      corrupted="${colon}/corrupted/mask_010.h5ad"
      coordinates="${colon}/coordinates/mask_010.parquet"
      splits="${colon}/splits.parquet"
      fits="${unit}"
      return 0
      ;;
    deploy_pancreas_*) root="${deploy}/pancreas/fold_${unit##*_}" ;;
    deploy_norman_crispra) root=artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt/deployment ;;
    deploy_*) root="${deploy}/${unit#deploy_}" ;;
    *)
      echo "unknown unit ${unit}" >&2
      return 2
      ;;
  esac
  if [[ "${unit}" == deploy_* ]]; then
    corrupted="${root}/hybrid.h5ad"
    recorded="${root}/recorded.h5ad"
  else
    corrupted="${root}/corrupted.h5ad"
  fi
  coordinates="${root}/coordinates.parquet"
  splits="${root}/splits.parquet"
  fits="${unit}"
}
