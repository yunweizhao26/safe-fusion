UNITS=(pancreas_0 pancreas_1 pancreas_2 colon adamson_crispri dixit_ko papalexi_eccite zebrafish papalexi_crossmodal)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)

unit_paths() {
  local key="$1"
  local pan=artifacts/pancreas_runs/0b2469810675-45c81b160d78
  local cf=artifacts/paper_evidence/pancreas_crossfit
  local norman=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
  local col=artifacts/colon_runs/0b2469810675-c0db6f963e94
  case "${key}" in
    pancreas_*)
      local fold="${key##*_}"
      input="${pan}/data/pancreas_islets/corrupted/mask_010.h5ad"
      coordinates="${pan}/data/pancreas_islets/coordinates/mask_010.parquet"
      splits="${cf}/fold_${fold}/splits.parquet"
      truth="${pan}/data/pancreas_islets/preprocessed.h5ad"
      methods_root="${cf}/fold_${fold}"
      ;;
    colon)
      input="${col}/data/colon_epithelial/corrupted/mask_010.h5ad"
      coordinates="${col}/data/colon_epithelial/coordinates/mask_010.parquet"
      splits="${col}/data/colon_epithelial/splits.parquet"
      truth="${col}/data/colon_epithelial/preprocessed.h5ad"
      methods_root="${col}/methods/standardized/colon_epithelial/mask_010"
      ;;
    norman_crispra)
      input="${norman}/corrupted.h5ad"
      coordinates="${norman}/coordinates.parquet"
      splits="${norman}/splits.parquet"
      truth="${norman}/prepared.h5ad"
      methods_root="${norman}/methods"
      ;;
    adamson_crispri|dixit_ko|papalexi_eccite)
      local root="artifacts/external_perturbseq/${key}"
      input="${root}/corrupted.h5ad"
      coordinates="${root}/coordinates.parquet"
      splits="${root}/splits.parquet"
      truth="external_data/prepared/${key}.h5ad"
      methods_root="${root}"
      ;;
    zebrafish)
      local root=artifacts/external_trajectory/zebrafish
      input="${root}/corrupted.h5ad"
      coordinates="${root}/coordinates.parquet"
      splits="${root}/splits.parquet"
      truth=external_data/prepared/zebrafish_trajectory.h5ad
      methods_root="${root}"
      ;;
    papalexi_crossmodal)
      local root=artifacts/paper_evidence/papalexi_crossmodal/benchmark
      input="${root}/corrupted.h5ad"
      coordinates="${root}/coordinates.parquet"
      splits="${root}/splits.parquet"
      truth=external_data/prepared/papalexi_eccite_crossmodal.h5ad
      methods_root="${root}"
      ;;
    *)
      echo "unknown dataset ${key}" >&2
      return 2
      ;;
  esac
}

fused_value_contract() {
  if [[ "$1" == boosted ]]; then
    echo safe_fusion
  else
    echo "safe_fusion_$1"
  fi
}

teacher_contract_args() {
  local root="$1"
  local name
  for name in "${TEACHERS[@]}"; do
    printf -- '--teacher-contract\n%s\n' "${root}/${name}"
  done
}
