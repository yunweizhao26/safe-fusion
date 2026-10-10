#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
root="$PWD/artifacts/paper_evidence/review_round4/scale_caps"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export XDG_CACHE_HOME="$root/cache" MPLCONFIGDIR="$root/cache/matplotlib" NUMBA_CACHE_DIR="$root/cache/numba"
export TMPDIR="$root/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
stage="$1" genes="$2" variant="${3:-default}"
base="$root/genes_$genes"
source=$(cat "$base/source.txt")
export PYTHONPATH="$PWD/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python - "$root/code/source_sha256.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
for p, expected in json.loads(Path(sys.argv[1]).read_text()).items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected, f'Source drift: {p}'
PY
teachers=()
for method in gene_median svd_impute graph_smooth magic scvi; do
    teachers+=(--teacher-contract "$source/transductive/$method")
done
value_cap=600000; selector_cap=2000000
if [[ "$variant" == large ]]; then
    .venv/bin/python - "$root" <<'PY'
import json,sys
from pathlib import Path
for genes in [2000,5000]:
    assert json.loads((Path(sys.argv[1])/f'genes_{genes}/reproduction.json').read_text())['passed']
PY
    value_cap=6000000; selector_cap=20000000
fi
case "$stage" in
    value)
        .venv/bin/python "scripts/analyses/scale_caps/measure.py" "$base/${variant}_value.resources.json" .venv/bin/python "scripts/analyses/scale_caps/run_leakage_safe_method.py" \
            --method safe_fusion --transductive --input "$source/masked/corrupted.h5ad" \
            --coordinates "$source/masked/coordinates.parquet" --splits "$source/splits.parquet" \
            --output "$base/$variant/safe_fusion" --seed 1729 "${teachers[@]}" --max-value-fit-entries "$value_cap"
        ;;
    selector)
        .venv/bin/python "scripts/analyses/scale_caps/measure.py" "$base/${variant}_selector.resources.json" .venv/bin/python "scripts/analyses/scale_caps/scale_selector.py" \
            --corrupted "$source/masked/corrupted.h5ad" --coordinates "$source/masked/coordinates.parquet" \
            --splits "$source/splits.parquet" --fusion-contract "$base/$variant/safe_fusion" "${teachers[@]}" \
            --fit-split development --output-dir "$base/$variant/selector" --seed 1729 --max-fit-rows "$selector_cap"
        ;;
    check)
        .venv/bin/python "scripts/analyses/scale_caps/measure.py" "$base/reproduction.resources.json" .venv/bin/python "scripts/analyses/scale_caps/check_reproduction.py" "$base"
        ;;
    evaluate)
        .venv/bin/python "scripts/analyses/scale_caps/measure.py" "$base/evaluation.resources.json" .venv/bin/python "scripts/analyses/scale_caps/scale_evaluate.py" \
            --root "$base/evaluation" --sizes 200000 --draws 2000 --seed 1729
        ;;
    *) exit 2;;
esac
