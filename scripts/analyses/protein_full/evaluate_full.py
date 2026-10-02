import importlib.util
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
O = ROOT / 'artifacts/paper_evidence/review_round4/protein_followup'
F = ROOT / 'artifacts/paper_evidence/review_round4/protein_full/full'
os.environ['SAFE_FUSION_ROOT'] = str(ROOT)
spec = importlib.util.spec_from_file_location('full_original', Path(__file__).with_name('evaluate.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for name in ['prepared.h5ad', 'corrupted.h5ad', 'coordinates.parquet', 'splits.parquet',
             'teachers', 'safe_fusion', 'svd_impute', 'graph_smooth', 'magic_standard', 'scvi']:
    path = O / 'full' / name
    if not path.exists():
        path.symlink_to(F / name)
for name in ['mlp_selector', 'global_fill_selector']:
    path = O / 'full' / name
    if not path.exists():
        path.symlink_to('selector')
extra = {'svd_all_cells': 'teachers/svd_impute', 'weighted_knn_all_cells': 'teachers/graph_smooth'}
m.e.VALUE_RANKINGS = {**m.e.VALUE_RANKINGS, **extra}
m.e.RNA_RANKINGS = (*m.e.RNA_RANKINGS, *extra)
m.e.RANKINGS = (*m.e.RANKINGS, *extra)
sys.argv = [sys.argv[0], '--prepared', str(F / 'prepared.h5ad'), '--benchmark-root', str(O / 'full'),
            '--output-dir', str(O / 'full/evaluation'), '--include-magic', '--workers', '3']
m.main()
