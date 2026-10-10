import json
from pathlib import Path
import subprocess
import sys
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

root = (Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras/C_scale')
checks = root / 'checks' / 'evaluator'
matrix = np.array([[0,2,0,1], [1,0,0,2], [0,0,3,0], [2,0,0,0], [0,1,0,0], [1,0,1,0]], dtype=np.float32)
rows, cols = np.where(matrix[2:] == 0)
rows = rows + 2
labels = np.zeros(len(rows), dtype=np.int8)
labels[[0, 3, 6, 9]] = 1
cell_ids = [f'cell{i}' for i in range(6)]
for version in ('original', 'streaming', 'fallback'):
    unit = checks / version / 'cells_6'
    (unit / 'masked').mkdir(parents=True, exist_ok=True)
    obs = pd.DataFrame({'donor': ['fit','fit','a','a','b','b']}, index=cell_ids)
    data = ad.AnnData(X=sparse.csr_matrix(matrix), obs=obs)
    data.layers['corrupted_counts'] = data.X.copy()
    data.write_h5ad(unit / 'masked' / 'corrupted.h5ad')
    pd.DataFrame({'cell_id':cell_ids, 'split':['development']*2+['test']*4}).to_parquet(unit / 'splits.parquet')
    pd.DataFrame({'cell_index':rows[labels == 1], 'gene_index':cols[labels == 1]}).to_parquet(unit / 'masked' / 'coordinates.parquet')
    for i, method in enumerate(('svd_impute', 'graph_smooth')):
        directory = unit / 'methods' / method
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'mean.npy', np.random.default_rng(1729+i).random(matrix.shape).astype(np.float32))
        (directory / 'metadata.json').write_text(json.dumps({'scale':'counts', 'cell_ids':cell_ids}))
    if version != 'fallback':
        directory = unit / 'selector'
        directory.mkdir(exist_ok=True)
        np.savez(directory / 'test_candidates.npz', rows=rows, cols=cols, labels=labels)
        np.save(directory / 'test_score.npy', np.linspace(1,0,len(labels),dtype=np.float32))
        (directory / 'selector_report.json').write_text('{}')
    script = Path('scripts/scale_evaluate.py') if version == 'original' else root / 'code/scale_evaluate.py'
    subprocess.run([sys.executable, str(script), '--root', str(checks / version), '--sizes', '6'], check=True,
                   stdout=subprocess.DEVNULL)
for name in ('method_summary.csv', 'paired_differences.csv', 'masked_f1_curves.csv'):
    assert (checks / 'original/results' / name).read_bytes() == (checks / 'streaming/results' / name).read_bytes(), name
expected = pd.read_csv(checks / 'original/results/method_summary.csv')
expected = expected[expected.method != 'Safe Fusion'].reset_index(drop=True)
actual = pd.read_csv(checks / 'fallback/results/method_summary.csv')
pd.testing.assert_frame_equal(expected, actual)
print('Original/streaming metric tables byte-identical; fallback comparator metrics/2000-draw intervals exactly equal.')
selected = checks / 'selected'
selected.mkdir(exist_ok=True)
link = selected / 'cells_6'
if not link.exists():
    link.symlink_to('../streaming/cells_6')
subprocess.run([sys.executable, str(root / 'code/scale_evaluate.py'), '--root', str(selected), '--sizes', '6',
                '--method', 'SVD'], check=True, stdout=subprocess.DEVNULL)
expected = pd.read_csv(checks / 'original/results/method_summary.csv')
pd.testing.assert_frame_equal(expected[expected.method == 'SVD'].reset_index(drop=True),
                              pd.read_csv(selected / 'results/method_summary.csv'))
print('Method-filtered evaluation exactly matches the corresponding original estimates and intervals.')
