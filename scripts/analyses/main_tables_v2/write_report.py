import json
from pathlib import Path
import pandas as pd

O = Path.cwd() / 'artifacts/paper_evidence/review_round4/main_tables_v2'
f = pd.read_csv(O / 'estimates.csv')
verification = json.loads((O / 'verification.json').read_text())
assert verification['passed']
lines = ['# Manuscript estimates', '', 'Observed estimates ± bootstrap standard errors (2,000 draws, seed 1729, sample SD).', '']
blocks = [
    ('Safe Fusion', ['Safe Fusion, labels', 'Safe Fusion']),
    ('Published methods', ['EnImpute', 'MAGIC', 'DCA', 'scVI', 'SAVER', 'scImpute', 'ALRA', 'scGPT', 'scRecover']),
    ('Ablation, the Safe Fusion teachers', ['SVD', 'Weighted kNN']),
]
for table, part in f.groupby('table', sort=False):
    columns = part.column.drop_duplicates().tolist()
    lines += ['## ' + table, '', '| Method | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
    indexed = part.set_index(['method', 'column'])
    groups = blocks if table in ['Table 1', 'Table 2'] else [('Rankings', part.method.drop_duplicates().tolist())]
    for title, methods in groups:
        present = [m for m in methods if m in set(part.method)]
        if not present:
            continue
        lines.append('| **' + title + '** | ' + ' | '.join([''] * len(columns)) + ' |')
        for method in present:
            cells = []
            for column in columns:
                row = indexed.loc[(method, column)]
                cells.append(f'{row["mean"]:.2f} ± {row.bootstrap_se:.2f}')
            lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines.append('')
lines += ['## Validation', '', f'All {verification["point_checks"]} printed manuscript checks passed.',
          'Source hashes, full bootstrap draws and percentile intervals accompany the estimates.',
          'Uncertainty conditions on the saved models, masks, splits and reference definitions.', '']
(O / 'README.md').write_text('\n'.join(lines))
print(O / 'README.md')
