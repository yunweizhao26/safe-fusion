import json
from pathlib import Path
import pandas as pd

O = Path('artifacts/paper_evidence/review_round4/protein_multi/corrected')
audit = json.loads((O / 'evaluation/final_audit.json').read_text())
assert audit['passed']
pairs = pd.read_csv(O / 'selection/selected_pairs.csv')
summary = pd.read_csv(O / 'evaluation/summary.csv')
text = ['# Hao RNA-protein pairs', '', f'{len(pairs)} pairs; {pairs.gene.nunique()} genes.', '',
        'The corrected selection includes pairs with two detected training cells. '
        'The selector uses balanced sampling when positives exceed its training cap. '
        'Intervals condition on held-out donors P7/P8; pair summaries are descriptive.', '',
        '```csv\n' + summary.to_csv(index=False) + '```', '',
        'Complete estimates: evaluation/per_pair.csv. Audit: evaluation/final_audit.json.', '']
(O / 'README.md').write_text('\n'.join(text))
