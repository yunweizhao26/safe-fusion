import json
from pathlib import Path

O = Path('artifacts/paper_evidence/review_round4/protein_depth')
audit = json.loads((O / 'runtime/final_audit.json').read_text())
assert audit['passed']
text = ['# Protein library-size analysis', '',
        'All 41 Hao pairs and both Papalexi benchmarks use 2,000 paired bootstrap draws. '
        'The partial Spearman endpoint adjusts ranked scores and protein for ranked log library size. '
        'Standard-depth scores fix library size to the training median and rescale count proposals. '
        'Residual-logit scores remove a linear log-library-size effect without protein outcomes.', '',
        'Per-pair and aggregate tables retain undefined correlations. '
        'Across-pair intervals are descriptive; Hao cell intervals condition on P7/P8.', '']
for path in sorted((O / 'tables/markdown').glob('*.md')):
    text.extend(['## ' + path.stem.replace('_', ' '), '', path.read_text()])
text.extend(['## Library-size observation', '', (O / 'runtime/verified_observation.json').read_text(), ''])
text.extend(['## Validation', '', json.dumps(audit, indent=2), ''])
(O / 'README.md').write_text('\n'.join(text))
