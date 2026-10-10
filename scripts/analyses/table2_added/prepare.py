import hashlib
import json
from pathlib import Path

root = Path.cwd()
out = root / 'artifacts/paper_evidence/review_round4/table2_added'
for directory in ['audit', 'evaluation', 'fits', 'logs', 'mpl']:
    (out / directory).mkdir(parents=True, exist_ok=True)
paths = list((root / 'scripts').rglob('*.py')) + list((root / 'scripts/standard_imputers').glob('*.R'))
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
path = out / 'source_sha256.json'
if path.exists():
    assert json.loads(path.read_text()) == hashes, 'Sources changed since preparation'
else:
    path.write_text(json.dumps(hashes, indent=2) + '\n')
print('Prepared known-zero analysis directories and source hashes')
