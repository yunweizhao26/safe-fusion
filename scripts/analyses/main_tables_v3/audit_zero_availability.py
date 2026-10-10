
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.cwd()
O = (Path.cwd() / 'artifacts/paper_evidence/review_round4/main_tables_v3')
files = subprocess.check_output(['rg','--files','--hidden','--no-ignore','artifacts'], text=True).splitlines()
inspected, matches, hashes = [], [], {}
for name in sorted(files):
    path = Path(name)
    if O.is_relative_to(ROOT/path) or path.is_relative_to(O.relative_to(ROOT)):
        continue
    if not any(s in name.lower() for s in ['zero','knockdown','auroc','per_target']):
        continue
    if path.suffix not in ['.csv','.parquet']:
        continue
    found = []
    if path.suffix == '.csv':
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'method' not in reader.fieldnames:
                continue
            for row in reader:
                if row['method'].lower() in ['scgpt','screcover']:
                    found.append(row)
    else:
        schema = pq.read_schema(path)
        score_columns = [c for c in schema.names if c.lower() in ['score:scgpt','score:screcover']]
        if 'method' in schema.names:
            methods = pd.read_parquet(path, columns=['method']).method.dropna().unique()
            found.extend({'method': str(m), 'evidence':'method column'} for m in methods if str(m).lower() in ['scgpt','screcover'])
        found.extend({'method': c.split(':',1)[1], 'evidence':'saved score column'} for c in score_columns)
        if not score_columns and 'method' not in schema.names:
            continue
    inspected.append(name)
    if found:
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        matches.extend(dict(source=name, **row) for row in found)
pd.DataFrame(matches).to_csv(O/'zero_availability_matches.csv', index=False)
(O/'zero_availability_inventory.json').write_text(json.dumps(dict(
    scope='All CSV method columns and Parquet method/score columns under artifacts, including hidden and ignored files, with zero/knockdown/auroc/per_target in path.',
    inspected=inspected, matched_sources=hashes, matches=len(matches)), indent=2)+'\n')
print('Zero availability:', len(inspected), 'tables inspected;', len(hashes), 'matching sources;', len(matches), 'rows')
