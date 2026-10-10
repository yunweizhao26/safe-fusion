
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
O = (Path.cwd() / 'artifacts/paper_evidence/review_round4/main_tables_v3')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from masked_f1_units import Unit, load_units_manifest, load_unit, count_scale_values
from compute_matched_baseline_f1_curves import tie_broken_order

spec_path = O.parent/'round2_extras/part_c/units.json'
specs = json.loads(spec_path.read_text())
methods = ['DCA', 'scImpute', 'scRecover']
entries = []
for spec in specs:
    if spec['seed'] != 1729:
        continue
    row = {key: value for key, value in spec.items() if key in Unit.__dataclass_fields__}
    dca = Path(spec['extras']['DCA'])
    row['contracts'] = {m: str(dca.parent.parent/m.lower()/dca.name) for m in methods}
    entries.append(row)
assert len(entries) == 7
manifest = O/'added_units.json'
manifest.write_text(json.dumps(entries, indent=2)+'\n')
rows, sources = [], {}
def record(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block)
    sources[str(path.relative_to(ROOT) if path.is_absolute() else path)] = sha.hexdigest()
record(spec_path)
for unit in load_units_manifest(manifest):
    data = load_unit(unit)
    rr, cc = np.where((data.counts == 0) & (data.split == 'test')[:, None])
    labels = data.masked[rr, cc]
    names, codes = np.unique(np.char.add(unit.key+':', data.unit_labels[rr].astype(str)), return_inverse=True)
    positive = np.bincount(codes[labels], minlength=len(names))
    for method in methods:
        scores, _ = count_scale_values(unit.contracts[method], data, rr, cc)

        order = tie_broken_order(scores.astype(np.float64), unit.tie_seed)
        for fraction in np.arange(1, 11)/100:
            chosen = order[:max(1, round(fraction*len(order)))]
            selected = np.bincount(codes[chosen], minlength=len(names))
            hits = np.bincount(codes[chosen[labels[chosen]]], minlength=len(names))
            rows.extend(dict(dataset=unit.dataset, method=method, group=unit.key, unit=name,
                             fraction=fraction, n_selected=s, n_true_positive=h, n_masked_positives=p)
                        for name, s, h, p in zip(names, selected, hits, positive))
        for path in [unit.contracts[method]/'mean.npy', unit.contracts[method]/'metadata.json']:
            record(path)
        print('Frozen counts:', unit.key, method, flush=True)
    for path in [unit.corrupted, unit.splits, unit.coordinates]:
        record(path)
    del data, rr, cc, labels, scores, order, codes
    gc.collect()
pd.DataFrame(rows).to_parquet(O/'added_unit_counts.parquet', index=False)
for path in ['scripts/masked_f1_units.py','scripts/compute_matched_baseline_f1_curves.py',
             'scripts/standard_imputers/evaluate_r3_masked.py', ROOT/'scripts/analyses/main_tables_v3/prepare_added_counts.py']:
    record(path)
(O/'added_input_sha256.json').write_text(json.dumps(sources, indent=2)+'\n')
