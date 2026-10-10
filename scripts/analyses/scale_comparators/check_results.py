#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path

root = (Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
repo = root.parents[3]
for name, digest in json.loads((root / 'source_sha256.json').read_text()).items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
for name in ('S28_table_PASS.json', 'selector_sampling_PASS.json', 'checks/selector_PASS.json'):
    assert (root / name).exists(), name
for n in (25000, 50000, 100000, 200000):
    assert (root / 'reproduction' / str(n) / 'PASS.json').exists(), n

def rows(name):
    with (root / name).open() as handle:
        return list(csv.DictReader(handle))

summary = {(r['cells'], r['method'], r['statistic']): r for r in rows('method_summary.csv')}
differences = [r for r in rows('paired_differences.csv') if r['reference'] == 'Safe Fusion']
keys = [(r['cells'], r['method'], r['statistic']) for r in differences]
assert len(keys) == len(set(keys)), 'Duplicate comparisons'
for r in differences:
    n, method, stat = r['cells'], r['method'], r['statistic']
    assert r['reference'] == 'Safe Fusion' and int(r['n_units']) == 24, r
    delta, lo, hi = (float(r[k]) for k in ('difference', 'lower', 'upper'))
    assert all(math.isfinite(x) for x in (delta, lo, hi)) and -1 <= lo <= hi <= 1, r
    expected = float(summary[n, 'Safe Fusion', stat]['estimate']) - float(summary[n, method, stat]['estimate'])
    assert abs(delta - expected) < 1e-12, r

methods = {'dca': 'DCA', 'selector_scvi': 'Selector on scVI', 'selector_magic': 'Selector on MAGIC',
           'scgpt': 'scGPT', 'enimpute': 'EnImpute', 'scimpute': 'scImpute', 'screcover': 'scRecover'}
resources = {(r['stage'], r['cells']): r for r in rows('resources.csv')}
for (stage, n), r in resources.items():
    if stage not in methods or r['state'] != 'COMPLETED':
        continue
    for stat in ('mean_f1_1_to_10', 'average_precision'):
        assert (n, methods[stage], stat) in keys, (stage, n, 'completed but unevaluated')
    assert int(r['cpus']) == 8 and float(r['sacct_peak_memory_gib']) > 0, r
    if stage.startswith('selector_'):
        assert r['partition'] == 'cs', r
    if stage in ('enimpute', 'scimpute', 'screcover'):
        assert r['time_limit'] == '1-00:00:00', r
    if stage == 'scgpt':
        assert 'gres/gpu=1' in r['allocated_tres'] and r['partition'] == 'l40s_public', r
record = {'source_hashes_unchanged': True, 'paired_comparisons_checked': len(differences),
          'completed_methods_all_evaluated': True, 'resource_constraints_checked': True}
(root / 'results_PASS.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
