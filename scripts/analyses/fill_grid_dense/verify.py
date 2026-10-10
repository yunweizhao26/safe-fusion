
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

out = (Path.cwd() / 'artifacts/paper_evidence/review_round4/fill_grid_dense')
old = out.parent / 'fill_grid'

def read(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)

def key(row):
    return row['dataset'], row['endpoint'], row['method'], float(row['fill_pct'])

original_columns, original = read(old / 'fill_grid_long.csv')
columns, rows = read(out / 'fill_grid_dense_long.csv')
assert columns == original_columns + ['gap_closed_pct', 'gap_ci_low_pct', 'gap_ci_high_pct']
assert len(rows) == 728 and len({key(r) for r in rows}) == 728
pcts = [0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.5, 3, 4, 5, 7, 10]
methods = ['detection', 'transductive', 'svd', 'magic', 'scvi', 'alra', 'selector_scvi', 'selector_magic']
endpoints = {(r['dataset'], r['endpoint']) for r in rows}
assert len(endpoints) == 7
expected = {(d, e, m, p) for d, e in endpoints for m in methods for p in pcts}
lookup = {key(r): r for r in rows}
assert set(lookup) == expected
exact_values = 0
exact_rows = 0
for r in original:
    matches = []
    for column in ['error_reduction', 'ci_low', 'ci_high']:
        matches.append(r[column] == lookup[key(r)][column])
    exact_values += sum(matches)
    exact_rows += all(matches)
reproduction = json.loads((out / 'reproduction.json').read_text())
assert exact_values == reproduction['exact_printed_matches']
for r in original:
    for c in ['error_reduction', 'ci_low', 'ci_high']:
        value = lookup[key(r)][c]
        assert r[c] == value or (r['dataset'] == 'zebrafish' and r['endpoint'] == 'Stage error'
            and abs(float(r[c]) - float(value)) <= 5e-8
            and f'{100 * float(r[c]):.2f}' == f'{100 * float(value):.2f}')
for r in rows:
    assert float(r['gap_closed_pct']) == (float(r['error_reduction']) / float(r['error_unfilled'])) * 100
    assert all(math.isfinite(float(r[c])) for c in ['gap_closed_pct', 'gap_ci_low_pct', 'gap_ci_high_pct'])

archive = np.load(out / 'bootstrap_draws.npz', allow_pickle=False)
for d in json.loads((out / 'bootstrap_diagnostics.json').read_text()):
    p = d['array_prefix']
    numerator = archive[p + '_numerator']
    denominator = archive[p + '_denominator']
    ratio = archive[p + '_gap_pct']
    assert numerator.shape == ratio.shape == (104, 2000)
    assert denominator.shape == (2000,) and (denominator > 0).all()
    assert np.array_equal(ratio, numerator / denominator[None, :] * 100)
    for i, name in enumerate(archive[p + '_methods']):
        method, pct = name.rsplit('_', 1)
        row = lookup[d['dataset'], d['endpoint'], method, float(pct.removesuffix('pct'))]
        assert np.array_equal(np.nanquantile(ratio[i], [.025, .975]),
                              [float(row['gap_ci_low_pct']), float(row['gap_ci_high_pct'])])

checks = [r for p in out.glob('*/*/fill_checks.csv') for r in read(p)[1]]
assert len(checks) == 832
assert sum(r['saved_fill_bitwise_equal'] == 'True' for r in checks) == 192
assert all(r['saved_fill_bitwise_equal'] != 'False' for r in checks)
for r in checks:
    assert int(r['selected']) == max(1, round(float(r['fill_pct']) / 100 * int(r['candidates'])))
stage_sources = {}
for path in out.glob('*/*/source_sha256.json'):
    relative = path.relative_to(out)
    old_path = old / ('grid' if relative.parts[0] == 'dense' else relative.parts[0]) / Path(*relative.parts[1:])
    old_sources = json.loads(old_path.read_text())
    for name, digest in json.loads(path.read_text()).items():
        assert stage_sources.setdefault(name, digest) == digest, name
        assert old_sources[name] == digest, name
        h = hashlib.sha256()
        with Path(name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == digest, name
assert reproduction['passed']
receipt = dict(passed=True, rows=728, original_rows=392, exact_reproduced_rows=exact_rows,
    exact_printed_values=exact_values, max_absolute_difference=reproduction['max_absolute_difference'],
    same_draw_gap_intervals=728, filled_matrices=832, saved_matrices_bitwise_equal=192,
    input_files_matching_original_run=len(stage_sources))
(out / 'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
