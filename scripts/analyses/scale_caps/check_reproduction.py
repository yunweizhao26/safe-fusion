import json
import sys
from pathlib import Path
import numpy as np

base = Path(sys.argv[1])
source = Path((base / 'source.txt').read_text().strip()) / 'transductive'
report = {'passed': False, 'comparisons': {}}

def compare(name, left, right):
    assert left.shape == right.shape and left.dtype == right.dtype, name
    different, maximum = 0, 0.0
    for start in range(0, len(left), 256 if left.ndim == 2 else 1_000_000):
        stop = start + (256 if left.ndim == 2 else 1_000_000)
        a, b = left[start:stop], right[start:stop]
        different += int(np.count_nonzero(a != b))
        maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b))))
    report['comparisons'][name] = {'shape': list(left.shape), 'different': different, 'maximum_absolute_difference': maximum}

compare('fused_mean', np.load(source / 'safe_fusion/mean.npy', mmap_mode='r'),
        np.load(base / 'default/safe_fusion/mean.npy', mmap_mode='r'))
for name in ['test_score', 'test_fused_value']:
    compare(name, np.load(source / f'selector/{name}.npy', mmap_mode='r'),
            np.load(base / f'default/selector/{name}.npy', mmap_mode='r'))
with np.load(source / 'selector/test_candidates.npz') as old, np.load(base / 'default/selector/test_candidates.npz') as new:
    for name in ['rows', 'cols', 'labels']:
        compare(name, old[name], new[name])
report['passed'] = all(item['different'] == 0 for item in report['comparisons'].values())
(base / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
assert report['passed'], 'Default reproduction failed; larger-cap jobs must not run.'
