from common import *
import json
import os
import numpy as np
import pandas as pd
import fusion_value_bootstrap as b
import fusion_value_selectors as f

dataset = ['Pancreas', 'Colon', 'CRISPRa'][int(os.environ.get('SLURM_ARRAY_TASK_ID', sys.argv[1]))]
uu = [u for u in units() if u.dataset == dataset]
root = fusion_root(uu[0])
overlay = OUT / 'masked/selector_overlay'
for u in uu:
    for directory in (root / 'selectors' / u.key).iterdir():
        link(directory, overlay / u.key / directory.name)
    subset = 'colon' if dataset == 'Colon' else 'pancreas_norman'
    for directory in (OUT / 'item3_selectors' / subset / u.key).iterdir():
        link(directory, overlay / u.key / directory.name)

all_variants = f.variants()
selected = [v for v in all_variants if v.family in f.DEFAULT_FAMILIES]
b.variants = lambda: selected
dest = OUT / 'masked' / dataset
sys.argv = ['fusion_value_bootstrap.py', '--units-manifest', str(CC / 'units_manifest.json' if dataset == 'Colon' else LF / 'units_manifest.json'),
            '--unit-keys', *[u.key for u in uu], '--selector-root', str(overlay), '--probability-root', str(root / 'nonzero_probability'),
            '--output-dir', str(dest), '--draws', '2000', '--seed', '1729']
b.main()
checks = []
for name, keys, columns in [('absolute.csv', ['dataset','method','statistic'], ['estimate_percent','lower_percent','upper_percent']),
                            ('paired_differences.csv', ['dataset','reference','comparator','statistic'], ['estimate_pp','lower_pp','upper_pp'])]:
    old = pd.read_csv(root / 'evaluation' / name)
    new = pd.read_csv(dest / name)
    merged = old.merge(new, on=keys, suffixes=('_old','_new'))
    error = max(float(np.max(np.abs(merged[c+'_old']-merged[c+'_new']))) for c in columns)
    assert len(merged) == len(new) and error <= 1.00001e-4, (name, len(merged), len(new), error)
    checks.append(dict(file=name, rows=len(merged), maximum_absolute_difference=error))
(dest / 'parity.json').write_text(json.dumps(checks, indent=2)+'\n')

r3 = E / 'review_round3/comparators/evaluation' / ('masked_colon_crossfit' if dataset == 'Colon' else 'masked')
new_abs = pd.read_csv(dest / 'absolute.csv')
old_abs = pd.read_csv(r3 / 'absolute.csv')
old_abs['dataset'] = old_abs.dataset.replace({'Colon cross-fit': 'Colon'})
shared = new_abs.merge(old_abs, on=['dataset','method','statistic'], suffixes=('_new','_old'))
for c in ['estimate_percent','lower_percent','upper_percent']:
    assert np.allclose(shared[c+'_new'], shared[c+'_old'], atol=1.00001e-4, rtol=0), c
for name in ['absolute.csv', 'paired_differences.csv']:
    table = pd.read_csv(r3 / name)
    table['dataset'] = table.dataset.replace({'Colon cross-fit': 'Colon'})
    table[table.dataset == dataset].to_csv(dest / ('new_comparators_'+name), index=False)

selected = [v for v in all_variants if v.family in ('transductive','transductive_leave_one_out','transductive_architecture') or v.name == 'Safe Fusion']
b.method_families = lambda: {v.name: v.family for v in selected}
sys.argv[sys.argv.index('--output-dir')+1] = str(OUT / 'item3_evaluation' / dataset)
b.main()
