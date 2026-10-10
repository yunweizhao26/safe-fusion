import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
O = Path('artifacts/paper_evidence/review_round4/sle_transductive')
OLD = Path('artifacts/paper_evidence/sle_treg_case')
checks = []
def same(a,b):
    if isinstance(a,str):
        if a == b: return True
        if a.startswith('[') and b.startswith('['):
            try: return np.allclose(json.loads(a.replace('nan','NaN')),json.loads(b.replace('nan','NaN')),rtol=1e-10,atol=1e-10,equal_nan=True)
            except ValueError: pass
        return False
    return bool(np.isclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))
for name in json.loads((O/'units.json').read_text()):
    for original in (OLD/name/'fills').glob('*.npz'):
        with np.load(original) as a, np.load(O/'verification'/name/'fills'/original.name) as b:
            ok = a.files == b.files and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
            checks.append({'path':str(original.relative_to(OLD)), 'ok':ok,'kind':'fill_array_exact'})
for original in sorted((OLD/'results').glob('*/*.csv')):
    new = O/'verification'/original.relative_to(OLD)
    a,b = pd.read_csv(original),pd.read_csv(new)
    ok = list(a.columns)==list(b.columns) and a.shape==b.shape
    bad=[]
    if ok:
        for column in a:
            if not all(same(x,y) for x,y in zip(a[column],b[column])): bad.append(column)
        ok = not bad
    checks.append({'path':str(original.relative_to(OLD)), 'ok':ok,'different_columns':bad,'kind':'evaluator_replay'})
(O/'verification/checks.json').write_text(json.dumps(checks,indent=2)+'\n')
failed=[r for r in checks if not r['ok']]
print(json.dumps({'checks':len(checks),'failed':failed},indent=2))
assert not failed, 'Original inductive replay failed; do not trust transductive evaluation until resolved.'
(O/'verification/PASSED').write_text(f'{len(checks)} replay checks passed\n')
