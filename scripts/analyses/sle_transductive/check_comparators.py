import json
from pathlib import Path
import numpy as np
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/sle_transductive')
OLD=Path('artifacts/paper_evidence/sle_treg_case')
def equal(x,y):
    if isinstance(x,str):
        if x==y: return True
        if x.startswith('[') and y.startswith('['):
            return np.allclose(json.loads(x.replace('nan','NaN')),json.loads(y.replace('nan','NaN')),rtol=1e-10,atol=1e-10,equal_nan=True)
        return False
    return bool(np.isclose(x,y,rtol=1e-10,atol=1e-10,equal_nan=True))
checks=[]
for p in sorted((OLD/'results').glob('*/*.csv')):
    a=pd.read_csv(p,keep_default_na=False,na_values=[''])
    if 'comparator' in a: continue
    keep=a['method']!='Safe Fusion' if 'method' in a else np.ones(len(a),bool)
    for variant in ('conditional','conditional_detection'):
        b=pd.read_csv(O/variant/p.relative_to(OLD),keep_default_na=False,na_values=[''])
        for c in a:
            if c.startswith('safe_fusion_minus'): continue
            ok=all(equal(x,y) for x,y in zip(a.loc[keep,c],b.loc[keep,c]))
            checks.append({'variant':variant,'table':str(p.relative_to(OLD)),'column':c,'passed':ok})
(O/'comparator_endpoint_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
failed=[c for c in checks if not c['passed']]
print(json.dumps({'checks':len(checks),'failed':failed}))
assert not failed
