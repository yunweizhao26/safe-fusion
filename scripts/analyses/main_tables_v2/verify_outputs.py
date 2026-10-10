
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path.cwd()
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/main_tables_v2')
f=pd.read_csv(O/'estimates.csv')
a=np.load(O/'bootstrap_quantities.npz')
assert len(f)==91 and len(a.files)==91
for row in f.itertuples(index=False):
    draws=a[f'{row.table}|{row.method}|{row.column}']
    assert draws.shape==(2000,) and np.isfinite(draws).all()
    np.testing.assert_allclose(np.std(draws,ddof=1),row.bootstrap_se,rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.quantile(draws,[.025,.975]),[row.ci_low,row.ci_high],rtol=0,atol=1e-14)
    assert row.excludes_zero==(row.ci_low>0 or row.ci_high<0)
source_hashes=json.loads((O/'source_sha256.json').read_text())
for name,expected in source_hashes.items():
    path=Path(name)
    if not path.is_absolute():path=ROOT/path
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
    assert sha.hexdigest()==expected,name
report={'quantity_draw_checks':91,'source_hashes_unchanged':len(source_hashes),'all_checks_passed':True}
(O/'final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
