
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

O = (Path.cwd() / 'artifacts/paper_evidence/review_round4/main_tables_v3')
ROOT = Path.cwd()
f = pd.read_csv(O/'estimates.csv')
c = pd.read_csv(O/'point_checks.csv')
b = np.load(O/'bootstrap_quantities.npz', allow_pickle=False)
assert len(f) == 83 and len(c) == 83 and c.passed.all()
assert f.groupby('table').size().to_dict() == {'Table 1':39, 'Table 2':44}
assert not f.duplicated(['table','method','column']).any()
assert len(b.files) == 83
for r in f.to_dict('records'):
    samples = b['|'.join([r['table'],r['method'],r['column']])]
    assert samples.shape == (2000,) and np.isfinite(samples).all()
    np.testing.assert_allclose([np.std(samples,ddof=1),*np.quantile(samples,[.025,.975])],
                               [r['bootstrap_se'],r['ci_low'],r['ci_high']], atol=1e-12, rtol=0)
assert (f.draws == 2000).all() and (f.seed == 1729).all()
sources = json.loads((O/'source_sha256.json').read_text())
for name, wanted in sources.items():
    digest = hashlib.sha256()
    with Path(name).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    assert digest.hexdigest() == wanted, name
result = dict(passed=True, source_hashes_verified=len(sources), estimates=83, draw_vectors=83, draws=166000, point_checks=83)
(O/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
