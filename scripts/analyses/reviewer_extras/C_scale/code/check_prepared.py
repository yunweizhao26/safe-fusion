import json
from pathlib import Path
import anndata as ad
import pandas as pd

root = (Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras/C_scale')
old = Path('artifacts/paper_evidence/review_round3/scale/cells_200000')
new = root / 'cells_200000'
a = ad.read_h5ad(old / 'truth.h5ad', backed='r')
b = ad.read_h5ad(new / 'truth.h5ad', backed='r')
assert a.shape == (200000, 2000)
assert b.shape == (200000, 5000)
assert a.obs_names.equals(b.obs_names)
assert a.obs['donor'].equals(b.obs['donor'])
assert set(a.var_names).issubset(b.var_names)
pa, pb = pd.read_parquet(old / 'splits.parquet'), pd.read_parquet(new / 'splits.parquet')
pd.testing.assert_frame_equal(pa, pb)
result = {'cells': 200000, 'genes': 5000, 'same_cell_order_as_S28': True,
          'same_donors_as_S28': True, 'same_splits_as_S28': True,
          'original_2000_genes_subset_of_5000': True,
          'test_donors': int(pb.loc[pb['split'] == 'test', 'biological_unit'].nunique())}
a.file.close()
b.file.close()
(root / 'prepared_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
