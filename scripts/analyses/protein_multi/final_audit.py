import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/protein_multi/corrected')
for manifest in [O/'selection/FROZEN.sha256',O/'runtime/source_sha256.txt',O/'runtime/dependency_sha256.txt']:
 base=O if manifest==O/'selection/FROZEN.sha256' else O.parent if manifest==O.parent/'selection/FROZEN.sha256' else Path.cwd()
 for line in manifest.read_text().splitlines():
  digest,rel=line.split(maxsplit=1);assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==digest,rel
pairs=pd.read_csv(O/'selection/selected_pairs.csv'); df=pd.read_csv(O/'evaluation/per_pair.csv')
assert len(pairs)==41 and pairs.gene.nunique()==38 and len(df)==287
assert not df.duplicated(['pair_index','method']).any()
for metric in ['spearman','mean_auroc']:
 wide=df.pivot(index='pair_index',columns='method',values=metric)
 for method in wide:
  rows=df[df.method==method].sort_values('pair_index')
  np.testing.assert_allclose(rows['delta_'+metric],wide['Safe Fusion']-wide[method],atol=1e-12,rtol=0,equal_nan=True)
  for prefix in ['', 'delta_']:
   finite=np.isfinite(rows[prefix+metric+'_ci_low'])&np.isfinite(rows[prefix+metric+'_ci_high'])
   assert (rows.loc[finite,prefix+metric+'_ci_low']<=rows.loc[finite,prefix+metric+'_ci_high']).all()
input_hash=hashlib.sha256((O/'prepared/deploy/hybrid.h5ad').read_bytes()).hexdigest()
coordinate_hash=hashlib.sha256((O/'prepared/deploy/coordinates.parquet').read_bytes()).hexdigest()
for rel in ['teachers/gene_median','teachers/svd_impute','teachers/graph_smooth','teachers/magic','teachers/scvi']:
 m=json.loads((O/rel/'metadata.json').read_text());assert m['scale']=='counts' and m['parameters']['transductive']
 assert m['parameters']['fit_cells']==60000
 assert m['input_sha256']==input_hash and m['coordinates_sha256']==coordinate_hash
value=json.loads((O/'safe_fusion/metadata.json').read_text()); assert value['parameters']['fit_cells']==40672
assert value['parameters']['value_model']=='boosted'
assert value['input_sha256']==input_hash and value['coordinates_sha256']==coordinate_hash
selector=json.loads((O/'selector/selected_gene_scores_metadata.json').read_text())
assert selector['architecture']=='mlp' and selector['fit_split']=='development' and not selector['test_labels_used_for_training']
assert not selector['missing_genes']; assert set(selector['requested_genes'])==set(pairs.gene)
overflow=json.loads((O/'selector/overflow_sampling.json').read_text())
assert len(overflow['calls'])==1
assert overflow['calls'][0]['selected_positive']==overflow['calls'][0]['selected_negative']==1000000
summary=json.loads((O/'evaluation/summary.json').read_text())
report={'passed':True,'frozen_corrected_selection_unchanged':True,'production_and_dependency_hashes_unchanged':True,'paired_differences_recomputed':True,'pair_method_rows':len(df),'teacher_fit_cells':60000,'value_training_cells':40672,'selector_test_labels_used':False,'all_bootstrap_draws_finite':summary['all_bootstrap_draws_finite'],'incomplete_bootstrap_rows':int((df.filter(regex='valid_draws$')<2000).any(axis=1).sum())}
(O/'evaluation/final_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
