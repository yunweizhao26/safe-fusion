from pipeline import *
import anndata as ad, numpy as np, pandas as pd
import evaluate_pancreas_crossfit_biology as pan
import r3_downstream_thinning as thin
index=int(sys.argv[1]); d=D[index]; assert d['key'].startswith('pancreas_')
p=Path(d['dir']); a=ad.read_h5ad(p/'truth.h5ad'); b=ad.read_h5ad(p/('hybrid.h5ad' if d['design']=='masked' else 'recorded.h5ad'))
split=pd.read_parquet(p/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy(); test=split=='test'
a=a[test].copy(); b=b[test].copy(); truth=a.layers['counts'].toarray().astype(np.float32)
raw=b.layers['corrupted_counts'].toarray().astype(np.float32)
parts=['full','masked_only','zeros_only'] if d['design']=='masked' else ['full']
for part in parts:
 out=R/d['design']/'evaluation'/d['key']/part; out.mkdir(parents=True,exist_ok=True)
 matrices={'corrupted_raw':raw,'graph_smooth':np.load(p/'inductive/graph_smooth/mean.npy')[test]}
 for name in NAMES:
  for pct in PCTS:
   n=f'{name}_{pct}pct'; source=p/n if part=='full' else p/'decomposition'/f'{n}__{part}'
   matrices[n]=np.load(source/'mean.npy')[test]
 summary,paired,markers=pan.summarize_disease_contrasts(matrices,truth,a.obs.donor.astype(str).to_numpy(),a.obs.condition.astype(str).to_numpy(),a.obs.cell_type.astype(str).to_numpy(),a.var.feature_name.astype(str).to_numpy(),2000,1730)
 summary.to_csv(out/'disease_fold_summary.csv',index=False); paired.to_csv(out/'disease_fold_comparisons.csv',index=False)
 if d['design']=='thinning':
  temp=R/'thinning/disease_slopes'/d['key']; target=temp/'evaluation/pancreas_biology'; target.mkdir(parents=True,exist_ok=True)
  a.write_h5ad(temp/'truth.h5ad'); b.write_h5ad(temp/'thinned.h5ad')
  for n,v in matrices.items(): np.save(target/f'oof_{n}.npy',v)
  thin.METHODS={n:n for n in NAMES}
  thin.disease_slopes(temp,temp/'truth.h5ad',temp/'thinned.h5ad',2000,1729).to_csv(temp/'slopes.csv',index=False)
