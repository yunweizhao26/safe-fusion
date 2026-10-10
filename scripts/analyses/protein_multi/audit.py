import hashlib,json,sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
O=Path('artifacts/paper_evidence/review_round4/protein_multi/corrected')
for line in (O/'selection/FROZEN.sha256').read_text().splitlines():
 digest,rel=line.split(maxsplit=1);assert hashlib.sha256((O/rel).read_bytes()).hexdigest()==digest
sample=ad.read_h5ad(O/'prepared/sampled.h5ad');train=sample.obs['split'].to_numpy()=='development';test=~train
p=pd.read_parquet(O/'prepared/protein_clr.parquet');audit=pd.read_csv(O/'selection/all_pairs.csv'); pairs=pd.read_csv(O/'selection/selected_pairs.csv')
x=sample.X.tocsc();lib=np.asarray(x.sum(axis=1)).ravel();diff=[];maxerr=0;low=[]
for _,r in audit[audit.mapping_status=='mapped'].iterrows():
 j=sample.var_names.get_loc(r.gene);raw=x[:,j].toarray().ravel();d=train&(raw>0);z=test&(raw==0)
 rho=float(spearmanr(np.log1p(1e4*raw[d]/lib[d]),p[r.protein].to_numpy()[d]).statistic) if d.sum()>=2 else np.nan
 top=int((z&(p[r.protein].to_numpy()>np.median(p[r.protein].to_numpy()[test]))).sum())
 expected=bool(np.isfinite(rho) and rho>=.31 and z.sum()>=200 and top>=50)
 if expected!=r.selected:diff.append({'gene':r.gene,'protein':r.protein,'expected':expected,'selected':r.selected,'rho':rho})
 if d.sum()<3:low.append({'gene':r.gene,'n_detected':int(d.sum()),'rho':rho,'expected':expected})
 if np.isfinite(rho) and np.isfinite(r.train_detected_spearman):maxerr=max(maxerr,abs(rho-r.train_detected_spearman))
truth=ad.read_h5ad(O/'prepared/truth.h5ad');hybrid=ad.read_h5ad(O/'prepared/deploy/hybrid.h5ad');coords=pd.read_parquet(O/'prepared/deploy/coordinates.parquet')
assert truth.obs_names.tolist()==hybrid.obs_names.tolist()==sample.obs_names.tolist()
assert truth.var_names.tolist()==hybrid.var_names.tolist()
assert (truth.X[test]!=hybrid.layers['corrupted_counts'][test]).nnz==0
assert set(coords['split'])=={'development'}
assert set(pairs.gene)<=set(truth.var_names)
assert int(truth.var.variable_gene.sum())==2000
assert set(truth.obs.loc[test,'donor'])=={'P8','P7'}
rows=coords.cell_index.to_numpy();cols=coords.gene_index.to_numpy()
assert np.array_equal(np.asarray(truth.X[rows,cols]).ravel(),coords.original_value)
assert np.asarray(hybrid.layers['corrupted_counts'][rows,cols]).ravel().sum()==0
report={'selection_recomputed':True,'selection_mismatches':diff,'max_correlation_error':maxerr,'low_detection_pairs':low,'test_counts_unchanged':True,'mask_training_only':True,'training_masked_positives':len(coords),'mask_fraction':len(coords)/truth.X[train].nnz,'hvg_count':2000,'frozen_hashes_match':True}
(O/'evaluation/pre_fit_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
assert not diff, 'Frozen selection differs from literal requested rule; must report, never silently change selection'
