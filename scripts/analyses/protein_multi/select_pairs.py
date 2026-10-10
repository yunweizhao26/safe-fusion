import hashlib, json, re, sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path.cwd()/'scripts'))
from sle_build_fold import variable_genes
O=Path('artifacts/paper_evidence/review_round4/protein_multi/corrected'); t0=time.time()
assert not (O/'selection/selected_pairs.csv').exists(), 'Selection already frozen'
a=ad.read_h5ad(O/'prepared/sampled.h5ad'); x=a.X.tocsc()
p=pd.read_parquet(O/'prepared/protein_clr.parquet'); assert p.index.tolist()==a.obs_names.tolist()
ann=pd.read_csv(O/'sources/hao_antibodies.csv',keep_default_na=False)
ann['tag']=ann['#protein'].str.replace('_','-',regex=False); assert ann.tag.is_unique
h=pd.read_csv(O/'sources/hgnc_complete_set.txt',sep='\t',keep_default_na=False)
train=a.obs['split'].to_numpy()=='development'; test=~train
library=np.asarray(x.sum(axis=1)).ravel(); symbols=a.var.gene_symbol.astype(str)
unique=set(symbols[~symbols.duplicated(keep=False)]); rows=[]
for tag in p.columns:
 r=ann.loc[ann.tag==tag]; assert len(r)==1, tag
 r=r.iloc[0]; ids=sorted(set(re.findall(r'ENSG\d+',r['Ensembl Gene Id'])))
 item={'protein':tag,'gene':'','ensembl_gene_id':';'.join(ids),'mapping_status':'','n_train_detected':0,'train_detected_spearman':np.nan,'n_test_zero':0,'n_test_zero_top_half':0,'test_protein_median':np.nan,'selected':False}
 if 'IgG' in tag: item['mapping_status']='isotype'
 elif len(ids)!=1:item['mapping_status']='no_unique_human_gene'
 else:
  ht=h.loc[h.ensembl_gene_id==ids[0]]
  if len(ht)!=1:item['mapping_status']='no_unique_hgnc_gene'
  else:
   ht=ht.iloc[0]; gene=ht.symbol
   if gene not in unique:
    alternatives=(set(ht.prev_symbol.split('|'))|set(ht.alias_symbol.split('|')))&unique
    gene=next(iter(alternatives)) if len(alternatives)==1 else ''
   if not gene:item['mapping_status']='gene_absent_or_ambiguous_in_RNA'
   else:
    item['gene']=gene; item['mapping_status']='mapped'
    j=np.flatnonzero(symbols.to_numpy()==gene); assert len(j)==1
    raw=x[:,j[0]].toarray().ravel(); detected=train&(raw>0); zeros=test&(raw==0)
    protein=p[tag].to_numpy(); median=float(np.median(protein[test]))
    rho=float(spearmanr(np.log1p(1e4*raw[detected]/library[detected]),protein[detected]).statistic) if detected.sum()>=2 else np.nan
    item.update(n_train_detected=int(detected.sum()),train_detected_spearman=rho,n_test_zero=int(zeros.sum()),n_test_zero_top_half=int((zeros&(protein>median)).sum()),test_protein_median=median)
    item['selected']=bool(np.isfinite(rho) and rho>=.31 and item['n_test_zero']>=200 and item['n_test_zero_top_half']>=50)
 rows.append(item)
audit=pd.DataFrame(rows); audit.to_csv(O/'selection/all_pairs.csv',index=False)
selected=audit.loc[audit.selected].sort_values(['gene','protein']).reset_index(drop=True)
with (O/'selection/selected_pairs.csv').open('x') as f:selected.to_csv(f,index=False)
files=['selection/rule.md','selection/selected_pairs.csv','selection/all_pairs.csv','selection/cells.csv','selection/donors.csv','sources/hao_antibodies.xlsx','sources/hao_antibodies.csv','sources/hgnc_complete_set.txt']
with (O/'selection/FROZEN.sha256').open('x') as f:
 for rel in files:
  f.write(hashlib.sha256((O/rel).read_bytes()).hexdigest()+'  '+rel+'\n'); (O/rel).chmod(0o444)
assert len(selected)>0, 'No pair passes the frozen rule; do not fit'
hvg,score=variable_genes(a.X,train,2000); assert len(hvg)==2000
extra=a.var_names.get_indexer(selected.gene.unique()); assert (extra>=0).all()
cols=np.unique(np.concatenate([hvg,extra])); truth=a[:,cols].copy(); truth.layers['counts']=truth.X.copy()
truth.var['variable_gene']=np.isin(cols,hvg); truth.var['selected_pair']=truth.var_names.isin(selected.gene)
truth.write_h5ad(O/'prepared/truth.h5ad')
pd.DataFrame({'cell_id':a.obs_names,'split':a.obs['split'].to_numpy(),'biological_unit':a.obs.donor.to_numpy()}).to_parquet(O/'prepared/splits.parquet',index=False)
report={'selected_pairs':len(selected),'unique_genes':selected.gene.nunique(),'hvg':len(hvg),'panel_genes':len(cols),'mapping_counts':audit.mapping_status.value_counts().to_dict(),'seconds':time.time()-t0}
(O/'selection/report.json').write_text(json.dumps(report,indent=2)+'\n'); print(report,flush=True);print(selected.to_string(index=False),flush=True)
