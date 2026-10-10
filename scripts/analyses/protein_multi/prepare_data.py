import gzip, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import io, sparse
import anndata as ad
sys.path.insert(0, str(Path.cwd()/'scripts'))
from sle_prepare_citeseq import read_lines, sha256_file
O=Path('artifacts/paper_evidence/review_round4/protein_multi/corrected')
S=Path('external_data/citeseq_hao2021')
t0=time.time()
for line in (S/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split()
    assert sha256_file(S/name)==digest, name
meta=pd.read_csv(S/'GSE164378_sc.meta.data_3P.csv.gz',index_col=0)
barcodes=read_lines(S/'GSM5008737_RNA_3P-barcodes.tsv.gz')
assert barcodes==read_lines(S/'GSM5008738_ADT_3P-barcodes.tsv.gz')==meta.index.tolist()

groups=list(meta.groupby(['donor','time','celltype.l2'],sort=True,dropna=False).indices.items())
n=min(60000,len(meta)); sizes=np.array([len(v) for _,v in groups]); quotas=n*sizes/sizes.sum()
allocation=np.floor(quotas).astype(int)
allocation[np.argsort(-(quotas-allocation),kind='stable')[:n-allocation.sum()]]+=1
rng=np.random.default_rng(1729)
cols=np.sort(np.concatenate([rng.choice(v,size=k,replace=False) for (_,v),k in zip(groups,allocation)]))
obs=meta.iloc[cols].copy()
donors=obs.groupby('donor').size().rename('cells').reset_index().sort_values(['cells','donor'],ascending=[False,True])
test=donors.donor.iloc[:2].tolist(); assert len(donors)==8
obs['split']=np.where(obs.donor.isin(test),'test','development')
obs.to_csv(O/'selection/cells.csv')
donors.to_csv(O/'selection/donors.csv',index=False)
print('Selected cells',len(obs),'test donors',test,flush=True)
rna=sparse.csc_matrix(io.mmread(S/'GSM5008737_RNA_3P-matrix.mtx.gz'))
counts=rna[:,cols].T.tocsr().astype(np.float32); del rna
genes=read_lines(S/'GSM5008737_RNA_3P-features.tsv.gz')

var=pd.DataFrame({'gene_symbol':genes},index=pd.Index(genes))
a=ad.AnnData(X=counts,obs=obs,var=var); a.var_names_make_unique()
a.obs['total_counts']=np.asarray(counts.sum(axis=1)).ravel()
a.write_h5ad(O/'prepared/sampled.h5ad')
adt=sparse.csc_matrix(io.mmread(S/'GSM5008738_ADT_3P-matrix.mtx.gz'))[:,cols].T.toarray().astype(np.float64)
tags=read_lines(S/'GSM5008738_ADT_3P-features.tsv.gz')
log=np.log1p(adt); clr=log-log.mean(axis=1,keepdims=True)
pd.DataFrame(clr,index=obs.index,columns=tags).to_parquet(O/'prepared/protein_clr.parquet')
report={'cells':len(obs),'genes':len(genes),'antibodies':len(tags),'test_donors':test,'train_cells':int((obs.split=='development').sum()),'test_cells':int((obs.split=='test').sum()),'seconds':time.time()-t0,'doublets_retained':int((obs['celltype.l2']=='Doublet').sum())}
(O/'prepared/data_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(report,flush=True)
