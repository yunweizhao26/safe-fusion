import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import average_precision_score
O=Path('artifacts/paper_evidence/review_round4/sle_transductive')
OLD=Path('artifacts/paper_evidence/sle_treg_case')
def ap_draws(labels, scores, donor, weights):
    order=np.argsort(-scores,kind='stable')
    y=labels[order]; s=scores[order]; d=donor[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    positive=np.cumsum(y,dtype=np.int64)[ends]
    ends=ends[np.diff(np.r_[0,positive])>0]
    n=np.empty((weights.shape[1],len(ends)),dtype=np.float64)
    tp=np.empty_like(n)
    for k in range(weights.shape[1]):
        member=d==k
        n[k]=np.cumsum(member,dtype=np.int64)[ends]
        tp[k]=np.cumsum(member & (y==1),dtype=np.int64)[ends]
    result=[]
    for start in range(0,len(weights),32):
        w=weights[start:start+32]
        selected=w@n; positive=w@tp
        delta=np.diff(positive,axis=1,prepend=0.)
        precision=np.divide(positive,selected,out=np.zeros_like(positive),where=selected>0)
        numerator=np.sum(delta*precision,axis=1)
        result.extend(np.divide(numerator,positive[:,-1],out=np.full(len(w),np.nan),where=positive[:,-1]>0))
    return np.asarray(result)

y=np.array([0,1,0,1,1,0,1]); s=np.array([.2,.2,.1,.8,.2,.4,.8],dtype=np.float32); d=np.array([0,1,0,1,2,2,0])
w=np.array([[1,1,1],[0,2,1],[3,0,1]],float)
assert np.allclose(ap_draws(y,s,d,w),[average_precision_score(y,s,sample_weight=x[d]) for x in w],rtol=1e-14)
counts=pd.read_parquet(OLD/'results/masked/masked_f1_unit_counts.parquet')
counts=counts[counts.method=='Safe Fusion']
keys=sorted((counts['group'].astype(str)+':'+counts['unit'].astype(str)).unique())
rng=np.random.default_rng(1729)
picks=rng.integers(0,len(keys),(2000,len(keys)))
weights=np.zeros((2001,len(keys)))
weights[0]=1
for i,draw in enumerate(picks,1): np.add.at(weights[i],draw,1)
records=[]
for variant,case in [('Inductive',OLD),('Conditional',O/'conditional')]:
    for fold in range(3):
        unit=case/f'main/fold_{fold}/masked'
        frame=pd.read_parquet(unit/'selector/selected_gene_scores.parquet',columns=['split','cell_index','selector_score','masked_positive'],filters=[('split','=','test')])
        data=ad.read_h5ad(unit.parent/'truth.h5ad',backed='r')
        donor_names=data.obs['donor'].astype(str).to_numpy()[frame.cell_index.to_numpy()]
        data.file.close()
        names=sorted(set(donor_names))
        codes=pd.Index(names).get_indexer(donor_names)
        columns=[keys.index(f'sle_main_fold_{fold}:{d}') for d in names]
        labels=frame.masked_positive.to_numpy(); scores=frame.selector_score.to_numpy()
        values=ap_draws(labels,scores,codes,weights[:,columns])
        expected=json.loads((unit/'selector/calibration_report.json').read_text())['test']['pr_auc']
        assert abs(values[0]-expected)<1e-12,(variant,fold,values[0],expected)
        lo,hi=np.nanquantile(values[1:],[.025,.975])
        records.append({'variant':variant,'fold':fold,'average_precision':values[0],'lower':lo,'upper':hi,'finite_draws':int(np.isfinite(values[1:]).sum()),'original_ap_check_abs_error':abs(values[0]-expected)})
        print(json.dumps(records[-1]),flush=True)
for r in records.copy():
    if r['variant']=='Conditional': records.append({**r,'variant':'Detection-weighted'})
(O/'comparisons').mkdir(exist_ok=True)
pd.DataFrame(records).to_csv(O/'comparisons/masked_ap_intervals.csv',index=False)
