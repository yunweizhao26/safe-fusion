from common import *
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from masked_f1_units import load_unit
from v2_value_select import bootstrap_weights

def ap_draws(score,labels,codes,weights):
    order=np.argsort(-score,kind='stable')
    score,labels,codes=score[order],labels[order],codes[order]
    end=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    start=np.r_[0,end[:-1]+1]
    positive=np.flatnonzero(np.add.reduceat(labels.astype(int),start)>0)
    ng=weights.shape[1]
    true=np.zeros((len(end),ng))
    total=np.empty((len(end),ng))
    for g in range(ng):
        true[:,g]=np.add.reduceat(((codes==g)&labels).astype(float),start)
        total[:,g]=np.cumsum(codes==g)[end]
    cumulative=np.cumsum(true,axis=0)[positive]
    true,total=true[positive],total[positive]
    denominator=np.bincount(codes[labels],minlength=ng)
    out=[]
    for offset in range(0,len(weights),100):
        w=weights[offset:offset+100].T
        tp=cumulative@w
        selected=total@w
        precision=np.divide(tp,selected,out=np.zeros_like(tp),where=selected>0)
        num=((true@w)*precision).sum(axis=0)
        den=denominator@w
        out.extend(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
    return np.asarray(out)

rows=[]
for dataset in ['Pancreas','Colon','CRISPRa']:
    loaded=[]
    for u in [u for u in units() if u.dataset==dataset]:
        data=load_unit(u)
        rr,cc=np.where((data.counts==0)&(data.split=='test')[:,None])
        labels=data.masked[rr,cc].astype(bool)
        groups=np.char.add(u.key+':',data.unit_labels[rr].astype(str))
        loaded.append((u,labels,groups))
    group_order=np.unique(np.concatenate([g for _,_,g in loaded]))
    weights=bootstrap_weights(len(group_order),2000,1729)
    for model in ['inductive','transductive']:
        for feature in ['all','teacher','context']:
            samples=[]
            for u,labels,groups in loaded:
                slug={'all':'full','teacher':'teacher_only','context':'context_only'}[feature]
                root=OUT/'s6_exact'/u.key/model/(slug+'__mlp')
                score=np.load(root/'test_scores.npy')
                sample=ap_draws(score,labels,np.searchsorted(group_order,groups),weights)
                report=json.loads((root/'report.json').read_text())
                assert abs(sample[0]-report['test']['pr_auc'])<1e-10,(root,sample[0],report['test']['pr_auc'])
                samples.append(sample)
            sample=100*np.nanmean(samples,axis=0)
            lo,hi=np.quantile(sample[1:],[.025,.975])
            rows.append(dict(dataset=dataset,model=model,features=feature,estimate=sample[0],lower=lo,upper=hi,folds=len(loaded)))
dest=OUT/'item3_evaluation'
dest.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(dest/'feature_pr_auc_exact.csv',index=False)
