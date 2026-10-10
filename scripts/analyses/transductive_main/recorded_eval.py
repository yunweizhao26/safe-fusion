from common import *
import json
import anndata as ad
import numpy as np
import pandas as pd
import evaluate_inserted_value as v
from masked_f1_units import fraction_name
from v2_value_select import bootstrap_weights

v.DEPLOYMENT={'CRISPRa (Norman)':v.DEPLOYMENT['CRISPRa (Norman)']}
old=pd.read_csv(E/'review_round2/inserted_value/recorded_zero_fills.csv')
fresh=v.recorded_zero_fills()
joined=old.merge(fresh,on=['dataset','method','fill_fraction'],suffixes=('_old','_new'))
columns=[c for c in fresh if c not in ['dataset','method','fill_fraction']]
errors=[float(np.nanmax(np.abs(joined[c+'_old']-joined[c+'_new']))) for c in columns]
assert len(joined)==len(fresh) and max(errors)<1e-9,(len(joined),errors)
(OUT/'value_accuracy/recorded_parity.json').write_text(json.dumps(dict(rows=len(joined),max_difference=max(errors)),indent=2)+'\n')

def median_samples(values,codes,weights):
    order=np.argsort(values,kind='stable')
    values,codes=values[order],codes[order]
    result=[]

    for w in weights:
        counts=w[codes].astype(np.int64)
        cumulative=np.cumsum(counts)
        n=cumulative[-1]
        if n==0:
            result.append(np.nan)
        else:
            a=np.searchsorted(cumulative,(n-1)//2,side='right')
            b=np.searchsorted(cumulative,n//2,side='right')
            result.append((values[a]+values[b])/2)
    return np.array(result)

rows=[]
for dataset in ['Pancreas','Colon','CRISPRa']:
    for model in ['inductive','transductive']:
        pools={b:[] for b in [.01,.05,.10]}
        all_expected=[]
        for u in [u for u in units() if u.dataset==dataset]:
            root=OUT/'recorded'/u.key
            data=ad.read_h5ad(root/'input/recorded.h5ad')
            counts=v.dense(data.layers['corrupted_counts'])
            split=pd.read_parquet(u.splits).set_index('cell_id').loc[data.obs_names.astype(str),'split'].to_numpy()
            rr,cc=np.where((counts==0)&(split=='test')[:,None])
            names=['gene_median','svd_impute','graph_smooth',*(['magic','scvi'] if model=='transductive' else ['magic_inductive','scvi_inductive'])]
            expected=np.mean([np.maximum(v.at(root/model/n/'mean.npy',rr,cc),0) for n in names],axis=0)
            groups=np.char.add(u.key+':',data.obs[u.unit_column].astype(str).to_numpy()[rr].astype(str))
            all_expected.append((expected,groups))
            for b in pools:
                values=v.at(root/model/'selector'/f'safe_fusion_calibrated_mlp_topk_{fraction_name(b)}'/'mean.npy',rr,cc)
                selected=values!=0
                assert abs(int(selected.sum())-max(1,int(round(b*len(rr)))))<=1
                pools[b].append((values[selected],expected[selected],groups[selected]))
        group_order=np.unique(np.concatenate([g for _,g in all_expected]))
        weights=bootstrap_weights(len(group_order),2000,1729)
        expected_all=np.concatenate([x for x,_ in all_expected])
        all_codes=np.searchsorted(group_order,np.concatenate([g for _,g in all_expected]))
        nall=np.bincount(all_codes,minlength=len(group_order))
        tall=np.bincount(all_codes,weights=expected_all>2,minlength=len(group_order))
        all_sample=100*(weights@tall)/(weights@nall)
        for b,parts in pools.items():
            values=np.concatenate([p[0] for p in parts])
            expected=np.concatenate([p[1] for p in parts])
            groups=np.concatenate([p[2] for p in parts])
            codes=np.searchsorted(group_order,groups)
            med=median_samples(values,codes,weights)
            low,high=np.nanquantile(med[1:],[.025,.975])
            row=dict(dataset=dataset,model=model,fill_fraction=b,**v.distribution(values,expected),median_low=low,median_high=high)
            n=np.bincount(codes,minlength=len(group_order))
            for label,condition in [('expected_above_2',expected>2),('expected_above_5',expected>5),('value_above_1',values>1)]:
                hits=np.bincount(codes,weights=condition,minlength=len(group_order))
                sample=100*(weights@hits)/(weights@n)
                row[label+'_percent']=sample[0]
                row[label+'_low'],row[label+'_high']=np.nanquantile(sample[1:],[.025,.975])
            row['all_expected_above_2_percent']=all_sample[0]
            row['all_expected_above_2_low'],row['all_expected_above_2_high']=np.quantile(all_sample[1:],[.025,.975])
            rows.append(row)
pd.DataFrame(rows).to_csv(OUT/'value_accuracy/recorded.csv',index=False)
