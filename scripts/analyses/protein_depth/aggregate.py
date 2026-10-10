from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
O=Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_depth'
REFS=['safe_fusion','standard_depth','residual_logit']

def interval(v):
    v=np.asarray(v); v=v[np.isfinite(v)]
    return np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]

def aggregate():
    cases=[f'hao_{i:03}' for i in range(41)]+['papalexi_small','papalexi_full']
    for kind in ['association','differences','diagnostics','groups']:
        frames=[pd.read_csv(O/f'tables/{case}_{kind}.csv') for case in cases]
        pd.concat(frames,ignore_index=True).to_csv(O/f'tables/{kind}.csv',index=False)
    a=pd.read_csv(O/'tables/association.csv'); d=pd.read_csv(O/'tables/differences.csv')
    diag=pd.read_csv(O/'tables/diagnostics.csv')
    ha=a[a.case.str.startswith('hao_')]
    meta=ha[['case','pair_index','gene','protein','n_train_detected','n_test_zero']].drop_duplicates().set_index('case').sort_index()
    assert len(meta)==41 and len(a.case.unique())==43
    depth=diag[diag.metric=='protein_depth_spearman'].set_index('case').estimate
    meta['protein_depth_spearman']=depth
    meta['training_stratum']=np.where(meta.n_train_detected<200,'<200','>=200')
    meta['depth_stratum']=np.select([meta.protein_depth_spearman<-.1,meta.protein_depth_spearman>.1],['negative','positive'],default='neutral')
    meta.to_csv(O/'tables/pairs.csv')
    strata={'all':list(meta.index)}
    for n,b in meta.groupby('training_stratum'):strata['training '+n]=list(b.index)
    for n,b in meta.groupby('depth_stratum'):strata['depth '+n]=list(b.index)
    for (t,s),b in meta.groupby(['training_stratum','depth_stratum']):strata['training '+t+'; depth '+s]=list(b.index)
    summaries=[]; medians=[]; highest=[]; diagnostic_summaries=[]
    for label,ids in strata.items():
        draws=np.random.default_rng(1729).integers(len(ids),size=(2000,len(ids)))
        for (method,metric),b in ha[ha.case.isin(ids)].groupby(['method','metric'],sort=False):
            values=b.set_index('case').reindex(ids).estimate.to_numpy()
            lo,hi=interval(np.nanmedian(values[draws],axis=1))
            medians.append(dict(stratum=label,method=method,metric=metric,pairs=len(ids),defined_pairs=int(np.isfinite(values).sum()),median=np.nanmedian(values),ci_low=lo,ci_high=hi))
        for (ref,comp,metric),b in d[d.case.isin(ids)].groupby(['ranking','comparator','metric'],sort=False):
            b=b.set_index('case').reindex(ids); values=b.estimate.to_numpy(); nz=values[np.isfinite(values)&(values!=0)]
            lo,hi=interval(np.nanmedian(values[draws],axis=1))
            summaries.append(dict(stratum=label,ranking=ref,comparator=comp,metric=metric,pairs=len(ids),defined_pairs=int(np.isfinite(values).sum()),wins=int((values>0).sum()),ties=int((values==0).sum()),positive_ci=int((b.ci_low>0).sum()),negative_ci=int((b.ci_high<0).sum()),median_delta=np.nanmedian(values),ci_low=lo,ci_high=hi,sign_test_n=len(nz),sign_test_p=binomtest(int((nz>0).sum()),len(nz),.5).pvalue if len(nz) else np.nan))
        for metric,b in diag[diag.case.isin(ids)].groupby('metric',sort=False):
            v=b.set_index('case').reindex(ids).estimate.to_numpy(); lo,hi=interval(np.nanmedian(v[draws],axis=1))
            diagnostic_summaries.append(dict(stratum=label,metric=metric,pairs=len(ids),median=np.nanmedian(v),ci_low=lo,ci_high=hi))
        for metric in ha.metric.unique():
            wide=ha[(ha.case.isin(ids))&(ha.metric==metric)].pivot(index='case',columns='method',values='estimate')
            for ref in REFS:
                for family in ['existing','including_corrections']:
                    cols=[c for c in wide.columns if c!=ref and c!='library_size' and (family=='including_corrections' or c not in ['standard_depth','residual_logit'])]
                    other=wide[cols]; complete=other.notna().all(axis=1)&wide[ref].notna()
                    win=wide[ref]>other.max(axis=1)
                    highest.append(dict(stratum=label,ranking=ref,metric=metric,comparison_set=family,pairs=len(ids),strict_highest=int(win.sum()),complete_pairs=int(complete.sum()),strict_highest_complete=int((win&complete).sum())))
    for name,rows in [('summary',summaries),('ranking_summary',medians),('highest',highest),('diagnostic_summary',diagnostic_summaries)]:
        pd.DataFrame(rows).to_csv(O/f'tables/{name}.csv',index=False)
    assert set(meta.training_stratum.value_counts())=={10,31}
    assert meta.depth_stratum.value_counts().to_dict()=={'negative':23,'neutral':11,'positive':7}
    (O/'runtime/aggregate_audit.json').write_text(json.dumps(dict(passed=True,cases=43,hao_pairs=41,bootstrap_draws=2000,strata={k:len(v) for k,v in strata.items()}),indent=2)+'\n')
if __name__=='__main__':aggregate()
