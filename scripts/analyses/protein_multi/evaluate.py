import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
O=Path('artifacts/paper_evidence/review_round4/protein_multi/corrected')
METHODS=['Safe Fusion','SVD','Weighted kNN','MAGIC','scVI','scVI P(X>0)','Library size']

def metric_setup(scores, protein):

    inverses=[np.unique(v,return_inverse=True)[1] for v in [*scores.T,protein]]
    thresholds=np.quantile(protein,np.linspace(.3,.7,41))
    bucket=np.searchsorted(thresholds,protein,side='left')
    return inverses,bucket,thresholds

def metrics(weights, inverses, bucket):
    n=weights.sum(); center=(n+1)/2
    ranks=[]
    for inverse in inverses:
        count=np.bincount(inverse,weights=weights)
        ranks.append((np.cumsum(count)-count/2+.5)[inverse])
    ry=ranks[-1]-center; vy=np.dot(weights,ry*ry)
    npos=np.cumsum(np.bincount(bucket,weights=weights,minlength=42)[::-1])[::-1][1:]
    nneg=n-npos
    result=[]
    for rank in ranks[:-1]:
        rx=rank-center; vx=np.dot(weights,rx*rx)
        rho=np.dot(weights,rx*ry)/np.sqrt(vx*vy) if vx>0 and vy>0 else np.nan
        sums=np.cumsum(np.bincount(bucket,weights=weights*rank,minlength=42)[::-1])[::-1][1:]
        valid=(npos>0)&(nneg>0)
        auc=float(np.mean((sums[valid]-npos[valid]*(npos[valid]+1)/2)/(npos[valid]*nneg[valid]))) if valid.any() and vx>0 else np.nan
        result.append([rho,auc])
    return np.asarray(result)

def selftest():
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    from evaluate_papalexi_crossmodal import safe_spearman, quantile_metrics
    rng=np.random.default_rng(1729); scores=rng.integers(0,7,(31,7)).astype(float); protein=rng.integers(0,9,31).astype(float)
    inv,bucket,thresholds=metric_setup(scores,protein)
    for repeat in range(10):
        index=np.arange(31) if repeat==0 else np.r_[rng.choice(13,13),rng.choice(np.arange(13,31),18)]
        w=np.bincount(index,minlength=31); got=metrics(w,inv,bucket)
        want=np.asarray([[safe_spearman(s[index],protein[index]),np.nanmean([quantile_metrics((protein[index]>t).astype(int),s[index])[0] for t in thresholds])] for s in scores.T])
        np.testing.assert_allclose(got,want,atol=1e-12,rtol=0)
        assert np.allclose(got-got,0)
    assert np.isnan(metrics(np.ones(3),*metric_setup(np.ones((3,7)),np.arange(3))[:2])).all()
    (O/'evaluation/selftest.json').write_text(json.dumps({'passed':True,'draws_checked':10,'reference':'Papalexi safe_spearman and quantile_metrics','absolute_tolerance':1e-12})+'\n')
    print('Exact bootstrap metrics match Papalexi/scipy/sklearn, including ties.',flush=True)

def extract():
    import anndata as ad
    import hashlib
    for line in (O/'selection/FROZEN.sha256').read_text().splitlines():
        digest,rel=line.split(maxsplit=1); assert hashlib.sha256((O/rel).read_bytes()).hexdigest()==digest,rel
    a=ad.read_h5ad(O/'prepared/truth.h5ad'); split=a.obs['split'].to_numpy(); test=split=='test'
    p=pd.read_parquet(O/'prepared/protein_clr.parquet'); assert p.index.tolist()==a.obs_names.tolist()
    selected=pd.read_csv(O/'selection/selected_pairs.csv'); scores=pd.read_parquet(O/'selector/selected_gene_scores.parquet')
    scores=scores.loc[scores['split']=='test']; assert not scores.duplicated(['cell_id','gene_id']).any()
    sources={'SVD':'teachers/svd_impute','Weighted kNN':'teachers/graph_smooth','MAGIC':'baselines/magic','scVI':'baselines/scvi','scVI P(X>0)':'baselines/scvi_probability'}
    arrays={}
    for method,rel in sources.items():
        metadata=json.loads((O/rel/'metadata.json').read_text())
        assert metadata['cell_ids']==a.obs_names.tolist() and metadata['gene_ids']==a.var_names.tolist()
        arrays[method]=np.load(O/rel/'mean.npy',mmap_mode='r'); assert arrays[method].shape==a.shape
    library=np.asarray(a.X.sum(axis=1)).ravel()
    (O/'evaluation/inputs').mkdir(exist_ok=True)
    for i,r in selected.iterrows():
        g=a.var_names.get_loc(r.gene); raw=a.X[:,g].toarray().ravel(); rows=np.flatnonzero(test&(raw==0))
        assert len(rows)==r.n_test_zero
        sf=scores.loc[scores.gene_id==r.gene].set_index('cell_id').loc[a.obs_names[rows]]
        assert np.array_equal(sf.cell_index.to_numpy(),rows)
        assert (sf.masked_positive.to_numpy()==0).all()
        frame=pd.DataFrame({'cell_id':a.obs_names[rows],'donor':a.obs.donor.to_numpy()[rows],'protein':p[r.protein].to_numpy()[rows],'Safe Fusion':sf.selector_score.to_numpy()})
        for method,arr in arrays.items():frame[method]=arr[rows,g]
        frame['Library size']=np.log1p(library[rows])
        assert np.isfinite(frame[METHODS+['protein']].to_numpy()).all()
        frame.to_parquet(O/f'evaluation/inputs/pair_{i:03}.parquet',index=False)
    print('Extracted',len(selected),'pairs',flush=True)

def evaluate(i):
    t0=time.time(); pair=pd.read_csv(O/'selection/selected_pairs.csv').iloc[i]
    frame=pd.read_parquet(O/f'evaluation/inputs/pair_{i:03}.parquet'); n=len(frame)
    inv,bucket,thresholds=metric_setup(frame[METHODS].to_numpy(),frame.protein.to_numpy())
    point=metrics(np.ones(n),inv,bucket)
    groups=[np.flatnonzero(frame.donor.to_numpy()==d) for d in sorted(frame.donor.unique())]
    assert len(groups)==2
    rng=np.random.default_rng(1729); boot=np.empty((2000,len(METHODS),2))
    for draw in range(2000):
        index=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        boot[draw]=metrics(np.bincount(index,minlength=n),inv,bucket)
    records=[]
    for k,method in enumerate(METHODS):
        rec={'pair_index':i,'gene':pair.gene,'protein':pair.protein,'method':method,'n_test_zero':n}
        for j,metric in enumerate(['spearman','mean_auroc']):
            v=boot[:,k,j]; d=boot[:,0,j]-v
            lo,hi=np.nanquantile(v,[.025,.975]); dlo,dhi=np.nanquantile(d,[.025,.975])
            rec.update({metric:point[k,j],metric+'_ci_low':lo,metric+'_ci_high':hi,metric+'_valid_draws':int(np.isfinite(v).sum()),'delta_'+metric:point[0,j]-point[k,j],'delta_'+metric+'_ci_low':dlo,'delta_'+metric+'_ci_high':dhi,'delta_'+metric+'_valid_draws':int(np.isfinite(d).sum())})
        records.append(rec)
    pd.DataFrame(records).to_csv(O/f'evaluation/pair_{i:03}.csv',index=False)
    np.savez(O/f'evaluation/pair_{i:03}_bootstrap.npz',bootstrap=boot,point=point,thresholds=thresholds,methods=np.asarray(METHODS))
    print(json.dumps({'pair':i,'gene':pair.gene,'protein':pair.protein,'n':n,'seconds':time.time()-t0}),flush=True)

def aggregate():
    pairs=pd.read_csv(O/'selection/selected_pairs.csv')
    frames=[pd.read_csv(O/f'evaluation/pair_{i:03}.csv') for i in range(len(pairs))]
    df=pd.concat(frames,ignore_index=True); assert len(df)==len(pairs)*len(METHODS)
    all_finite=bool((df.filter(regex='valid_draws$')==2000).all().all())
    df.to_csv(O/'evaluation/per_pair.csv',index=False)
    summary=[]
    for method in METHODS[1:]:
        f=df[df.method==method]
        for metric in ['spearman','mean_auroc']:
            d=f['delta_'+metric].to_numpy(); nz=d[np.isfinite(d)&(d!=0)]; wins=int((nz>0).sum())
            summary.append({'method':method,'metric':metric,'pairs':len(f),'positive_differences':int((d>0).sum()),'positive_CI_excludes_zero':int(((d>0)&(f['delta_'+metric+'_ci_low']>0)).sum()),'negative_CI_excludes_zero':int(((d<0)&(f['delta_'+metric+'_ci_high']<0)).sum()),'median_delta':float(np.nanmedian(d)),'sign_test_nonzero_pairs':len(nz),'sign_test_two_sided_p':float(binomtest(wins,len(nz),.5).pvalue) if len(nz) else 1.})
    pd.DataFrame(summary).to_csv(O/'evaluation/summary.csv',index=False)
    wide=df.pivot(index='pair_index',columns='method',values='spearman')[METHODS[:-1]]
    complete=wide.notna().all(axis=1)
    highest=(wide['Safe Fusion']==wide.max(axis=1))&complete; strict=(wide['Safe Fusion']>wide[METHODS[1:-1]].max(axis=1))&complete
    report={'selected_pairs':len(pairs),'unique_genes':int(pairs.gene.nunique()),'safe_fusion_highest_spearman_including_ties':int(highest.sum()),'safe_fusion_strict_highest_spearman':int(strict.sum()),'test_donors':json.loads((O/'prepared/data_report.json').read_text())['test_donors'],'all_bootstrap_draws_finite':all_finite,'pairs_with_all_imputation_spearmans_defined':int(complete.sum())}
    (O/'evaluation/summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(report)

if __name__=='__main__':
    command=sys.argv[1]
    if command=='selftest':selftest()
    elif command=='extract':extract()
    elif command=='pair':evaluate(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif command=='aggregate':aggregate()
    else:raise ValueError(command)
