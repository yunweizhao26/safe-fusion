import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path[:0] = ['scripts', 'src']
BASE=Path('artifacts/paper_evidence/review_round4/transductive_downstream')
OUT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras/A_calibration')

def statistics(p,y):
    bins=np.array_split(np.arange(len(p)),10)
    table=[dict(bin=i+1,count=len(b),mean_p=float(p[b].mean()),observed_share=float(y[b].mean())) for i,b in enumerate(bins)]
    ece=sum(t['count']*abs(t['mean_p']-t['observed_share']) for t in table)/len(p)
    return ece,table

def bootstrap(p,y,u,draws=2000):

    m=int(u.max())+1
    indices=[np.flatnonzero(u==i) for i in range(m)]
    cp=[np.r_[0.,np.cumsum(p[ix],dtype=np.float64)] for ix in indices]
    cy=[np.r_[0.,np.cumsum(y[ix],dtype=np.float64)] for ix in indices]
    lengths=np.array([len(ix) for ix in indices])
    sampled=np.random.default_rng(1729).integers(m,size=(draws,m))
    w=np.array([np.bincount(x,minlength=m) for x in sampled])
    total=w@lengths

    j=np.arange(1,10)[None,:]
    cuts=(total[:,None]//10)*j+np.minimum(j,total[:,None]%10)
    lo=np.zeros_like(cuts); hi=np.full_like(cuts,len(p))
    while np.any(lo<hi):
        mid=(lo+hi)//2
        cnt=sum(w[:,i,None]*np.searchsorted(ix,mid,side='left') for i,ix in enumerate(indices))
        lower=cnt<cuts
        lo=np.where(lower,mid+1,lo); hi=np.where(lower,hi,mid)

    before=np.maximum(lo-1,0)
    counts=np.zeros_like(cuts); ps=np.zeros_like(cuts,dtype=float); ys=ps.copy()
    for i,ix in enumerate(indices):
        k=np.searchsorted(ix,before,side='left')
        counts+=w[:,i,None]*k
        ps+=w[:,i,None]*cp[i][k]; ys+=w[:,i,None]*cy[i][k]
    remainder=cuts-counts
    ps+=remainder*p[before]; ys+=remainder*y[before]
    ptotal=w@np.array([a[-1] for a in cp]); ytotal=w@np.array([a[-1] for a in cy])
    bp=np.diff(np.c_[np.zeros(draws),ps,ptotal],axis=1)
    by=np.diff(np.c_[np.zeros(draws),ys,ytotal],axis=1)
    return np.abs(bp-by).sum(axis=1)/total,(ptotal-ytotal)/total,sampled

def check():
    p=np.linspace(0,1,31); y=(np.arange(31)%3==0).astype(int); u=np.arange(31)%3
    e,b,s=bootstrap(p,y,u,20)
    for i,draw in enumerate(s):
        w=np.bincount(draw,minlength=3); ix=np.repeat(np.arange(31),w[u])
        assert np.isclose(e[i],statistics(p[ix],y[ix])[0])
        assert np.isclose(b[i],(p[ix]-y[ix]).mean())
    print('expanded-bootstrap self-check passed',flush=True)

check()
manifest=json.loads((BASE/'manifest.json').read_text())
results=[]; tables=[]; provenance=[]
fig,axes=plt.subplots(1,3,figsize=(8.4,2.8))
for ax,dataset in zip(axes,['pancreas','colon','norman_crispra']):
    pieces=[]
    for d in manifest:
        if d['design']!='masked' or not (d['key']==dataset or d['key'].startswith(dataset+'_')): continue
        folder=Path(d['dir'])/'transductive/selector'
        a=np.load(folder/'detection.npz'); rows=a['rows']; cols=a['cols']; detection=a['detection'].astype(float)
        assert np.all(np.isfinite(detection)) and np.all((detection>=0)&(detection<=1))
        p=np.clip(.1*detection/(1-.9*detection),0,1)
        report=json.loads((folder/'calibration_report.json').read_text())['detection_rule']
        assert report['mask_rate']==.1
        assert abs(p.mean()-report['test_expected_positive_share'])<1e-7
        data=ad.read_h5ad(d['corrupted'],backed='r')
        split=pd.read_parquet(d['splits']).set_index('cell_id').loc[data.obs_names,'split'].to_numpy()
        assert np.all(split[rows]=='test')
        assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
        coordinates=pd.read_parquet(d['coordinates'])
        positive=coordinates.cell_index.to_numpy(dtype=np.int64)*data.n_vars+coordinates.gene_index.to_numpy(dtype=np.int64)
        y=np.isin(rows.astype(np.int64)*data.n_vars+cols,positive).astype(np.int8)
        assert abs(y.mean()-report['test_observed_positive_share'])<1e-12
        unit=data.obs[d['unit_column']].astype(str).to_numpy()[rows]
        assert all(not np.intersect1d(np.unique(unit),np.unique(previous[2])).size for previous in pieces)
        pieces.append((p,y,unit))
        provenance.append(dict(unit=d['key'],source=str(folder/'detection.npz'),candidates=len(p),positive=int(y.sum()),mean_p_recovery_error=float(p.mean()-report['test_expected_positive_share'])))
        data.file.close()
    p=np.concatenate([x[0] for x in pieces]); y=np.concatenate([x[1] for x in pieces]); units=np.concatenate([x[2] for x in pieces])
    names,u=np.unique(units,return_inverse=True)
    order=np.argsort(p,kind='stable'); p,y,u=p[order],y[order],u[order]
    ece,table=statistics(p,y)
    be,bb,draws=bootstrap(p,y,u)
    np.savez_compressed(OUT/f'{dataset}_bootstrap.npz',ece=be,bias=bb,draws=draws,units=names)
    el,eh=np.quantile(be,[.025,.975]); bl,bh=np.quantile(bb,[.025,.975])
    results.append(dict(dataset=dataset,candidates=len(p),units=len(names),observed_share=float(y.mean()),mean_p=float(p.mean()),bias=float(p.mean()-y.mean()),bias_ci_low=bl,bias_ci_high=bh,ece=ece,ece_ci_low=el,ece_ci_high=eh,brier=float(np.mean((p-y)**2)),brier_constant=float(y.mean()*(1-y.mean()))))
    tables.extend(dict(dataset=dataset,**t) for t in table)
    ax.plot([0,1],[0,1],color='0.7',lw=1,ls='--'); ax.plot([t['mean_p'] for t in table],[t['observed_share'] for t in table],'o-',ms=4,lw=1)
    upper=1.1*max(max(t['mean_p'],t['observed_share']) for t in table)
    ax.set(title={'pancreas':'Pancreas','colon':'Colon','norman_crispra':'Norman CRISPRa'}[dataset],xlabel='Mean predicted p',xlim=(-.02*upper,upper),ylim=(-.02*upper,upper))
    print(dataset,'completed',flush=True)
axes[0].set_ylabel('Observed masked-positive share'); fig.tight_layout(); fig.savefig(OUT/'reliability.png',dpi=240)
pd.DataFrame(results).to_csv(OUT/'calibration.csv',index=False)
pd.DataFrame(tables).to_csv(OUT/'reliability.csv',index=False)
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
