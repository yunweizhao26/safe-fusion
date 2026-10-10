from common import *
import json
import argparse
import numpy as np
import pandas as pd
from masked_f1_units import load_units_manifest, load_unit, count_scale_values
from compute_matched_baseline_f1_curves import tie_broken_order
from summarize_seed_replicates import COUNT_SCALE, f1, GRID, INTEGER, unit_curves

def counts(u,score,selector):
    d=load_unit(u)
    rr,cc=np.where((d.counts==0)&(d.split=='test')[:,None])
    y=d.masked[rr,cc]
    labels=np.char.add(u.key.split('_s')[0]+':',d.unit_labels[rr].astype(str))
    groups,codes=np.unique(labels,return_inverse=True)
    order=np.argsort(-score.astype(float),kind='stable') if selector else tie_broken_order(score,u.tie_seed)
    rank=np.empty(len(order),np.int64); rank[order]=np.arange(len(order))
    positive=np.bincount(codes,weights=y,minlength=len(groups))
    selected=[];hits=[]
    for b in GRID:
        chosen=rank<max(1,int(round(b*len(rank))))
        selected.append(np.bincount(codes[chosen],minlength=len(groups)))
        hits.append(np.bincount(codes[chosen&y],minlength=len(groups)))
    return groups,positive,np.asarray(selected).T,np.asarray(hits).T

parser=argparse.ArgumentParser()
parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1729,1734)))
parser.add_argument('--output-dir',type=Path,default=OUT/'rebuilt_replicates/evaluation')
args=parser.parse_args()
rows=[]; distributions={}; parity=[]; curve_rows=[]
for seed in args.seeds:
    by_dataset={}
    for u in units():
        root=OUT/'rebuilt_replicates'/f'{u.key}_s{seed}'
        ru=u if seed==1729 else load_units_manifest(root/'unit.json')[0]
        d=load_unit(ru)
        rr,cc=np.where((d.counts==0)&(d.split=='test')[:,None])
        scores={}
        for name in ['SVD','Weighted kNN','MAGIC','scVI']:
            contract=ru.contracts[name]
            metadata=json.loads((contract/'metadata.json').read_text())
            value=np.asarray(np.load(contract/'mean.npy',mmap_mode='r')[rr,cc],dtype=np.float64)

            scores[name]=(COUNT_SCALE[metadata['scale']](value,d.counts.sum(axis=1)[rr]),False)
        score_root=fusion_root(u)/'selectors'/u.key if seed==1729 else root/'scores'/ru.key
        for name,slug in [('inductive','safe_fusion'),('transductive','safe_fusion_transductive')]:
            scores[name]=(np.load(score_root/slug/'test_scores.npy'),True)

        selector=OUT/'rebuilt_replicates/evaluator_overlay'/ru.key
        link(score_root/'safe_fusion/report.json',selector/'calibration_report.json')
        spec=dict(corrupted=ru.corrupted,coordinates=ru.coordinates,splits=ru.splits,tie_seed=ru.tie_seed,
                  selector=selector,contracts={n:ru.contracts[n] for n in ['SVD','Weighted kNN','MAGIC','scVI']})
        legacy=unit_curves(spec)
        for name,(score,selector) in scores.items():
            part=counts(ru,score,selector)
            if name!='transductive':
                old=legacy['Safe Fusion' if name=='inductive' else name]
                assert np.array_equal(part[2].sum(0),old[:,0]) and np.array_equal(part[3].sum(0),old[:,1]),(ru.key,name)
                parity.append(dict(seed=seed,unit=ru.key,method=name,matched_fractions=100))
            by_dataset.setdefault((u.dataset,name),[]).append(part)
    for (dataset,name),parts in by_dataset.items():
        groups=np.concatenate([x[0] for x in parts])
        order=np.argsort(groups)
        p=np.concatenate([x[1] for x in parts])[order]
        s=np.concatenate([x[2] for x in parts])[order]
        h=np.concatenate([x[3] for x in parts])[order]
        draws=np.random.default_rng(1729).integers(0,len(p),size=(2000,len(p)))
        curve=200*h.sum(0)/(s.sum(0)+p.sum())
        curve_rows += [dict(dataset=dataset,seed=seed,method=name,fraction=float(b),f1=float(x)) for b,x in zip(GRID,curve)]
        h,s=h[:,INTEGER],s[:,INTEGER]
        point=np.mean(curve[INTEGER])
        sample=np.mean(200*h[draws].sum(1)/(s[draws].sum(1)+p[draws].sum(1)[:,None]),axis=1)
        distributions.setdefault((dataset,name),[]).append(np.r_[point,sample])
        low,high=np.quantile(sample,[.025,.975])
        rows.append(dict(dataset=dataset,seed=seed,method=name,estimate=point,lower=low,upper=high,n_units=len(p)))
dest=args.output_dir
dest.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(dest/'per_seed.csv',index=False)
(dest/'settings.json').write_text(json.dumps(dict(seeds=args.seeds,draws=2000,bootstrap_seed=1729),indent=2)+'\n')
pd.DataFrame(curve_rows).to_csv(dest/'curves.csv',index=False)
(dest/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
curves=pd.DataFrame(curve_rows)
wins=[]
for (dataset,seed),part in curves.groupby(['dataset','seed']):
    pivot=part.pivot(index='fraction',columns='method',values='f1')
    means=pivot.iloc[INTEGER].mean()
    for model in ['inductive','transductive']:
        wins.append(dict(dataset=dataset,seed=seed,model=model,best_fractions=int((pivot[model]>=pivot[['SVD','Weighted kNN','MAGIC','scVI']].max(axis=1)).sum()),
                         best_mean=bool(means[model]>means[['SVD','Weighted kNN','MAGIC','scVI']].max())))
pd.DataFrame(wins).to_csv(dest/'wins.csv',index=False)
summary=[]
for (dataset,name),samples in distributions.items():
    a=np.asarray(samples)
    mean=a.mean(axis=0)
    lo,hi=np.quantile(mean[1:],[.025,.975])
    summary.append(dict(dataset=dataset,method=name,estimate=mean[0],lower=lo,upper=hi,seed_sd=np.std(a[:,0],ddof=1)))
    np.save(dest/f'{dataset}_{name.replace(" ","_")}_draws.npy',a)
pd.DataFrame(summary).to_csv(dest/'across_seeds.csv',index=False)
paired=[]
for dataset in ['Pancreas','Colon','CRISPRa']:
    for reference in ['inductive','transductive']:
        for other in ['SVD','Weighted kNN','MAGIC','scVI','inductive']:
            if other==reference: continue
            delta=(np.asarray(distributions[dataset,reference])-np.asarray(distributions[dataset,other])).mean(0)
            lo,hi=np.quantile(delta[1:],[.025,.975])
            paired.append(dict(dataset=dataset,reference=reference,comparator=other,estimate=delta[0],lower=lo,upper=hi))
pd.DataFrame(paired).to_csv(dest/'paired_differences.csv',index=False)
