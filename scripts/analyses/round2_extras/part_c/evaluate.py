import json,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0]=['scripts','src']
from masked_f1_units import Unit,load_units_manifest,load_unit,COUNT_SCALE,contract_metadata
from compute_matched_baseline_f1_curves import tie_broken_order
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_c')
UNITS=json.loads((O/'units.json').read_text())
METHODS=['EnImpute','DCA','scVI probability','Selector on EnImpute','Selector on scVI','Selector on MAGIC','SVD (all-cell)','Weighted kNN (all-cell)']

def evaluate(i):
    spec=UNITS[i]; dest=Path(spec['destination']); dest.mkdir(parents=True,exist_ok=True)
    manifest=dest/'evaluation_unit.json'
    manifest.write_text(json.dumps([{k:v for k,v in spec.items() if k in Unit.__dataclass_fields__}],indent=2)+'\n')
    u=load_units_manifest(manifest)[0]; d=load_unit(u)
    rr,cc=np.where((d.counts==0)&(d.split=='test')[:,None]); y=d.masked[rr,cc]
    labels=np.char.add(spec['base_key']+':',d.unit_labels[rr].astype(str)); groups,codes=np.unique(labels,return_inverse=True)
    positive=np.bincount(codes,weights=y,minlength=len(groups)); rows=[]; missing=[]
    for method in ['Safe Fusion',*METHODS]:
        selector=method=='Safe Fusion' or method.startswith('Selector on ')
        if method=='Safe Fusion': path=Path(spec['reference'])
        elif selector: path=Path(spec['selectors'][method.removeprefix('Selector on ')])
        else: path=Path(spec['extras'][method])/'mean.npy'
        if not path.exists(): missing.append(dict(method=method,path=str(path))); continue
        if selector:
            if path.suffix=='.npz':
                saved=np.load(path); np.testing.assert_array_equal(saved['rows'],rr); np.testing.assert_array_equal(saved['cols'],cc)
                np.testing.assert_array_equal(saved['labels'],y)
                score=saved['score'].astype(float)
            else: score=np.load(path).astype(float)
        else:
            metadata=contract_metadata(path.parent,d)
            value=np.asarray(np.load(path,mmap_mode='r')[rr,cc],dtype=np.float64)

            score=value if method=='scVI probability' else COUNT_SCALE[metadata['scale']](value,d.counts.sum(axis=1)[rr])
        assert score.shape==y.shape and np.isfinite(score).all(),(u.key,method)
        order=np.argsort(-score,kind='stable') if selector else tie_broken_order(score,u.tie_seed)
        rank=np.empty(len(order),np.int64); rank[order]=np.arange(len(order))
        for pct in range(1,11):
            chosen=rank<max(1,round(pct/100*len(rank)))
            selected=np.bincount(codes[chosen],minlength=len(groups)); hits=np.bincount(codes[chosen&y],minlength=len(groups))
            rows.extend(dict(dataset=u.dataset,seed=spec['seed'],unit=group,method=method,fraction=pct/100,positive=p,selected=s,hits=h) for group,p,s,h in zip(groups,positive,selected,hits))
    pd.DataFrame(rows).to_parquet(dest/'unit_counts.parquet',index=False)
    (dest/'evaluation_missing.json').write_text(json.dumps(missing,indent=2)+'\n')
    print(u.key,missing,flush=True)

def summarize():
    frames=[]; missing=[]
    for u in UNITS:
        path=Path(u['destination'])/'unit_counts.parquet'
        if path.exists(): frames.append(pd.read_parquet(path))
        else: missing.append(str(path))
    f=pd.concat(frames,ignore_index=True); f.to_parquet(O/'unit_counts.parquet',index=False)
    distributions={}; rows=[]; parity=[]
    old=pd.read_csv(O.parent.parent/'transductive_main/rebuilt_replicates/evaluation/per_seed.csv')
    for (dataset,seed),part in f.groupby(['dataset','seed']):
        ref=part[part.method=='Safe Fusion']; units=sorted(ref.unit.unique())
        expected=[u for u in UNITS if u['dataset']==dataset and u['seed']==seed]
        if len({x.split(':')[0] for x in units})!=len(expected): missing.append(f'{dataset} {seed}: incomplete Safe Fusion units'); continue
        draws=np.random.default_rng(1729).integers(0,len(units),size=(2000,len(units)))
        for method in ['Safe Fusion',*METHODS]:
            block=part[part.method==method]
            if set(block.unit)!=set(units) or len(block)!=10*len(units): missing.append(f'{dataset} {seed} {method}: incomplete units'); continue
            def matrix(name): return block.pivot(index='unit',columns='fraction',values=name).loc[units].to_numpy()
            p=matrix('positive'); s=matrix('selected'); h=matrix('hits')
            observed=np.mean(200*h.sum(0)/(s.sum(0)+p.sum(0)))
            sample=np.mean(200*h[draws].sum(1)/(s[draws].sum(1)+p[draws].sum(1)),axis=1)
            values=np.r_[observed,sample]; distributions[dataset,seed,method]=values
            lo,hi=np.quantile(sample,[.025,.975]); rows.append(dict(dataset=dataset,seed=seed,method=method,estimate=observed,lower=lo,upper=hi,n_units=len(units)))
            if method=='Safe Fusion':
                source=old[(old.dataset==dataset)&(old.seed==seed)&(old.method=='transductive')].iloc[0]
                np.testing.assert_allclose([observed,lo,hi],[source.estimate,source.lower,source.upper],rtol=0,atol=1e-12)
                parity.append(dict(dataset=dataset,seed=int(seed),reference_exact_atol=1e-12))
    pd.DataFrame(rows).to_csv(O/'absolute_per_seed.csv',index=False)
    paired=[]; across=[]
    for dataset in ['Pancreas','Colon','CRISPRa']:
        for method in METHODS:
            all_seeds=[]
            for seed in range(1729,1734):
                key=(dataset,seed,method)
                if key not in distributions: continue
                delta=distributions[dataset,seed,'Safe Fusion']-distributions[key]
                lo,hi=np.quantile(delta[1:],[.025,.975]); paired.append(dict(dataset=dataset,seed=seed,method=method,difference=delta[0],lower=lo,upper=hi))
                all_seeds.append(delta)
            if len(all_seeds)!=5: continue
            a=np.array(all_seeds); mean=a.mean(0); lo,hi=np.quantile(mean[1:],[.025,.975])
            across.append(dict(dataset=dataset,method=method,mean_difference=mean[0],seed_sd=np.std(a[:,0],ddof=1),lower=lo,upper=hi,seeds_higher=int((a[:,0]>0).sum()),n_seeds=5))
    pd.DataFrame(paired).to_csv(O/'paired_per_seed.csv',index=False)
    pd.DataFrame(across).to_csv(O/'across_seeds.csv',index=False)
    (O/'verification.json').write_text(json.dumps(dict(reference_parity=parity,missing=missing,complete=len(across)==24,draws=2000,seed=1729),indent=2)+'\n')
    print('across rows',len(across),'missing',len(missing),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='summary': summarize()
    else: evaluate(int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))+int(sys.argv[1]))
