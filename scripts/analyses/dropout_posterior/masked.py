from posterior import *
from masked_f1_units import Unit,load_unit
from fusion_value_bootstrap import Group,group_statistics,statistics_from_sums,interval
from compute_matched_baseline_f1_curves import tie_broken_order

def run(dataset):
    specs=[s for s in json.loads((OUT/'manifest.json').read_text()) if s['analysis']=='masked' and s['dataset']==dataset]
    keys=sorted(set(s['key'] for s in specs)); loaded=[]; labels=[]
    for key in keys:
        spec=next(s for s in specs if s['key']==key); fields=dict(spec['unit'])
        for k in ['corrupted','coordinates','splits','truth','selector_dir']: fields[k]=ROOT/fields[k]
        fields['contracts']={k:ROOT/p for k,p in fields['contracts'].items()}
        unit=Unit(**fields); data=load_unit(unit)
        rr,cc=np.nonzero((data.counts==0)&(data.split=='test')[:,None]); lab=np.char.add(key+':',data.unit_labels[rr].astype(str))
        loaded.append((unit,data,rr,cc,lab)); labels.append(lab)
    bio=np.unique(np.concatenate(labels)); n=len(bio)
    draws=np.random.default_rng(1729).integers(0,n,size=(2000,n)); w=np.zeros((2001,n));w[0]=1
    np.add.at(w,(np.repeat(np.arange(1,2001),n),draws.ravel()),1)
    sums={}; missing=[]
    for unit,data,rr,cc,lab in loaded:
        scores={}; seed=unit.tie_seed
        for chain in ['inductive','transductive']:
            spec=next(s for s in specs if s['key']==unit.key and s['chain']==chain)
            z=np.load(OUT/spec['id']/'scores.npz'); assert np.array_equal(rr,z['rows']) and np.array_equal(cc,z['cols'])
            for name in ['baseline','posterior','posterior_fitted_rate','posterior_nb']:
                if name in z and not (name=='posterior_nb' and chain=='inductive'): scores[chain+':'+name]=(z[name],False)
            if dataset=='Colon':
                for name in v.TEACHERS: scores[chain+':teacher_'+name]=(z['teacher_'+name],True)
        prob=Path(spec.get('probability',''))
        if prob.is_file(): scores['scVI P(X>0)']=(np.load(prob,mmap_mode='r')[rr,cc],True)
        for method in ['scimpute','screcover','dca_zinb','scvi_zinb']:
            p=ROOT/'artifacts/paper_evidence/review_round3/comparators/fits'/method/(unit.key.replace('colon_','colon_cf_'))/'dropout_probability.npy'
            if p.exists(): scores[method+' dropout']=(np.load(p,mmap_mode='r')[rr,cc],True)
            else: missing.append(str(p))
        codes=np.searchsorted(bio,lab); group=Group(unit,rr,cc,data.masked[rr,cc],codes,np.unique(codes))
        for name,(score,random_ties) in scores.items():
            order=tie_broken_order(score,seed) if random_ties else np.argsort(-score.astype(np.float64),kind='stable')
            rank=np.empty(len(order),dtype=np.int64); rank[order]=np.arange(len(order))
            result=group_statistics(rank,group,w); result.pop('unit_counts')
            if name not in sums: sums[name]=result
            else: sums[name]={k:sums[name][k]+result[k] for k in result}
            print(dataset,unit.key,name,flush=True)
    stats={name:statistics_from_sums(s) for name,s in sums.items()}
    oldpath=ROOT/'artifacts/paper_evidence'/('review_round3/colon_crossfit/fusion_value/evaluation/absolute.csv' if dataset=='Colon' else 'review_round2/fusion_value/evaluation/absolute.csv')
    old=pd.read_csv(oldpath); gates=[]
    for name,oldname in [('inductive:baseline','Safe Fusion'),('transductive:baseline','Safe Fusion (transductive)')]:
        for metric,values in stats[name].items():
            row=old[(old.dataset==dataset)&(old.method==oldname)&(old.statistic==metric)].iloc[0]
            got=[100*values[0],*np.quantile(100*values[1:],[.025,.975])]; expected=[row.estimate_percent,row.lower_percent,row.upper_percent]
            np.testing.assert_allclose(got,expected,rtol=0,atol=.000051)
            gates.append(dict(method=name,metric=metric,max_error=float(np.max(np.abs(np.array(got)-expected)))))
    rows=[]
    for method,metrics in stats.items():
        for metric,values in metrics.items():
            if metric=='mean_f1_1_to_2': continue
            for chain in ['inductive','transductive']:
                delta=values-stats[chain+':baseline'][metric]
                rows.append(dict(dataset=dataset,method=method,metric=metric,reference=chain+':baseline',estimate=100*values[0],low=100*interval(values[1:])[0],high=100*interval(values[1:])[1],difference=100*delta[0],difference_low=100*interval(delta[1:])[0],difference_high=100*interval(delta[1:])[1],n_units=n))
    dest=OUT/'evaluation/masked'/dataset; dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'results.csv',index=False)
    (dest/'audit.json').write_text(json.dumps(dict(reproduction=gates,missing=missing,draws=2000,seed=1729),indent=2)+'\n')
    np.savez(dest/'bootstrap.npz',**{name+'|'+metric:values for name,m in stats.items() for metric,values in m.items()})
if __name__=='__main__': run(sys.argv[1])
