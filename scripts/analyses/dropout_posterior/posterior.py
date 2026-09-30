import os,sys,json,time
from pathlib import Path
ROOT=Path.cwd(); OUT=ROOT/'artifacts/paper_evidence/review_round4/dropout_posterior'
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import v2_value_models as v
from selector_attribution import exact_topk

def posterior(d,rate,theta=None):
    d=np.asarray(d,dtype=np.float64); rate=np.asarray(rate,dtype=np.float64)
    if not (np.all(np.isfinite(d)) and np.all((d>=0)&(d<=1)) and np.all(np.isfinite(rate)) and np.all(rate>=0)):
        raise ValueError('invalid detection probability or rate')
    if theta is not None and not np.all(np.isfinite(theta)&(theta>0)): raise ValueError('invalid theta')
    logq=-rate if theta is None else -theta*np.log1p(rate/theta)
    detected=-np.expm1(logq)
    pi=np.divide(d,detected,out=np.ones_like(d),where=detected>0)
    pi[d==0]=0
    pi=np.minimum(1,pi)
    p=np.zeros_like(d); p[pi==1]=1
    inside=(pi>0)&(pi<1)
    from scipy.special import expit
    p[inside]=expit(np.log(pi[inside])+logq[inside]-np.log1p(-pi[inside]))
    return p,pi

def secondary_rate(bench,rows,cols,seed):

    context=v.value_context(bench,bench.training)
    fr,fc,y=v.positive_entries(bench,bench.training)
    logs,features=v.value_inputs(bench,context,fr,fc)
    print('secondary conditional fit',len(y),'positives',flush=True)
    conditional=v.fit_value_model('boosted',logs,features,np.log1p(y),seed)
    keep=v.subsample(len(y),seed)
    print('secondary Poisson fit',len(keep),'positives',flush=True)
    model=v.fit_poisson_offset(conditional[1],features[keep],y[keep],v.conditional_offset(conditional,logs[keep],features[keep]))
    print('secondary prediction',len(rows),'zeros',flush=True)
    parts=[]
    for start in range(0,len(rows),v.PREDICT_BATCH):
        logs,features=v.value_inputs(bench,context,rows[start:start+v.PREDICT_BATCH],cols[start:start+v.PREDICT_BATCH])
        mean=v.conditional_offset(conditional,logs,features)*model.predict(features)
        parts.append(v.truncated_poisson_rate(mean))
    return np.concatenate(parts).astype(np.float32)

def check():
    d=np.array([0,.2,.8,.2,.8]); rate=np.array([0,0,0,2,2])
    p,pi=posterior(d,rate)
    assert np.array_equal(p[:3],[0,1,1])
    np.testing.assert_allclose(p[3:],pi[3:]*np.exp(-2)/(pi[3:]*np.exp(-2)+1-pi[3:]))
    assert posterior(np.array([1.]),np.array([1000.]))[0][0]==1
    lam=v.truncated_poisson_rate(np.array([0.,1.,1.01,2.,10.]))
    np.testing.assert_allclose(lam[2:]/-np.expm1(-lam[2:]),[1.01,2,10])
    np.testing.assert_allclose(posterior(d,rate,np.full(5,1e8))[0],p,rtol=1e-7)
    print('posterior checks passed')

def run(index):
    spec=json.loads((OUT/'manifest.json').read_text())[index]; dest=OUT/spec['id']; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists(): return
    t=time.time(); alias=dest/'teachers'; alias.mkdir(exist_ok=True)
    for name,path in zip(v.TEACHERS,spec['teachers']):
        path=(ROOT/path).resolve()
        if not (path/'mean.npy').exists(): raise FileNotFoundError(path/'mean.npy')
        link=alias/name
        if not link.exists(): link.symlink_to(path,target_is_directory=True)
    bench=v.load_benchmark(ROOT/spec['input'],ROOT/spec['coordinates'],ROOT/spec['splits'],alias)
    rows,cols=np.nonzero((bench.counts==0)&(bench.split=='test')[:,None])
    cache=ROOT/spec['cache'] if spec.get('cache') else None
    saved=cache/'detection.npz' if cache else None
    reproduction={}
    if spec.get('exact_calibration'):
        exact=dest/'exact_calibration'
        if not (exact/'detection.npz').exists():
            import subprocess
            args=[str(ROOT/'.venv/bin/python'),str(Path('scripts/analyses/dropout_posterior/calibrate_exact.py')),'--corrupted',spec['input'],'--truth',spec['truth'],'--coordinates',spec['coordinates'],'--splits',spec['splits'],'--fusion-contract',spec['fusion'],'--output-dir',str(exact),'--fit-split',spec['fit_split'],'--architecture','mlp','--budget-mode','apply_topk','--budgets','.01','--curve-points','2','--seed',str(spec['seed']),'--detection-rule-mask-rate','.1']
            for teacher in spec['teachers']:args+=['--teacher-contract',teacher]
            subprocess.run(args,check=True)
        source=np.load(exact/'detection.npz');score,p,d=source['score'],source['probability'],source['detection'];report={'exact_production_calibration':str(exact)}
    elif saved and saved.exists():
        provenance=json.loads((cache/'report.json').read_text())['paths']
        for old,new in [('input','input'),('coordinates','coordinates'),('splits','splits')]:
            if (ROOT/provenance[old]).resolve()!=(ROOT/spec[new]).resolve():
                if old=='splits':
                    a=pd.read_parquet(ROOT/provenance[old]).set_index('cell_id').loc[bench.cell_ids,'split']
                    b=pd.read_parquet(ROOT/spec[new]).set_index('cell_id').loc[bench.cell_ids,'split']
                    pd.testing.assert_series_equal(a,b,check_names=False)
                else: raise ValueError((old,provenance[old],spec[new]))
        source=np.load(saved)
        assert np.array_equal(rows,source['rows']) and np.array_equal(cols,source['cols'])
        score,p,d=source['score'],source['probability'],source['detection']; report={'reused':str(saved)}
    elif spec.get('selector') and (ROOT/spec['selector']/'detection.npz').exists():
        source=np.load(ROOT/spec['selector']/'detection.npz')
        assert np.array_equal(rows,source['rows']) and np.array_equal(cols,source['cols'])
        score=source['score'] if 'score' in source else source['scores'];d=source['detection'];p=source['probability'] if 'probability' in source else .1*d/(1-d+.1*d); report={'reused_selector':spec['selector']}
    elif spec.get('selector') and (ROOT/spec['selector']/'isotonic_map.npz').exists():
        table=pd.read_parquet(ROOT/spec['selector']/'selected_gene_scores.parquet',columns=['split','cell_index','gene_index','selector_score'],filters=[('split','=','test')]).sort_values(['cell_index','gene_index'])
        assert np.array_equal(rows,table.cell_index) and np.array_equal(cols,table.gene_index)
        iso=np.load(ROOT/spec['selector']/'isotonic_map.npz'); score=table.selector_score.to_numpy()
        p=np.interp(score,iso['score'],iso['probability']).astype(np.float32); d=v.detection_probability(p,.1).astype(np.float32)
        report={'reused_isotonic_map':spec['selector']}
    elif (dest/'detection.npz').exists():
        source=np.load(dest/'detection.npz'); score,p,d=source['score'],source['probability'],source['detection']; report={'resumed':True}
    else:
        score,p,d,report=v.selector_detection(bench,bench.split==spec['fit_split'],rows,cols,.1,spec['seed'])
    np.savez(dest/'detection.npz',rows=rows.astype(np.int32),cols=cols.astype(np.int32),score=score,probability=p,detection=d)
    if spec.get('baseline') and (ROOT/spec['baseline']).exists():
        old=np.load(ROOT/spec['baseline']); assert len(old)==len(score)
        err=float(np.max(np.abs(old-score))); reproduction['max_score_difference']=err

        np.testing.assert_allclose(old,score,rtol=0,atol=1e-6)
        score=old
    elif spec.get('selector'):
        selector=ROOT/spec['selector']; checks=[]
        for b in np.arange(1,11)/100:
            path=selector/f"safe_fusion_calibrated_mlp_topk_{str(round(float(b),2)).replace('.','p')}"/'mean.npy'
            if not path.exists(): continue
            old=np.load(path,mmap_mode='r')[rows,cols]>0
            selected=exact_topk(score,max(1,round(float(b)*len(score))))
            assert np.array_equal(old,selected),(spec['id'],b,int((old!=selected).sum()))
            checks.append(float(b))
        if not checks: raise ValueError('no existing selector fills for reproduction')
        reproduction['exact_fill_fractions']=checks
    else: raise ValueError('no baseline reproduction source')
    fusion=ROOT/spec['fusion']; meta=json.loads((fusion/'metadata.json').read_text())
    assert meta['cell_ids']==bench.cell_ids and meta['gene_ids']==bench.gene_ids and meta['scale']=='counts'
    m=np.load(fusion/'mean.npy',mmap_mode='r')[rows,cols].astype(np.float64)
    rate=v.truncated_poisson_rate(m)
    primary,pi=posterior(d,rate)
    arrays=dict(rows=rows.astype(np.int32),cols=cols.astype(np.int32),baseline=score,detection=d,conditional=m,weighted=d*m,rate=rate,posterior=primary)
    secondary=cache/'rate_poisson/mean.npy' if cache else None
    if secondary and secondary.exists():
        fitted_rate=np.load(secondary,mmap_mode='r')[rows,cols]
    else:
        fitted_rate=secondary_rate(bench,rows,cols,spec['seed'])
    arrays['fitted_rate']=fitted_rate
    arrays['posterior_fitted_rate']=posterior(d,fitted_rate)[0]

    theta_root=Path(spec.get('theta_root',spec['teachers'][-1]))
    theta_path=theta_root/'theta.npy'
    standard=Path(spec.get('standard_scvi','/nonexistent'))
    if spec['chain']=='transductive' and not theta_path.exists() and (standard/'model/model.pt').exists():
        import subprocess
        theta_root=dest/'dispersion'
        subprocess.run([str(ROOT/'.conda-scvi-current/bin/python'),str(ROOT/'scripts/v2_selector_scvi_teacher.py'),'--from-model',str(standard),'--output',str(theta_root)],check=True)
        theta_path=theta_root/'theta.npy'
    if theta_path.exists():
        theta=np.load(theta_path); model=np.load(theta_root/'model_index.npy')
        arrays['posterior_nb']=posterior(d,rate,theta[model[rows],cols])[0]
    for name,path in zip(v.TEACHERS,spec['teachers']): arrays['teacher_'+name]=bench.teachers[name][rows,cols]
    if spec['chain']=='transductive' and (spec['analysis'].startswith('down_') or spec['analysis'] in ['sex','protein','sle']):
        probability=scvi_comparator(spec,rows,cols)
        if probability is not None:arrays['scvi_probability']=probability
    np.savez(dest/'scores.npz',**arrays)
    (dest/'complete.json').write_text(json.dumps(dict(spec=spec,calibration=report,reproduction=reproduction,n_candidates=len(rows),m_le_one=float(np.mean(m<=1)),pi_clipped_one=float(np.mean(pi==1)),nb_available='posterior_nb' in arrays,seconds=time.time()-t,slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_node=os.environ.get('SLURMD_NODENAME')),indent=2)+'\n')
    print(spec['id'],'complete',round(time.time()-t,1),flush=True)

def scvi_comparator(spec,rows,cols):
    import subprocess
    source=ROOT/spec.get('standard_scvi','nonexistent')
    if not (source/'model/model.pt').exists(): return None
    dest=OUT/'comparators'/spec['analysis']/spec['key']/'scvi'
    if not (dest/'metadata.json').exists():
        subprocess.run([str(ROOT/'.conda-scvi-current/bin/python'),str(ROOT/'scripts/nonzero_probability.py'),'--method','scvi','--contract',str(source),'--corrupted',spec['input'],'--output',str(dest)],check=True)
    return np.load(dest/'mean.npy',mmap_mode='r')[rows,cols]

if __name__=='__main__':
    if sys.argv[1]=='check': check()
    else: run(int(sys.argv[1]))
