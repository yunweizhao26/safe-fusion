import json, subprocess, sys
from pathlib import Path
sys.path[:0]=['scripts','src']
import numpy as np
import pandas as pd
BASE=Path('artifacts/paper_evidence/review_round4/transductive_downstream')
OUT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras/B_thinning')
D=[d for d in json.loads((BASE/'manifest.json').read_text()) if d['design']=='thinning']
NAMES=['alra','selector_scvi','selector_magic']; PCTS=[1,5,10]

def run(script,*args):
    cmd=['.venv/bin/python',str(script),*map(str,args)]
    with (OUT/'commands.jsonl').open('a') as f: f.write(json.dumps(cmd)+'\n')
    subprocess.run(cmd,check=True)

def parity():
    import r3_downstream_thinning as thin
    thin.METHODS={n:n for n in ['safe_fusion','transductive','detection','svd','magic','scvi']}
    old=pd.read_csv(BASE/'summary/thinning_reference.csv')
    new=thin.summarize_units(pd.read_csv(BASE/'summary/thinning_unit_values.csv'),2000,1729)
    keys=['dataset','endpoint','method','fill_pct']; old=old.set_index(keys).sort_index(); new=new.set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index,new.index)
    cols=old.select_dtypes('number').columns
    diff=np.abs(old[cols].to_numpy()-new[cols].to_numpy())
    assert np.allclose(old[cols],new[cols],rtol=0,atol=2e-14,equal_nan=True)
    (OUT/'reproduction.json').write_text(json.dumps(dict(rows=len(old),numeric_comparisons=int(diff.size),max_absolute_difference=float(np.nanmax(diff)),tolerance=2e-14,passed=True),indent=2)+'\n')
    new.reset_index().to_csv(OUT/'existing_reproduced.csv',index=False)
    print('existing Table 3 bootstrap reproduced',flush=True)

def fit(index):
    assert json.loads((OUT/'reproduction.json').read_text())['passed']
    d=D[index]; p=Path(d['dir']); dest=OUT/d['key']; dest.mkdir(parents=True,exist_ok=True)
    alra=dest/'alra'
    if not (alra/'metadata.json').exists():
        run('scripts/run_alra_baseline.py','--corrupted',p/'hybrid.h5ad','--splits',p/'splits.parquet','--output',alra,'--seed',1729)
    for name in ['scvi','magic']:
        sel=dest/f'selector_{name}'
        if not (sel/'test_scores.npz').exists():
            run('scripts/stacked_selector_scores.py','--corrupted',p/'hybrid.h5ad','--coordinates',p/'coordinates.parquet','--splits',p/'splits.parquet','--fit-split',d['fit_split'],'--unit-column',d.get('unit_column','stage_rank'),'--contract',p/'standard'/name,'--name','scVI' if name=='scvi' else 'MAGIC','--output-dir',sel,'--seed',1729)
    fill(d,dest)
    evaluate(d,dest)
    (dest/'DONE').write_text('completed\n')

def fill(d,dest):
    import anndata as ad
    from masked_f1_units import dense
    from selector_attribution import exact_topk
    from safefusion_benchmark.contracts import write_output_contract
    p=Path(d['dir']); a=ad.read_h5ad(p/'recorded.h5ad'); h=ad.read_h5ad(p/'hybrid.h5ad')
    recorded=dense(a.layers['corrupted_counts']).astype(np.float32)
    split=pd.read_parquet(p/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy()
    test=split=='test'; assert np.array_equal(recorded[test],dense(h.layers['corrupted_counts'])[test])
    rows,cols=np.where((recorded==0)&test[:,None]); del h
    checks=[]
    for name in NAMES:
        source=dest/'alra' if name=='alra' else p/'transductive'/f'{name.removeprefix("selector_")}_inductive'
        meta=json.loads((source/'metadata.json').read_text()); value=np.load(source/'mean.npy',mmap_mode='r')
        assert meta['cell_ids']==a.obs_names.astype(str).tolist() and meta['gene_ids']==a.var_names.astype(str).tolist()
        assert meta['scale']=='counts'
        values=np.maximum(value[rows,cols],0)
        if name=='alra': scores=values
        else:
            saved=np.load(dest/name/'test_scores.npz'); assert np.array_equal(saved['rows'],rows) and np.array_equal(saved['cols'],cols)
            scores=saved['score']
        for pct in PCTS:
            target=dest/f'{name}_{pct}pct'
            if (target/'metadata.json').exists(): continue
            k=max(1,round(pct/100*len(rows))); selected=exact_topk(scores,k)
            result=recorded.copy(); result[rows[selected],cols[selected]]=values[selected]
            assert selected.sum()==k and np.array_equal(result[~test],recorded[~test])
            assert np.array_equal(result[recorded>0],recorded[recorded>0])
            metadata={**meta,'method':f'{name}_{pct}pct','parameters':{**meta['parameters'],'reviewer_extras':{'source_contract':str(source),'fill_fraction':pct/100,'ranking':'value' if name=='alra' else 'single_method_mlp','selector_fit_split':d['fit_split'] if name!='alra' else None,'test_labels_used':False}}}
            write_output_contract(target,result,metadata)
            checks.append(dict(method=name,pct=pct,candidates=len(rows),selected=int(selected.sum()),changed=int(np.sum(values[selected]>0))))
    if checks: pd.DataFrame(checks).to_csv(dest/'fill_checks.csv',index=False)

def evaluate(d,dest):
    p=Path(d['dir']); key=d['key']; tissue=key.split('_')[0]
    out=dest/'evaluation'
    if (out/'bootstrap_summary.parquet').exists(): return
    methods=sum((['--method',f'{n}_{pct}pct={dest}/{n}_{pct}pct'] for n in NAMES for pct in PCTS),[])
    args=['--truth',p/'truth.h5ad','--corrupted',p/'recorded.h5ad','--splits',p/'splits.parquet',*methods,'--output-dir',out,'--bootstrap',2000,'--seed',1729,'--allow-transductive']
    if tissue=='pancreas': run(BASE/'code/pancreas_fold.py',*args,'--coordinates',p/'empty_coordinates.parquet','--fold',key[-1])
    elif tissue=='colon': run('scripts/evaluate_colon_donor_biology.py',*args,'--coordinates',p/'empty_coordinates.parquet','--marker-panel','source')
    elif key=='zebrafish': run('scripts/evaluate_trajectory_preservation.py',*args)
    else: run('scripts/evaluate_interventional_grn.py',*args,'--dataset','norman_crispra','--intervention','gain_of_function','--publication-doi','10.1126/science.aax4438')

def summarize():
    import r3_downstream_thinning as thin
    thin.METHODS={n:n for n in NAMES}
    old=pd.read_csv(BASE/'summary/thinning_unit_values.csv',dtype={'unit':str})
    frames=[]; checks=[]
    specs={'colon':[('Marker AUPRC','canonical_marker_pr_auc'),('Reference mapping F1','cell_identity_macro_f1')],'pancreas':[('Marker AUPRC','canonical_marker_pr_auc'),('Reference mapping F1','cell_identity_macro_f1')],'zebrafish':[('Dynamics','dynamic_gene_spearman'),('Stage error','stage_rank_mae')],'norman_crispra':[('Edge AUPRC','edge_pr_auc_q10')]}
    for dataset,endpoints in specs.items():
        donors=dataset in ['colon','pancreas']; unit='donor' if donors else 'unit'
        keys=[f'{dataset}_{i}' for i in range(3)] if donors else [dataset]
        file='donor_metrics.parquet' if donors else 'unit_metrics.parquet'
        frame=pd.concat([pd.read_parquet(OUT/k/'evaluation'/file) for k in keys])
        for endpoint,metric in endpoints:
            f=frame[frame.metric==metric].copy(); f[unit]=f[unit].astype(str)
            pivot=f.pivot(index=unit,columns='method',values='value')
            reference=old[(old.dataset==dataset)&(old.endpoint==endpoint)&(old.matrix=='corrupted_raw')].set_index('unit').reindex(pivot.index)
            assert reference.unthinned.notna().all()
            for column,saved in [('corrupted_raw','value'),('reference_truth','thinned_truth_pipeline')]:
                delta=np.nanmax(abs(pivot[column].to_numpy()-reference[saved].to_numpy()))
                assert delta<1e-12,(dataset,endpoint,column,delta)
                checks.append(dict(dataset=dataset,endpoint=endpoint,reference=column,max_difference=delta))
            for name in ['corrupted_raw']+[f'{n}_{p}pct' for n in NAMES for p in PCTS]:
                frames.append(pd.DataFrame(dict(dataset=dataset,endpoint=endpoint,unit=pivot.index,stratum=reference.stratum.to_numpy(),matrix=name,value=pivot[name].to_numpy(),unthinned=reference.unthinned.to_numpy(),thinned_truth_pipeline=reference.thinned_truth_pipeline.to_numpy())))
    values=pd.concat(frames,ignore_index=True); values.to_csv(OUT/'unit_values.csv',index=False)
    result=thin.summarize_units(values,2000,1729); result.to_csv(OUT/'new_rows.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'reference_parity.csv',index=False)
    allrows=pd.concat([pd.read_csv(OUT/'existing_reproduced.csv'),result]); allrows.to_csv(OUT/'all_rows.csv',index=False)
    print('new comparators completed',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='check': parity()
    elif sys.argv[1]=='fit': fit(int(sys.argv[2]))
    elif sys.argv[1]=='summary': summarize()
