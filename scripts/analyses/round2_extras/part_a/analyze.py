import sys,json,re,argparse,importlib.util,gc
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0]=['scripts','scripts/standard_imputers','src']
import evaluate_knockdown_zero_analyses as kd
from evaluate_r3_known_zeros import ranked_method
from selector_attribution import exact_topk
from disease_control_common import stratified_draws,dense
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_a')
R=O.parent.parent
E=R.parent
LABEL=R/'label_baselines'
VIEW=R/'transductive_references/knockdown/evaluation_inputs'
NAMES={'Safe Fusion':'safe_fusion','SVD':'svd','Weighted kNN':'weighted_knn','MAGIC':'magic_standard','scVI':'scvi_standard','ALRA':'alra','SAVER':'saver','DCA':'DCA','scImpute':'scImpute','EnImpute':'EnImpute','Safe Fusion, labels':'safe_fusion_condition','Weighted kNN, labels':'knn_condition','Per-label expected count':'label_detection'}
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
lb=module('label_screens',LABEL/'screens/baselines.py')
sex=module('label_sex',LABEL/'sex/evaluate.py')
def paper():
    snapshot=pd.read_csv(O/'table2_original_snapshot.csv')
    return {name:snapshot[snapshot.method==name].sort_values('column').paper.tolist() for name in NAMES}
def gate():
    p=paper(); rows=[]
    a=pd.read_csv(LABEL/'screens/verification_transductive/auroc.csv')
    b=pd.read_csv(E/'review_round3/comparators/evaluation/known_zeros/knockdown/auroc.csv')
    c=pd.read_csv(LABEL/'screens/baseline_per_target.csv')
    for name,key in NAMES.items():
        source=c if key in ['knn_condition','label_detection'] else b if key in ['DCA','scImpute','EnImpute'] else a
        for j,(dataset,adj) in enumerate([('adamson_crispri','none'),('adamson_crispri','depth_strata'),('norman_crispra','none')]):
            f=source[(source.dataset==dataset)&(source.method==key)]
            if 'score_type' in f: f=f[f.score_type=='fill_order']
            assert len(f)==(26 if j<2 else 41),(name,j,len(f))
            val=f[adj].mean(); rows.append(dict(method=name,column=j,source_score='fill_order',source_value=val,paper=p[name][j],rounded_match=f'{val:.2f}'==f'{p[name][j]:.2f}'))
        for j,tissue in enumerate(['pancreas','colon'],3):
            if key in ['knn_condition','label_detection']:
                file='label_rankings_auroc.csv' if key=='knn_condition' else 'transductive_expected_count_auroc.csv'
                f=pd.read_csv(LABEL/'sex'/tissue/file)
                tissue_key='graph_smooth_donor' if key=='knn_condition' else 'transductive_expected_count'
                f=f[(f.method==tissue_key)&(f.gene=='RPS4Y1')&(f.adjustment=='none')]
            else:
                f=pd.read_csv(R/'sex_zero_comparators/reproduction_run/evaluation'/tissue/'auroc.csv')
                tissue_key='safe_fusion' if key=='safe_fusion' else 'safe_fusion_donor' if key=='safe_fusion_condition' else name
                f=f[(f.method==tissue_key)&(f.gene=='RPS4Y1')]
            assert len(f)==1,(name,tissue,len(f))
            val=f.auroc.iloc[0]; rows.append(dict(method=name,column=j,source_score='continuous',source_value=val,paper=p[name][j],rounded_match=f'{val:.2f}'==f'{p[name][j]:.2f}'))
    f=pd.DataFrame(rows); f.to_csv(O/'source_gate.csv',index=False)
    result=dict(n_cells=len(f),source_values_match_printed=bool(f.rounded_match.all()),fill_order_definition_matches_all_columns=False,reason='All 26 tissue cells use continuous AUROC in saved sources; caption says fill order.',mismatches=f[~f.rounded_match].to_dict('records'))
    (O/'gate.json').write_text(json.dumps(result,indent=2)+'\n'); print(result,flush=True)
def tied(y,s,bins):
    numer=denom=0
    for b in np.unique(bins):
        ix=bins==b; yy=y[ix]; ss=s[ix]
        for value in np.unique(ss): numer+=int(np.sum(yy[ss==value]))*int(np.sum(1-yy[ss==value]))
        denom+=int(yy.sum())*int((1-yy).sum())
    return numer/denom if denom else np.nan

def screen(dataset):
    import anndata as ad
    args=argparse.Namespace(deploy_root=str(VIEW/'deployment'),review_root=str(VIEW/'review'),norman_root=str(VIEW/'norman'),norman_benchmark=str(E/'review_round2/leakage_free/norman_crispra'))
    inputs=kd.screen_inputs(dataset,args); data=kd.load_screen(dataset,inputs.deploy,inputs.splits,inputs.prepared)
    targets=kd.eligible_targets(data,.2); library=data.recorded.sum(axis=1,dtype=np.float64)
    columns=np.array([t.g for t in targets]); methods=kd.build_methods(data,columns,inputs,1729)
    for name in ['DCA','scImpute','EnImpute']:
        path=E/'review_round3/comparators/fits'/name.lower()/f'deploy_{dataset}'
        value=kd.count_scale_contract(path,data,library)
        methods[name]=ranked_method(name,value,value,data,columns,1729)
    hybrid=ad.read_h5ad(inputs.deploy/'hybrid.h5ad')
    probability,_=lb.expected_detection(hybrid.layers['corrupted_counts'],data.split=='development',data.target,library)
    methods['label_detection']=lb.teacher_ranking('label_detection',probability,data)
    rows=[]
    for item in targets:
        y=kd.likely_dropout_labels(data,item); loglib=np.log(library[item.cells])
        bins=np.searchsorted(np.quantile(loglib,np.linspace(0,1,6)[1:-1]),loglib,side='right')
        for name,key in NAMES.items():
            scores=kd.target_scores(methods[key],item,library)
            fill=scores['fill_order']
            for adj,bs in [('none',np.zeros(len(y),int)),('depth_strata',bins)]:
                for kind in ['fill_order','continuous']:
                    score=scores[kind]
                    auc=kd.auroc(y,score) if adj=='none' else kd.stratified_auroc(y,score,loglib,5)[0]
                    rows.append(dict(dataset=dataset,gene=item.gene,method=name,adjustment=adj,score_type=kind,auroc=auc,tied_pair_share=tied(y,fill,bs),unfilled_zero_share=float((fill==0).mean())))
    f=pd.DataFrame(rows); f.to_csv(O/f'{dataset}_per_target.csv',index=False)
    genes=[t.gene for t in targets]
    draws=np.random.default_rng([1729,kd.SCOPES.index(dataset)]).integers(0,len(genes),size=(2000,len(genes)))
    np.save(O/f'{dataset}_draws.npy',draws)
    summary=[]
    for (name,adj,kind),part in f.groupby(['method','adjustment','score_type'],sort=False):
        v=part.set_index('gene').loc[genes]
        sample=np.nanmean(v.auroc.to_numpy()[draws],axis=1); low,high=np.nanquantile(sample,[.025,.975])
        summary.append(dict(dataset=dataset,method=name,adjustment=adj,score_type=kind,estimate=v.auroc.mean(),lower=low,upper=high,tied_pair_share=v.tied_pair_share.mean(),unfilled_zero_share=v.unfilled_zero_share.mean(),n_units=len(genes),fill_grid='1,2,3,4,5,6,7,8,9,10'))
    pd.DataFrame(summary).to_csv(O/f'{dataset}_summary.csv',index=False)
    print(dataset,'done',flush=True)

def tissue(tissue):
    import anndata as ad
    t=pd.read_parquet(LABEL/'sex'/tissue/'transductive_expected_count_scores.parquet')
    t=t[t.gene=='RPS4Y1'].copy().reset_index(drop=True); t.donor=t.donor.astype(str)
    mapping={n:n for n in NAMES}
    mapping.update({'Safe Fusion':'safe_fusion','Safe Fusion, labels':'safe_fusion_donor','Weighted kNN, labels':'graph_smooth_donor','Per-label expected count':'transductive_expected_count'})
    fractions=[.01,.05,.10]
    for unit,ix in t.groupby('unit').groups.items():
        base=R/'transductive_references/sex_zeros/deployment_rebuilt'/unit
        rr=t.loc[ix,'cell_index'].to_numpy(); cc=t.loc[ix,'gene_index'].to_numpy()
        for method,folder in [('safe_fusion','selector'),('safe_fusion_donor','selector_donor')]:
            for fraction in fractions:
                suffix=f'{fraction:g}'.replace('.','p')
                v=np.load(base/folder/f'safe_fusion_calibrated_mlp_topk_{suffix}/mean.npy',mmap_mode='r')
                t.loc[ix,f'filled:{method}:{fraction}']=v[rr,cc]>0
        a=ad.read_h5ad(base/'hybrid.h5ad'); counts=dense(a.layers['corrupted_counts']).astype(np.float64)
        split=pd.read_parquet(base/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy()
        zr,zc=np.where((counts==0)&(split=='test')[:,None]); flat=zr*counts.shape[1]+zc
        pos=np.searchsorted(flat,rr*counts.shape[1]+cc); np.testing.assert_array_equal(flat[pos],rr*counts.shape[1]+cc)
        donor=a.obs.donor.astype(str).to_numpy(); lib=counts.sum(1)
        expected=np.empty(len(zr),np.float32)
        for label in np.unique(donor[zr]):
            total=counts[donor==label].sum(0); take=donor[zr]==label
            expected[take]=(-np.expm1(-lib[zr[take]]*(total/total.sum())[zc[take]])).astype(np.float32)
        v=np.load(base/'graph_smooth_donor/mean.npy',mmap_mode='r')
        for method,values in [('graph_smooth_donor',np.maximum(v[zr,zc],0)),('transductive_expected_count',expected)]:
            np.testing.assert_array_equal(values[pos],t.loc[ix,f'score:{method}'])
            for fraction in fractions:
                filled=exact_topk(values,max(1,round(fraction*len(values))))&(values>0)
                t.loc[ix,f'filled:{method}:{fraction}']=filled[pos]
        del counts,a,expected; gc.collect()
    d=pd.read_csv(LABEL/'sex'/tissue/'donor_sex.csv',dtype={'donor':str}); donors=sorted(d.donor)
    draws=stratified_draws(d.set_index('donor').loc[donors,'sex'].to_numpy(),2000,1729)
    np.save(O/f'{tissue}_draws.npy',draws)
    weights=np.vstack([np.ones(len(donors)),[np.bincount(x,minlength=len(donors)) for x in draws]])
    codes=pd.Categorical(t.donor,categories=donors).codes; y=(t.kind=='recorded zero, expressing sex').to_numpy().astype(int); bins=np.zeros(len(t),int)
    rows=[]
    for name,method in mapping.items():
        fill=np.zeros(len(t))
        for fraction in reversed(fractions): fill[t[f'filled:{method}:{fraction}'].to_numpy(bool)]=11-100*fraction
        t[f'fill_order:{method}']=fill
        for kind,scores in [('fill_order',fill),('continuous',t[f'score:{method}'].to_numpy())]:
            assert np.isfinite(scores).all(),(name,tissue)
            values=sex.bootstrap_auc(y,scores,codes,weights,bins)
            np.testing.assert_allclose(values[0],kd.auroc(y,scores),atol=1e-12,rtol=0)
            low,high=np.nanquantile(values[1:],[.025,.975])
            rows.append(dict(dataset=tissue,method=name,adjustment='none',score_type=kind,estimate=values[0],lower=low,upper=high,tied_pair_share=tied(y,fill,bins),unfilled_zero_share=float((fill==0).mean()),n_units=len(donors),fill_grid='1,5,10'))
    t.to_parquet(O/f'{tissue}_zero_scores.parquet',index=False)
    pd.DataFrame(rows).to_csv(O/f'{tissue}_summary.csv',index=False)
    print(tissue,'done',flush=True)
if __name__=='__main__':
    assert tied(np.array([1,0,1,0]),np.array([0,0,1,2]),np.zeros(4))==.25
    action=sys.argv[1]
    if action=='gate': gate()
    elif action in ['adamson_crispri','norman_crispra']: screen(action)
    else: tissue(action)
