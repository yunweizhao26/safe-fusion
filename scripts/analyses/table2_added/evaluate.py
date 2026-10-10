from common import *
import re, os
import disease_control_sex_zeros as zeros
MODE=sys.argv[1]
for source,expected_hash in json.loads((O/'source_sha256.json').read_text()).items():
    assert sha(source)==expected_hash,(source,'source changed after audit')
METHODS=['SAVER'] if MODE=='reference' else ['scRecover','scGPT']
rows=[];boot={};checks=[];score_audit=[]

def summarize(dataset,method,adjustment,point,samples,n_units,n_positive,n_negative):
    assert samples.shape==(2000,) and np.isfinite(samples).all()
    low,high=np.quantile(samples,[.025,.975]); se=np.std(samples,ddof=1)
    rows.append(dict(dataset=dataset,method=method,adjustment=adjustment,estimate=float(point),se=float(se),lower=float(low),upper=float(high),n_units=n_units,n_positive=n_positive,n_negative=n_negative,draws=2000,seed=1729))
    boot[f'{dataset}|{method}|{adjustment}']=samples

def check(name,actual,expected,tolerance=1e-12):
    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)
    checks.append(dict(check=name,passed=True,max_abs_error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))))

for dataset in UNITS[:2]:
    inputs=kd.screen_inputs(dataset,ARGS); screen=kd.load_screen(dataset,inputs.deploy,inputs.splits,inputs.prepared)
    a=ad.read_h5ad(inputs.deploy/'hybrid.h5ad'); lib=screen.recorded.sum(1,dtype=np.float64)
    targets=kd.eligible_targets(screen,.2); genes=[t.gene for t in targets]
    saved=pd.read_csv(A/f'{dataset}_per_target.csv'); saved=saved[saved.score_type=='continuous']
    assert genes==saved.gene.drop_duplicates().tolist()
    draws=np.random.default_rng([1729,kd.SCOPES.index(dataset)]).integers(0,len(genes),size=(2000,len(genes)))
    np.testing.assert_array_equal(draws,np.load(A/f'{dataset}_draws.npy'))
    output=[]; zs=[]
    auditzeros=pd.read_parquet(O/'audit'/f'{dataset}_zeros.parquet')
    for method in METHODS:
        path=contract(dataset,method)
        metadata=json.loads((path/'metadata.json').read_text())
        if method=='scRecover' and not path.is_relative_to(O):
            assert sha(path/'mean.npy')==json.loads((O/'audit'/f'{dataset}.json').read_text())['legacy_mean_sha256']
        if path.is_relative_to(O):
            assert metadata['input_sha256']==sha(inputs.deploy/'hybrid.h5ad')==json.loads((O/'audit'/f'{dataset}.json').read_text())['input_sha256']
        for t in targets:
            y=kd.likely_dropout_labels(screen,t); rr=t.cells; cc=np.full(len(rr),t.g)
            z=auditzeros[auditzeros.gene==t.gene]
            np.testing.assert_array_equal(rr,z.cell_index);np.testing.assert_array_equal(y,z.positive)
            score=get_scores(path,a,rr,cc,lib,method)
            auc=kd.auroc(y,score); strata=kd.stratified_auroc(y,score,np.log(lib[rr]),5)[0]
            output.append(dict(dataset=dataset,method=method,gene=t.gene,none=auc,depth_strata=strata,n_positive=int(y.sum()),n_negative=int((1-y).sum())))
            zs.extend(dict(method=method,gene=t.gene,cell_id=screen.cell_ids[r],cell_index=int(r),gene_index=t.g,positive=int(l),score=float(s),library_size=float(lib[r])) for r,l,s in zip(rr,y,score))
        score_audit.append(dict(unit=dataset,method=method,contract=str(path),mean_sha256=sha(path/'mean.npy'),scale=metadata['scale']))
    frame=pd.DataFrame(output); frame.to_csv(O/'evaluation'/f'{MODE}_{dataset}_per_target.csv',index=False)
    pd.DataFrame(zs).to_parquet(O/'evaluation'/f'{MODE}_{dataset}_zero_scores.parquet',index=False)
    for method in METHODS:
        f=frame[frame.method==method].set_index('gene').loc[genes]
        for adj in ['none','depth_strata']:
            values=f[adj].to_numpy(); summarize(dataset,method,adj,values.mean(),values[draws].mean(1),len(genes),int(f.n_positive.sum()),int(f.n_negative.sum()))
            if MODE=='reference':
                old=saved[(saved.method==method)&(saved.adjustment==adj)].set_index('gene').loc[genes].auroc.to_numpy()
                check(f'{dataset} SAVER per-target {adj}',values,old)
            elif method=='scRecover' and json.loads((O/'audit'/f'{dataset}.json').read_text())['reuse']:
                old=pd.read_csv(R3/'evaluation/known_zeros/knockdown/auroc.csv')
                old=old[(old.dataset==dataset)&(old.method==method)&(old.score_type=='continuous')].set_index('gene').loc[genes,adj].to_numpy()
                check(f'{dataset} legacy scRecover per-target {adj}',values,old)
    print(dataset,'evaluated',flush=True)

for tissue in ['pancreas','colon']:
    saved=pd.read_parquet(A/f'{tissue}_zero_scores.parquet'); saved=saved[saved.gene=='RPS4Y1'].copy().reset_index(drop=True)
    saved['donor']=saved.donor.astype(str)
    donors_frame=pd.read_csv(R4/'label_baselines/sex'/tissue/'donor_sex.csv',dtype={'donor':str})
    donors=sorted(donors_frame.donor); sexmap=donors_frame.set_index('donor').sex
    assert len(donors)==(24 if tissue=='pancreas' else 34)

    collected=[]
    for unit in [f'{tissue}_{i}' for i in range(3)]:
        b=base(unit); a=ad.read_h5ad(b/'hybrid.h5ad'); recorded=ad.read_h5ad(b/'recorded.h5ad')
        assert list(a.obs_names)==list(recorded.obs_names) and list(a.var_names)==list(recorded.var_names)
        counts=dense(recorded.layers['corrupted_counts']); x=dense(a.layers['corrupted_counts'])
        splits=pd.read_parquet(b/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy()
        test=np.flatnonzero(splits=='test'); np.testing.assert_array_equal(x[test],counts[test])
        obs=recorded.obs.copy().reset_index(drop=True);obs['donor']=obs.donor.astype(str);obs['sex']=obs.donor.map(sexmap)
        assert obs.iloc[test].sex.notna().all()
        g=list(recorded.var.feature_name.astype(str)).index('RPS4Y1')
        original=zeros.SEX_GENES
        try:
            zeros.SEX_GENES={'RPS4Y1':'male'}
            t=zeros.zero_table(obs.iloc[test].reset_index(drop=True),counts[test],{'RPS4Y1':g},None)
        finally: zeros.SEX_GENES=original
        rr=test[t.row.to_numpy()]; cc=t.col.to_numpy(); lib=x.sum(1,dtype=np.float64)
        t=t.assign(unit=unit,cell_index=rr,gene_index=cc)
        for method in METHODS:
            path=contract(unit,method); m=json.loads((path/'metadata.json').read_text())
            if path.is_relative_to(O):
                assert m['input_sha256']==sha(b/'hybrid.h5ad')==json.loads((O/'audit'/f'{unit}.json').read_text())['input_sha256']
            t[f'score:{method}']=get_scores(path,a,rr,cc,lib,method)
            score_audit.append(dict(unit=unit,method=method,contract=str(path),mean_sha256=sha(path/'mean.npy'),scale=m['scale']))
        collected.append(t)
    t=pd.concat(collected,ignore_index=True);t.donor=t.donor.astype(str)
    keys=['unit','cell_index','gene_index']
    t=t.set_index(keys).loc[pd.MultiIndex.from_frame(saved[keys])].reset_index()
    assert len(t)==len(saved)
    for col in ['donor','kind','gene','library_size']:
        np.testing.assert_array_equal(t[col],saved[col])
    y=(t.kind=='recorded zero, expressing sex').to_numpy().astype(int)
    assert (int(y.sum()),int((1-y).sum()))==((372,1449) if tissue=='pancreas' else (1670,1721))
    draws=stratified_draws(sexmap.loc[donors].to_numpy(),2000,1729)
    np.testing.assert_array_equal(draws,np.load(A/f'{tissue}_draws.npy'))
    weights=np.vstack([np.ones(len(donors)),[np.bincount(d,minlength=len(donors)) for d in draws]])
    codes=pd.Categorical(t.donor,categories=donors).codes
    assert (codes>=0).all()
    for method in METHODS:
        score=t[f'score:{method}'].to_numpy()
        values=sex.bootstrap_auc(y,score,codes,weights,np.zeros(len(t),int))
        check(f'{tissue} {method} observed weighted AUROC',values[0],kd.auroc(y,score))
        summarize(tissue,method,'none',values[0],values[1:],len(donors),int(y.sum()),int((1-y).sum()))
        if MODE=='reference': check(f'{tissue} SAVER all zero scores',score,saved['score:SAVER'].to_numpy())
        elif method=='scRecover' and all(json.loads((O/'audit'/f'{tissue}_{i}.json').read_text())['reuse'] for i in range(3)):

            from disease_control_common import load_heldout, OUTPUT
            legacy=load_heldout(tissue)
            oldsex=pd.read_csv(OUTPUT/'sex_zeros'/tissue/'donor_sex.csv',dtype={'donor':str}).set_index('donor').sex
            pd.testing.assert_series_equal(sexmap.sort_index(),oldsex.loc[donors].sort_index(),check_names=False)
            legacy_ids=legacy.obs.index.astype(str).to_numpy() if 'cell_id' not in legacy.obs else legacy.obs.cell_id.astype(str).to_numpy()

            old=pd.read_csv(R3/'evaluation/known_zeros/sex_zeros'/tissue/'auroc.csv')
            old=old[(old.gene=='RPS4Y1')&(old.method=='scRecover')].iloc[0]
            assert int(old.n_positive)==int(y.sum()) and int(old.n_negative)==int((1-y).sum())
            for v,k in zip([values[0],*np.quantile(values[1:],[.025,.975])],['auroc','ci_low','ci_high']):
                assert f'{v:.4f}'==f'{old[k]:.4f}',(tissue,k,v,old[k])
            checks.append(dict(check=f'{tissue} legacy scRecover labels, counts, point and interval',passed=True))
    t.to_parquet(O/'evaluation'/f'{MODE}_{tissue}_zero_scores.parquet',index=False)
    print(tissue,'evaluated',flush=True)

result=pd.DataFrame(rows)
if MODE=='reference':
    old=pd.read_csv(R4/'main_tables_v3/estimates.csv')
    old=old[(old.table=='Table 2')&(old.method=='SAVER')].set_index('column')
    paper=(Path(os.environ['SAFE_FUSION_MANUSCRIPT_DIR'])/'supplementary_results.tex').read_text().split('\\label{tab:s_knockdown_full}',1)[1].split('\\bottomrule',1)[0]
    cells=next(line for line in paper.splitlines() if line.startswith('SAVER &')).split(' & ')[1:]
    endpoints=[('adamson_crispri','none','CRISPRi'),('adamson_crispri','depth_strata',None),('norman_crispra','none','CRISPRa'),('pancreas','none','Pancreas sex'),('colon','none','Colon sex')]
    for i,(dataset,adj,col) in enumerate(endpoints):
        r=result[(result.dataset==dataset)&(result.adjustment==adj)].iloc[0]
        printed=re.findall(r'\d+\.\d+',cells[i]); actual=[f'{r[k]:.2f}' for k in ['estimate','lower','upper']]
        assert actual==printed,(dataset,adj,actual,printed)
        checks.append(dict(check=f'S17 SAVER {dataset} {adj}',actual=actual,expected=printed,passed=True))
        if col:
            check(f'Table 2 SAVER {col} estimate and SE',[r.estimate,r.se],old.loc[col,['mean','bootstrap_se']].to_numpy(float))
            check(f'Table 2 SAVER {col} all bootstrap values',boot[f'{dataset}|SAVER|none'],np.load(R4/'main_tables_v3/bootstrap_quantities.npz')[f'Table 2|SAVER|{col}'])
result.to_csv(O/'evaluation'/f'{MODE}_estimates.csv',index=False)
np.savez(O/'evaluation'/f'{MODE}_bootstrap_quantities.npz',**boot)
dump(O/'evaluation'/f'{MODE}_checks.json',checks)
dump(O/'evaluation'/f'{MODE}_score_audit.json',score_audit)
print(result.to_string(index=False),flush=True)
