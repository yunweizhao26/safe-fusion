from common import *
import gc
checks=[]
for unit in UNITS:
    b=base(unit); a=ad.read_h5ad(b/'hybrid.h5ad')
    split=pd.read_parquet(b/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy()
    x=dense(a.layers['corrupted_counts'])
    record=dict(unit=unit,input=str(b/'hybrid.h5ad'),input_sha256=sha(b/'hybrid.h5ad'),coordinates_sha256=sha(b/'coordinates.parquet'),splits_sha256=sha(b/'splits.parquet'),n_cells=a.n_obs,n_genes=a.n_vars,n_test=int((split=='test').sum()),reuse=False)
    legacy=R3/'fits/screcover'/f'deploy_{unit}'
    if not unit.startswith('colon'):
        old=legacy_base(unit); m=json.loads((legacy/'metadata.json').read_text()); old_a=ad.read_h5ad(old/'hybrid.h5ad')
        old_split=pd.read_parquet(old/'splits.parquet').set_index('cell_id').loc[old_a.obs_names,'split'].to_numpy()
        conditions=dict(legacy_file_hash=m['input_sha256']==sha(old/'hybrid.h5ad'),cell_order=m['cell_ids']==list(a.obs_names)==list(old_a.obs_names),gene_order=m['gene_ids']==list(a.var_names)==list(old_a.var_names),matrix=np.array_equal(x,dense(old_a.layers['corrupted_counts'])),splits=np.array_equal(split,old_split),count_scale=m['scale']=='counts',seed=m['seed']==1729,all_cells=m['parameters']['fit_cells']==a.n_obs)
        record.update(legacy=str(legacy),legacy_input=str(old/'hybrid.h5ad'),conditions=conditions,reuse=all(conditions.values()),legacy_parameters=m['parameters'],legacy_mean_sha256=sha(legacy/'mean.npy'))
        del old_a
    else: record['reason']='Mandatory refit on current 34-donor cohort; no smaller-cohort reuse.'
    if unit in UNITS[:2]:
        inputs=kd.screen_inputs(unit,ARGS); screen=kd.load_screen(unit,inputs.deploy,inputs.splits,inputs.prepared)
        np.testing.assert_array_equal(x[screen.test],screen.recorded[screen.test])
        targets=kd.eligible_targets(screen,.2)
        oldinputs=kd.screen_inputs(unit,argparse.Namespace(deploy_root=str(R4.parent/'downstream_deployment'),review_root=str(R4.parent/'review_round2/knockdown'),norman_root=str(R4.parent/'review_round2/knockdown/norman_rebuilt'),norman_benchmark=ARGS.norman_benchmark))
        oldscreen=kd.load_screen(unit,oldinputs.deploy,oldinputs.splits,oldinputs.prepared)
        oldtargets=kd.eligible_targets(oldscreen,.2)
        assert len(targets)==len(oldtargets)==(26 if unit=='adamson_crispri' else 41)
        zeros=[]
        for t,old_t in zip(targets,oldtargets):
            assert t.gene==old_t.gene and t.g==old_t.g
            np.testing.assert_array_equal(t.cells,old_t.cells)
            y=kd.likely_dropout_labels(screen,t); np.testing.assert_array_equal(y,kd.likely_dropout_labels(oldscreen,old_t))
            for row,label in zip(t.cells,y): zeros.append(dict(unit=unit,gene=t.gene,cell_index=int(row),gene_index=int(t.g),cell_id=screen.cell_ids[row],positive=int(label),library_size=float(screen.recorded[row].sum(dtype=np.float64))))
        pd.DataFrame(zeros).to_parquet(O/'audit'/f'{unit}_zeros.parquet',index=False)
        record['target_count']=len(targets);record['zero_sets_identical']=True
        del screen,oldscreen
    else:
        tissue=unit.rsplit('_',1)[0]
        t=pd.read_parquet(A/f'{tissue}_zero_scores.parquet'); t=t[(t.unit==unit)&(t.gene=='RPS4Y1')].copy()
        rr=t.cell_index.to_numpy(); cc=t.gene_index.to_numpy()
        assert (split[rr]=='test').all() and (x[rr,cc]==0).all()
        assert list(a.var.feature_name.astype(str))[int(cc[0])]=='RPS4Y1'
        np.testing.assert_array_equal(x.sum(1,dtype=np.float64)[rr],t.library_size)
        if 'cell_id' in t: np.testing.assert_array_equal(a.obs_names.to_numpy()[rr],t.cell_id)
        np.testing.assert_array_equal(a.obs.donor.astype(str).to_numpy()[rr],t.donor.astype(str))
        record['n_positive']=int((t.kind=='recorded zero, expressing sex').sum());record['n_negative']=int((t.kind=='absent, other sex').sum())
        if unit.startswith('pancreas') and record['reuse']:
            from disease_control_common import OUTPUT
            oldsex=pd.read_csv(OUTPUT/'sex_zeros/pancreas/donor_sex.csv',dtype={'donor':str}).set_index('donor').sex
            old_labels=(oldsex.loc[t.donor.astype(str)].to_numpy()=='male').astype(int)
            new_labels=(t.kind=='recorded zero, expressing sex').to_numpy().astype(int)
            old_rows=np.flatnonzero((x[:,int(cc[0])]==0)&(old_split=='test'))
            conditions['zero_rows']=np.array_equal(old_rows,rr)
            conditions['sex_labels']=np.array_equal(old_labels,new_labels)
            record['reuse']=all(conditions.values())
            record['zero_sets_identical']=record['reuse']
    dump(O/'audit'/f'{unit}.json',record);checks.append(record)
    print(unit,record['n_cells'],record['n_genes'],'reuse',record['reuse'],flush=True)
    del a,x; gc.collect()
dump(O/'audit/inputs.json',checks)

manifest=json.loads((R3/'units_manifest.json').read_text())

paths=set()
def walk(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k=='scGPT': paths.add(v)
            else: walk(v)
    elif isinstance(obj,list):
        for v in obj: walk(v)
walk(manifest)
walk(json.loads((R3/'colon_crossfit_units.json').read_text()))
ckpt=sha(ROOT/'external_data/baselines/scgpt_human/best_model.pt')
settings=[]
for p in sorted(paths):
    m=json.loads((Path(p)/'metadata.json').read_text())
    assert m['parameters']['checkpoint_sha256']==ckpt and m['seed']==1729
    assert m['parameters']['max_observed_sequence_length']==1200
    settings.append(dict(contract=p,parameters=m['parameters'],seed=m['seed']))
dump(O/'audit/table1_scgpt.json',settings)

dump(O/'audit/checkpoint_sha256.json',{name:sha(ROOT/'external_data/baselines/scgpt_human'/name) for name in ['best_model.pt','args.json','vocab.json']})
