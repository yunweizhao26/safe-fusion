from tables import *
from selector_attribution import exact_topk
import v2_value_evaluate as vv

def rule(index):
    import time
    while not all((O/'original/rule_rebuilt'/kind/'norman_crispra/calibration_report.json').exists() for kind in ['masked','recorded']):
        time.sleep(600)
    run(O/"code/rule_check.py")
    uu=units()
    if index<14:
        u=uu[index%7]; kind='masked' if index<7 else 'recorded'
        if kind=='masked':
            inp,coords=u.corrupted,u.coordinates; models=model_view(u)
        else:
            src=T/'recorded'/u.key
            inp,coords=src/'input/hybrid.h5ad',src/'input/coordinates.parquet'
            models=src/'transductive'
        dest=O/'current/rule'/kind/u.key
        names=TEACHERS if kind=='masked' else ['gene_median','svd_impute','graph_smooth','magic','scvi']
    else:
        u=next(x for x in uu if x.key==f'colon_{index-14}')
        key=f'colon_thinning_025_{index-14}'
        inp=E/'review_round3/colon_crossfit/thinning/mask_trained'/key/'input/hybrid.h5ad'
        coords=inp.parent/'coordinates.parquet'
        models=T/'thinning'/key/'mask-trained'
        dest=O/'current/rule/thinning'/key
        names=['gene_median','svd_impute','graph_smooth','magic','scvi']
    args=sum((['--teacher-contract',models/n] for n in names),[])
    run(O/'code/selector.py','--corrupted',inp,'--truth',u.truth,'--coordinates',coords,'--splits',u.splits,'--fusion-contract',models/'safe_fusion',*args,'--output-dir',dest,'--fit-split',u.fit_split,'--architecture','mlp','--budget-mode','apply_topk','--budgets',.01,.05,.10,'--curve-min-budget',.001,'--curve-max-budget',1.,'--curve-points',1000,'--detection-rule-mask-rate',.10,'--seed',1729)
    if index<7:
        saved=np.load(fusion_root(u)/'selectors'/u.key/'safe_fusion_transductive/test_scores.npy')
        new=np.load(dest/'detection.npz')['scores']
        assert np.array_equal(saved,new),(u.key,np.max(abs(saved-new)))
        (dest/'score_parity.json').write_text(json.dumps({'n_scores':len(new),'exact':True})+'\n')

def recorded():
    assert (O/'original/recorded/parity.json').exists()
    deployments={}
    for u in units():
        label={'Pancreas':'Pancreas','Colon':'Colon','CRISPRa':'CRISPRa (Norman)'}[u.dataset]
        root=T/'recorded'/u.key
        deployments.setdefault(label,[]).append((u.key,root/'input/recorded.h5ad',u.splits,root/'transductive',root/'transductive/selector'))
    z=D/'deployment/zebrafish'
    deployments['Zebrafish']=[('zebrafish',z/'recorded.h5ad',z/'splits.parquet',z/'transductive',z/'transductive/selector')]
    for key,label in [('adamson_crispri','CRISPRi (Adamson)'),('papalexi_eccite','ECCITE-seq (Papalexi)')]:
        root=E/'review_round4/transductive_references/knockdown/deployment'/key
        deployments[label]=[(key,root/'recorded.h5ad',root/'splits.parquet',root/'methods',root/'selector')]

    iv.DEPLOYMENT={}; iv.FILLS={k:v for k,v in iv.FILLS.items() if k!='Weighted kNN'}
    for label,parts in deployments.items():
        iv.DEPLOYMENT[label]=[]
        for key,inp,split,models,selector in parts:
            dest=O/'recorded_views'/key
            link(inp,dest/'recorded.h5ad'); link(split,dest/'splits.parquet'); link(selector,dest/'selector')
            for name in TEACHERS:
                source=models/name
                if not source.exists(): source=models/{'magic_inductive':'magic','scvi_inductive':'scvi'}.get(name,name)
                link(source,dest/'methods'/name)
            run('scripts/apply_fill_fraction.py','--corrupted',inp,'--splits',split,'--method-contract',dest/'methods/svd_impute','--method-name','svd','--output-root',dest/'matched','--fractions',.01,.05,.10)
            iv.DEPLOYMENT[label].append(dest)
    result=iv.recorded_zero_fills()
    dest=O/'current/recorded';dest.mkdir(parents=True,exist_ok=True)
    result.to_csv(dest/'recorded_zero_fills.csv',index=False)
    (dest/'inputs.json').write_text(json.dumps({k:[str(x) for x in v] for k,v in iv.DEPLOYMENT.items()},indent=2)+'\n')

def old_detection():
    dest=O/'original/detection_weighted';dest.mkdir(parents=True,exist_ok=True)
    run('scripts/v2_value_evaluate.py','--values','conditional','conditional_detection','--output-dir',dest,'--draws',2000,'--seed',1729)
    a=pd.read_csv(dest/'value_accuracy.csv');b=pd.read_csv(E/'review_round3/value_v2_ablations/value/test/value_accuracy.csv')
    b=b[b.value.isin(a.value.unique())]
    reference=dest/'reference_subset.csv';b.to_csv(reference,index=False)
    parity(dest/'value_accuracy.csv',reference,['benchmark','dataset','statistic','value'],dest/'parity.json')

def thinning():
    assert (O/'original/thinning/parity.json').exists()
    assert (O/'original/detection_weighted/parity.json').exists()
    frames={}
    for u,key,root,p in specs():
        models=O/'models/thinning'/key
        inp=root/'mask_trained'/key/'input'
        data=ad.read_h5ad(inp/'hybrid.h5ad')
        split=pd.read_parquet(u.splits).set_index('cell_id').loc[data.obs_names.astype(str),'split'].to_numpy()
        coords=pd.read_parquet(inp/'coordinates.parquet')
        rr,cc=coords.cell_index.to_numpy(),coords.gene_index.to_numpy(); keep=split[rr]=='test'
        rr,cc=rr[keep],cc[keep];x=coords.original_value.to_numpy()[keep].astype(float)
        assert np.all(iv.dense(data.layers['corrupted_counts'])[rr,cc]==0) and np.all(x>0)
        groups=data.obs[u.unit_column].astype(str).to_numpy()[rr]
        groups=np.char.add(key+':',groups.astype(str))
        selector=T/'thinning'/key/'mask-trained/selector'
        frame=pd.DataFrame(dict(group=groups,count=x,expected_count=p*x,selected=vv.filled(selector,.05,rr,cc)))
        for value in ['safe_fusion','safe_fusion_linear','scvi_inductive']:
            frame['value:'+value]=iv.at(models/value/'mean.npy',rr,cc)
        calibration=(O/'current/rule/thinning'/key if p==.25 else D/'thinning'/u.key/'transductive/selector')
        detection=np.load(calibration/'detection.npz')

        codes=detection['rows']*data.n_vars+detection['cols']; pos=np.searchsorted(codes,rr*data.n_vars+cc)
        assert np.array_equal(codes[pos],rr*data.n_vars+cc)
        selected=vv.filled(calibration,.05,rr,cc)
        assert np.array_equal(selected,frame.selected.to_numpy()),key
        if p==.25:
            from calibrated_selective_fill import detection_probability
            weight=detection_probability(detection['probability'].astype(np.float32),.1).astype(np.float32)[pos]
        else:
            weight=detection['detection'][pos]

        frame['value:conditional_detection']=(frame['value:safe_fusion'].to_numpy().astype(np.float32)*weight.astype(np.float32)).astype(np.float64)
        frames.setdefault(f'{u.dataset}, {int(100*p)}%',[]).append(frame)

    rows=[]
    for dataset,parts in frames.items():
        f=pd.concat(parts,ignore_index=True); groups=sorted(f.group.unique())
        draws=np.random.default_rng(1729).integers(0,len(groups),size=(2000,len(groups)))
        for scope in ['all positives','filled at 5%']:
            keep=np.ones(len(f),bool) if scope=='all positives' else f.selected.to_numpy()
            codes=pd.Categorical(f.group[keep],categories=groups).codes
            n=np.bincount(codes,minlength=len(groups)).astype(float);den=n[draws].sum(1)
            for value in ['safe_fusion','safe_fusion_linear','scvi_inductive','conditional_detection']:
                error=np.log1p(np.maximum(f['value:'+value].to_numpy()[keep],0))-np.log1p(f.expected_count.to_numpy()[keep])
                sums=np.bincount(codes,weights=error,minlength=len(groups)); samples=sums[draws].sum(1)[den>0]/den[den>0]
                lo,hi=np.quantile(samples,[.025,.975])
                rows.append(dict(dataset=dataset,scope=scope,value=value,estimate=error.mean(),lower=lo,upper=hi,groups=len(groups),entries=int(keep.sum())))
    dest=O/'current/thinning';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'bias.csv',index=False)

if __name__=='__main__':
    task=sys.argv[1]
    if task=='rule': rule(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else: globals()[task]()
