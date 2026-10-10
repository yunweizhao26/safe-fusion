import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path.cwd()
B = ROOT / 'artifacts/paper_evidence/review_round4'
O = B / 'protein_depth'
sys.path.insert(0, str(ROOT / 'scripts'))
import selector_attribution as a
import calibrated_selective_fill as p


def config(kind):
    if kind == 'hao':
        base = B / 'protein_multi/corrected'
        return dict(base=base, corrupted=base/'prepared/deploy/hybrid.h5ad', truth=base/'prepared/truth.h5ad',
                    coordinates=base/'prepared/deploy/coordinates.parquet', splits=base/'prepared/deploy/splits.parquet',
                    teachers=base/'teachers', selector=base/'selector', last='scvi',
                    wrapper=ROOT/'scripts/analyses/protein_multi/selector.py')
    if kind == 'papalexi_small':
        base = B / 'transductive_references/protein_cs'
        return dict(base=base, corrupted=base/'corrupted.h5ad', truth=ROOT/'external_data/prepared/papalexi_eccite_crossmodal.h5ad',
                    coordinates=base/'coordinates.parquet', splits=base/'splits.parquet', teachers=base,
                    selector=base/'mlp_selector', last='scvi_counts', wrapper=None)
    base = B/'protein_full/full'
    return dict(base=base, corrupted=base/'corrupted.h5ad', truth=base/'prepared.h5ad',
                coordinates=base/'coordinates.parquet', splits=base/'splits.parquet', teachers=base/'teachers',
                selector=B/'protein_followup/full/selector', last='scvi_counts', wrapper=ROOT/'scripts/analyses/protein_full/selector.py')


class Captured(Exception):
    pass


def reconstruct(kind, cfg):
    dest = O/'inputs'/f'model_{kind}.joblib'
    if dest.exists():
        return joblib.load(dest)
    candidates = [f for f in cfg['selector'].rglob('*') if f.is_file() and f.suffix in ('.joblib','.pkl','.pickle','.pt','.pth')]
    assert not candidates, f'Saved selector exists; must reuse it: {candidates}'
    if cfg['wrapper']:
        spec=importlib.util.spec_from_file_location('original_sampling',cfg['wrapper'])
        wrapper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        a.stratified_fit_indices=wrapper.fit_indices
    saved = {}
    def capture(model, scaler, features, indices, batch_rows):
        saved.update(model=model, scaler=scaler, indices=indices)
        joblib.dump(saved,dest)
        raise Captured()
    a.predict_scores_in_batches=capture
    argv=['prepare.py','--corrupted',str(cfg['corrupted']),'--truth',str(cfg['truth']),
          '--coordinates',str(cfg['coordinates']),'--splits',str(cfg['splits']),
          '--fusion-contract',str(cfg['base']/'safe_fusion'),'--output-dir',str(O/'runtime'/f'preparation_{kind}'),
          '--fit-split','development','--architecture','mlp','--budget-mode','apply_topk','--seed','1729']
    for name in ['gene_median','svd_impute','graph_smooth','magic',cfg['last']]:
        argv.extend(['--teacher-contract',str(cfg['teachers']/name)])
    sys.argv=argv
    try:
        p.main()
    except Captured:
        pass
    else:
        raise AssertionError('Model capture failed')
    report=json.loads((cfg['selector']/'calibration_report.json').read_text())['models']['mlp']
    assert saved['model'].n_iter_ == report['model_iterations'], (saved['model'].n_iter_, report['model_iterations'])
    return saved


def logits(model, x):
    for coef, intercept in zip(model.coefs_[:-1],model.intercepts_[:-1]):
        x=np.maximum(x@coef+intercept,0)
    return (x@model.coefs_[-1]+model.intercepts_[-1]).ravel()


def theta_for(cfg, gene):
    path=cfg['base']/'scvi'

    code="import torch,json,sys,numpy as np; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(json.dumps({'theta':float(np.exp(s['model_state_dict']['px_r'].numpy().astype(float))[list(s['var_names']).index(sys.argv[2])]),'settings':s['attr_dict']['init_params_']['non_kwargs']}))"
    value=json.loads(subprocess.check_output([str(ROOT/'.conda-scvi-current/bin/python'),'-c',code,str(path/'model/model.pt'),gene],text=True))
    assert value['settings']['dispersion']=='gene' and value['settings']['gene_likelihood']=='nb'
    return value['theta']


def prepare(kind):
    started=time.time()
    hypothesis=O/'hypothesis.md'
    assert hypothesis.exists()
    hypothesis_hash=hashlib.sha256(hypothesis.read_bytes()).hexdigest()
    cfg=config(kind)
    fitted=reconstruct(kind,cfg)
    model,scaler=fitted['model'],fitted['scaler']
    data=ad.read_h5ad(cfg['corrupted'])
    layer=data.layers['corrupted_counts']
    counts=p.dense(layer).astype(np.float32)
    split=pd.read_parquet(cfg['splits']).set_index('cell_id').loc[data.obs_names.astype(str),'split'].to_numpy()
    train=split=='development'
    library=counts.sum(axis=1)
    assert np.all(library>0)
    ref=float(np.median(library[train]))
    gene_mean=np.log1p(np.mean(counts[train],axis=0))
    gene_dropout=np.mean(counts[train]<=0,axis=0)
    teacher_names=['gene_median','svd_impute','graph_smooth','magic',cfg['last']]
    teachers=[np.load(cfg['teachers']/name/'mean.npy',mmap_mode='r') for name in teacher_names]
    saved=pd.read_parquet(cfg['selector']/'selected_gene_scores.parquet')
    all_checks=[]

    for gene, block in saved.groupby('gene_id',sort=False):
        rows=data.obs_names.get_indexer(block.cell_id)
        col=data.var_names.get_loc(gene)
        assert np.all(rows>=0)
        values=np.stack([t[rows,col] for t in teachers])
        features=a.selector_features(values,np.full(len(rows),gene_mean[col]),np.full(len(rows),gene_dropout[col]),np.log1p(library[rows]))
        pred=model.predict_proba(scaler.transform(features))[:,1]
        delta=float(np.max(np.abs(pred-block.selector_score.to_numpy())))
        all_checks.append(dict(gene=gene,n=len(rows),max_abs_difference=delta))
        np.testing.assert_allclose(pred,block.selector_score.to_numpy(),atol=1e-6,rtol=0)
    print(kind,'original scores verified',max(x['max_abs_difference'] for x in all_checks),flush=True)
    if kind=='hao':
        pairs=pd.read_csv(cfg['base']/'selection/selected_pairs.csv')
        sources=[(f'hao_{i:03}',str(pair.gene),str(pair.protein),int(pair.n_train_detected),cfg['base']/f'evaluation/inputs/pair_{i:03}.parquet') for i,pair in pairs.iterrows()]
    else:
        src=B/('protein_full/verification/evaluation/pdl1_cells.parquet' if kind=='papalexi_small' else 'protein_followup/full/evaluation/pdl1_cells.parquet')
        truth=ad.read_h5ad(cfg['truth'])
        col=data.var_names.get_loc('CD274')
        truth_counts=truth.layers['counts'] if 'counts' in truth.layers else truth.X
        detected=int((p.dense(truth_counts)[train,col]>0).sum())
        sources=[(kind,'CD274','PDL1',detected,src)]
    metadata={}
    for ident,gene,antibody,detected,src in sources:
        frame=pd.read_parquet(src)
        aliases={'Safe Fusion':'safe_fusion','SVD':'svd','Weighted kNN':'weighted_knn','MAGIC':'magic','scVI':'scvi','scVI P(X>0)':'p_nonzero','Library size':'library_size','SF mu':'fused_value','SF mu/library':'mu_per_library','SF d*mu':'d_times_mu','adt_clr':'protein'}
        frame=frame.rename(columns=aliases)
        rows=data.obs_names.get_indexer(frame.cell_id)
        col=data.var_names.get_loc(gene)
        assert np.all(rows>=0) and np.all(counts[rows,col]==0) and np.all(split[rows]=='test')
        frame['log_library']=np.log1p(library[rows]).astype(float)
        frame['library_count']=library[rows].astype(float)

        np.testing.assert_allclose(frame.library_size,frame.log_library,atol=1e-6,rtol=0)
        if kind=='hao':
            for level in (1,2):
                frame[f'cell_type_l{level}']=data.obs.iloc[rows][f'celltype.l{level}'].astype(str).to_numpy()
        else:
            frame['target_baseline']=frame.target_baseline_clr
            frame['magic']=np.load(cfg['base']/'magic_standard/mean.npy',mmap_mode='r')[rows,col]
            theta=theta_for(cfg,gene)
            normalized=np.load(cfg['base']/'scvi/mean.npy',mmap_mode='r')[rows,col].astype(float)
            frame['p_nonzero']=theta*np.log1p(normalized*library[rows]/1e4/theta)
        if kind != 'papalexi_full':
            mu=np.asarray(np.load(cfg['base']/'safe_fusion/mean.npy',mmap_mode='r')[rows,col],dtype=float)
            frame['fused_value']=mu
            frame['mu_per_library']=mu/library[rows].astype(float)
            frame['d_times_mu']=mu*frame.safe_fusion.to_numpy().astype(float)
        values=np.stack([t[rows,col] for t in teachers])
        features=a.selector_features(values,np.full(len(rows),gene_mean[col]),np.full(len(rows),gene_dropout[col]),np.log1p(library[rows]))
        x=scaler.transform(features)
        original=model.predict_proba(x)[:,1]
        raw_logit=logits(model,x)
        np.testing.assert_allclose(expit(raw_logit),original,atol=1e-7,rtol=0)
        np.testing.assert_allclose(original,frame.safe_fusion,atol=1e-6,rtol=0)
        fixed=a.selector_features(values*(ref/library[rows])[None,:],np.full(len(rows),gene_mean[col]),np.full(len(rows),gene_dropout[col]),np.full(len(rows),np.log1p(ref)))
        np.testing.assert_array_equal(features[:,5:7],fixed[:,5:7])
        frame['standard_depth']=model.predict_proba(scaler.transform(fixed))[:,1]
        design=np.column_stack([np.ones(len(rows)),frame.log_library.to_numpy()])
        residual=raw_logit.astype(float)-design@np.linalg.lstsq(design,raw_logit.astype(float),rcond=None)[0]
        frame['residual_logit']=rankdata(residual)
        frame['original_logit']=raw_logit
        methods=[n for n in ['safe_fusion','fused_value','mu_per_library','d_times_mu','p_nonzero','svd','weighted_knn','magic','scvi','library_size','svd_all_cells','weighted_knn_all_cells','ifng_score','target_baseline','cell_zero_score','standard_depth','residual_logit'] if n in frame]
        assert np.isfinite(frame[methods]).all().all()
        frame.to_parquet(O/'inputs'/f'{ident}.parquet',index=False)
        metadata[ident]=dict(id=ident,gene=gene,antibody=antibody,n_train_detected=detected,methods=methods,source=str(src.relative_to(ROOT)),reference_library=ref,bootstrap='cells_within_donor' if kind=='hao' else 'target_clusters')
    (O/'inputs'/f'metadata_{kind}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    receipt=dict(kind=kind,hypothesis_sha256=hypothesis_hash,original_model_missing=True,reconstruction='unchanged production main and fit_scores with original sampling wrapper; captured model before scoring',reference_library=ref,model_iterations=int(model.n_iter_),original_score_checks=all_checks,tolerance=1e-6,seconds=time.time()-started,job_id=os.environ.get('SLURM_JOB_ID'),files=list(metadata))
    (O/'runtime'/f'preparation_{kind}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


def audit(kind):

    from scipy.stats import spearmanr
    cfg=config(kind)
    receipt_path=O/'runtime'/f'preparation_{kind}.json'
    receipt=json.loads(receipt_path.read_text())
    source_paths=[Path(__file__),ROOT/'scripts/selector_attribution.py',ROOT/'scripts/calibrated_selective_fill.py',
                  ROOT/'scripts/nonzero_probability.py',cfg['corrupted'],cfg['truth'],cfg['coordinates'],cfg['splits'],
                  cfg['selector']/'selected_gene_scores.parquet',cfg['selector']/'calibration_report.json',
                  O/'inputs'/f'model_{kind}.joblib']
    if cfg['wrapper']:
        source_paths.append(cfg['wrapper'])
    import h5py
    with h5py.File(cfg['corrupted'], 'r') as handle:
        obs_index=handle['obs'].attrs['_index']
        var_index=handle['var'].attrs['_index']
        cells=handle['obs'][obs_index].asstr()[:].tolist()
        genes=handle['var'][var_index].asstr()[:].tolist()
    contracts=[]
    for name in ['gene_median','svd_impute','graph_smooth','magic',cfg['last']]:
        directory=cfg['teachers']/name
        meta=json.loads((directory/'metadata.json').read_text())
        assert meta['scale']=='counts', (directory,meta['scale'])
        assert meta['cell_ids']==cells and meta['gene_ids']==genes, directory
        contracts.append(dict(path=str(directory.relative_to(ROOT)),scale=meta['scale']))
        source_paths.extend([directory/'metadata.json',directory/'mean.npy'])
    if kind!='hao':
        scvi_meta=json.loads((cfg['base']/'scvi/metadata.json').read_text())
        assert scvi_meta['scale']=='normalized_expression_1e4'
        assert scvi_meta['cell_ids']==cells and scvi_meta['gene_ids']==genes
        source_paths.extend([cfg['base']/'scvi/model/model.pt',cfg['base']/'scvi/mean.npy',cfg['base']/'scvi/metadata.json'])
    hashes={}
    for path in source_paths:
        with path.open('rb') as handle:
            hashes[str(path.relative_to(ROOT))]=hashlib.file_digest(handle,'sha256').hexdigest()
    ranks=[]
    for ident in receipt['files']:
        frame=pd.read_parquet(O/'inputs'/f'{ident}.parquet')
        replay=expit(frame.original_logit.to_numpy())
        ranks.append(dict(id=ident,n=len(frame),spearman_original_replay=float(spearmanr(frame.safe_fusion,replay).statistic),
                          max_abs_probability_difference=float(np.max(np.abs(frame.safe_fusion.to_numpy()-replay)))))
    receipt.update(source_sha256=hashes,teacher_contracts=contracts,evaluation_original_replay=ranks,
                   p_nonzero_implementation='scripts/nonzero_probability.py scvi_score formula; saved theta times log1p(normalized mean times observed panel library / 1e4 / theta)' if kind!='hao' else 'saved original scVI P(X>0)')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(kind,'audit complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['hao','papalexi_small','papalexi_full','merge','audit_hao','audit_papalexi_small','audit_papalexi_full'])
    args=parser.parse_args()
    if args.kind=='merge':
        meta={}
        for kind in ['hao','papalexi_small','papalexi_full']:
            meta.update(json.loads((O/'inputs'/f'metadata_{kind}.json').read_text()))
        assert len(meta)==43
        (O/'inputs/metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    elif args.kind.startswith('audit_'):
        audit(args.kind.removeprefix('audit_'))
    else:
        prepare(args.kind)
