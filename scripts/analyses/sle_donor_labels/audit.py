import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import pyarrow.parquet as pq
from scipy import sparse
O=Path('artifacts/paper_evidence/review_round4/sle_donor_labels').resolve()
OLD=Path('artifacts/paper_evidence/sle_treg_case').resolve()
for line in (O/'runtime/source_sha256_before.txt').read_text().splitlines():
    expected,path=line.split('  ',1)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,('Scientific source changed during run',path)
rows=[]
for name in json.loads((O/'units.json').read_text()):
    unit=O/'conditional'/name
    app=name.split('/')[-1]
    split=pd.read_parquet(unit/('splits.parquet' if app=='deploy' else '../splits.parquet'))
    n=len(split); train=int(split['split'].isin(['development','validation']).sum()); test=int((split['split']=='test').sum())
    assert train+test==n
    for teacher in ('gene_median','svd_impute','graph_smooth','magic','scvi'):
        meta=json.loads((unit/'teachers'/teacher/'metadata.json').read_text())
        assert meta['parameters']['transductive'] and meta['parameters']['fit_cells']==n
        if teacher in ('graph_smooth','scvi'): assert meta['parameters']['condition_column']=='donor'
        assert meta['scale']=='counts' and len(meta['cell_ids'])==n
    meta=json.loads((unit/'methods/safe_fusion/metadata.json').read_text())
    assert meta['parameters']['fit_cells']==train
    assert meta['parameters']['test_labels_used_for_fit'] is False
    assert meta['parameters']['teacher_names']==['gene_median','svd_impute','graph_smooth','magic','scvi']
    selector=json.loads((unit/'selector/selected_gene_scores_metadata.json').read_text())
    assert selector['fit_split']=='development' and selector['test_labels_used_for_training'] is False
    assert selector['missing_genes']==[]
    report=json.loads((unit/'selector/calibration_report.json').read_text())
    assert report['condition_feature_source']=='all_cells'
    source=unit/('hybrid.h5ad' if app=='deploy' else 'corrupted.h5ad')
    original=Path('artifacts/paper_evidence/review_round4/sle_transductive/conditional')/name
    for relative in ('coordinates.parquet', '../truth.h5ad', 'splits.parquet' if app=='deploy' else '../splits.parquet', source.name):
        assert (unit/relative).resolve()==(original/relative).resolve(),(name,relative)
    data=ad.read_h5ad(source)
    counts=data.layers['corrupted_counts']
    counts=counts.toarray().astype(np.float32) if sparse.issparse(counts) else np.asarray(counts,dtype=np.float32)
    names,codes=np.unique(data.obs['donor'].astype(str).to_numpy(),return_inverse=True)
    mean=np.asarray([np.log1p(counts[codes==k].mean(0)) for k in range(len(names))])
    zero=np.asarray([(counts[codes==k]<=0).mean(0) for k in range(len(names))],dtype=np.float32)
    features=pq.ParquetFile(unit/'selector/selected_gene_scores.parquet')
    assert 'condition_gene_mean' in features.schema.names and 'condition_gene_dropout' in features.schema.names
    sampled=0
    for batch in features.iter_batches(batch_size=262144,columns=['cell_index','gene_index','condition_gene_mean','condition_gene_dropout']):
        frame=batch.slice(0,min(16,len(batch))).to_pandas()
        rr=frame.cell_index.to_numpy(); cc=frame.gene_index.to_numpy()
        assert np.array_equal(frame.condition_gene_mean.to_numpy(),mean[codes[rr],cc]),name
        assert np.array_equal(frame.condition_gene_dropout.to_numpy(),zero[codes[rr],cc]),name
        sampled+=len(frame)
    del data,counts,mean,zero
    changed=0; zero_selected={}
    with np.load(unit/'fills/Safe_Fusion.npz') as a, np.load(O/'conditional_detection'/name/'fills/Safe_Fusion.npz') as b:
        for k in a.files:
            if k!='value': assert np.array_equal(a[k],b[k]), (name,k)
        d=np.load(O/'conditional_detection'/name/'fills/detection_probability.npy')
        assert np.array_equal((a['value']*d).astype(np.float32),b['value'])
        changed=int(np.sum(a['value']!=b['value']))
        assert np.isfinite(b['value']).all() and (b['value']>=0).all()
        zero_selected={mode:int(np.sum((b['first_'+mode]<=10)&(b['value']==0))) for mode in ('global','donor','gene')}
    for method in ('SVD','Weighted_kNN','MAGIC','scVI'):
        with np.load(unit/f'fills/{method}.npz') as a, np.load(OLD/name/f'fills/{method}.npz') as b:
            assert all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files), (name,method)
    rows.append({'unit':name,'all_teacher_cells':n,'value_training_cells':train,'heldout_cells':test,'donor_feature_rows_checked':sampled,'weighted_values_changed':changed,'selected_zero_values_10pct':zero_selected})
(O/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
sources=['scripts/sle_common.py','scripts/sle_fills.py','scripts/run_leakage_safe_method.py','scripts/count_scale_contract.py','scripts/calibrated_selective_fill.py','scripts/selector_attribution.py','scripts/v2_value_models.py','scripts/paired_masked_f1_bootstrap.py']
sources += [str(p) for p in Path('scripts').glob('sle_evaluate_*.py')]
sources += [str(p) for p in (O/'code').glob('*') if p.is_file()]
(O/'runtime/code_sha256.txt').write_text(''.join(f'{hashlib.sha256(Path(p).read_bytes()).hexdigest()}  {p}\n' for p in sources))
print(json.dumps({'audited_units':len(rows),'all_checks':'passed'}))
