import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/sle_transductive').resolve()
OLD=Path('artifacts/paper_evidence/sle_treg_case').resolve()
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
        assert meta['scale']=='counts' and len(meta['cell_ids'])==n
    meta=json.loads((unit/'methods/safe_fusion/metadata.json').read_text())
    assert meta['parameters']['fit_cells']==train
    assert meta['parameters']['test_labels_used_for_fit'] is False
    assert meta['parameters']['teacher_names']==['gene_median','svd_impute','graph_smooth','magic','scvi']
    selector=json.loads((unit/'selector/selected_gene_scores_metadata.json').read_text())
    assert selector['fit_split']=='development' and selector['test_labels_used_for_training'] is False
    assert selector['missing_genes']==[]
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
    rows.append({'unit':name,'all_teacher_cells':n,'value_training_cells':train,'heldout_cells':test,'weighted_values_changed':changed,'selected_zero_values_10pct':zero_selected})
(O/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
sources=['scripts/sle_common.py','scripts/sle_fills.py','scripts/run_leakage_safe_method.py','scripts/count_scale_contract.py','scripts/calibrated_selective_fill.py','scripts/selector_attribution.py','scripts/v2_value_models.py','scripts/paired_masked_f1_bootstrap.py']
sources += [str(p) for p in Path('scripts').glob('sle_evaluate_*.py')]
sources += [str(p) for p in (Path('scripts/analyses/sle_transductive')).glob('*') if p.is_file()]
(O/'runtime/code_sha256.txt').write_text(''.join(f'{hashlib.sha256(Path(p).read_bytes()).hexdigest()}  {p}\n' for p in sources))
print(json.dumps({'audited_units':len(rows),'all_checks':'passed'}))
