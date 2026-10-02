import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
O=Path('artifacts/paper_evidence/review_round4/sle_donor_labels')
SRC=O.with_name('sle_transductive')
checks=[]
for old,new in [('conditional','verification'),('conditional_detection','verification_detection')]:
    for unit in json.loads((O/'units.json').read_text()):
        files=list((SRC/old/unit/'fills').glob('*.npz'))
        assert len(files)==6,(unit,len(files))
        for p in files:
            with np.load(p) as a,np.load(O/new/unit/'fills'/p.name) as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files),p
            checks.append({'variant':old,'path':str(p.relative_to(SRC/old)),'kind':'exact fill replay'})
    files=sorted((SRC/old/'results').glob('*/*.csv'))
    assert len(files)==25,len(files)
    for p in files:
        a=pd.read_csv(p,keep_default_na=False,na_values=[''])
        b=pd.read_csv(O/new/p.relative_to(SRC/old),keep_default_na=False,na_values=[''])
        assert list(a.columns)==list(b.columns) and a.shape==b.shape,p
        for column in a:
            for x,y in zip(a[column],b[column]):
                if isinstance(x,str):
                    if x==y: continue
                    assert x.startswith('[') and y.startswith('['),(p,column,x,y)
                    x=json.loads(x.replace('nan','NaN')); y=json.loads(y.replace('nan','NaN'))
                assert np.allclose(x,y,rtol=1e-10,atol=1e-10,equal_nan=True),(p,column,x,y)
        checks.append({'variant':old,'path':str(p.relative_to(SRC/old)),'kind':'endpoint replay'})
for fold in range(3):
    unit=O/f'verification/main/fold_{fold}/masked/selector'
    frame=pd.read_parquet(unit/'selected_gene_scores.parquet',columns=['selector_score','masked_positive'],filters=[('split','=','test')])
    actual=average_precision_score(frame.masked_positive,frame.selector_score)
    expected=json.loads((unit/'calibration_report.json').read_text())['test']['pr_auc']
    saved=pd.read_csv(SRC/'comparisons/masked_ap_intervals.csv')
    saved=saved[(saved.fold==fold)&(saved.variant=='Conditional')].iloc[0].average_precision
    assert abs(actual-expected)<1e-12 and abs(actual-saved)<1e-12,(fold,actual,expected,saved)
    checks.append({'fold':fold,'kind':'exact AP replay','value':actual})
    del frame
(O/'verification/checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(O/'verification/PASSED').write_text(f'{len(checks)} unlabeled replay checks passed\n')
print((O/'verification/PASSED').read_text())
