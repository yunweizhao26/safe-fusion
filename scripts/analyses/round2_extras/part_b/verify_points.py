import json
from pathlib import Path
import pandas as pd
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_b')
f=pd.read_csv(O/'selector_f1_fillrate_allcell_1000_points.csv')
source=O.parent.parent/'transductive_comparators'
a=pd.read_csv(source/'absolute.csv'); d=pd.read_csv(source/'paired_differences.csv')
checks=[]
for (dataset,method),part in f[f.method.isin(['SVD','Weighted kNN','ALRA','Safe Fusion (transductive)'])].groupby(['dataset','method']):
    points=part[part.coverage_index.isin(range(9,100,10))]
    estimate=100*points.masked_f1.mean()
    key=method if method.startswith('Safe Fusion') else method+' (transductive)'
    saved=a[(a.dataset==dataset)&(a.method==key)&(a.statistic=='mean_f1_1_to_10')].iloc[0]
    assert round(estimate,4)==round(saved.estimate_percent,4),(dataset,method,estimate,saved.estimate_percent)
    checks.append(dict(dataset=dataset,method=method,estimate_percent=estimate,saved_estimate_percent=saved.estimate_percent))
d=d[(d.reference=='Safe Fusion (transductive)')&d.comparator.isin([m+' (transductive)' for m in ['SVD','Weighted kNN','ALRA']])&(d.statistic=='mean_f1_1_to_10')]
assert len(d)==9
d.to_csv(O/'paired_mean_f1.csv',index=False)
(O/'table1_parity.json').write_text(json.dumps(dict(checks=checks,exact_at_saved_precision=True,interval_source=str(source/'paired_differences.csv')),indent=2)+'\n')
comparators=['Weighted kNN','SVD','ALRA','SAVER','MAGIC','scVI','scGPT']
rows=[]
for dataset,frame in f.groupby('dataset'):
    wins=[]; strict=[]; margins=[]
    for _,point in frame[frame.coverage_index<100].groupby('coverage_index'):
        point=point.set_index('method')
        def counts(name):
            r=point.loc[name]
            values=[r.n_true_positive,r.n_selected,r.n_masked_positives]
            assert all(float(v).is_integer() for v in values)
            tp,selected,positive=map(int,values)
            return 2*tp,selected+positive
        numerator,denominator=counts('Safe Fusion (transductive)')
        differences=[]; margin=[]
        for method in comparators:
            n,d=counts(method)

            difference=numerator*d-n*denominator
            differences.append(difference); margin.append(difference/(denominator*d))
        wins.append(all(v>=0 for v in differences)); strict.append(all(v>0 for v in differences)); margins.append(min(margin))
    assert len(wins)==100
    rows.append(dict(dataset=dataset,n_fractions=100,highest=sum(wins),strictly_highest=sum(strict),minimum_margin=min(margins)))
pd.DataFrame(rows).to_csv(O/'highest_f1.csv',index=False)
print(rows)
