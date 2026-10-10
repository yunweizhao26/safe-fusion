import csv
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import evaluate as local

OUT=local.OUT; ROUND=OUT.parent; frozen=local.frozen
OLD=local.OLD; NEW=local.NEW; METHODS=OLD+NEW; PCTS=local.PCTS
METHODS=['svd_all_cells','alra_all_cells']
LABELS={'svd_all_cells':'SVD (all cells)','alra_all_cells':'ALRA (all cells)'}
KEYS=['dataset','endpoint','method','fill_pct']

def collect(method,dataset,pcts):
    keys=[f'{dataset}_{i}' for i in range(3)] if dataset in ('colon','pancreas') else [dataset]
    statuses=[]
    for key in keys:
        path=OUT/'evaluation'/method/key/'status.json'
        statuses.append(json.loads(path.read_text()) if path.exists() else dict(status='missing',unit=key,error='No completed evaluation receipt'))
    bad=[s for s in statuses if s['status']!='completed']
    if bad: return None,bad
    unitcol='donor' if dataset in ('colon','pancreas') else 'unit'
    filename='donor_metrics.parquet' if unitcol=='donor' else 'unit_metrics.parquet'
    frame=pd.concat([pd.read_parquet(OUT/'evaluation'/method/key/filename) for key in keys])
    old=pd.read_csv(frozen.BASE/'summary/thinning_unit_values.csv',dtype={'unit':str})
    frames=[]
    for endpoint,metric in frozen.SPECS[dataset]:
        f=frame[frame.metric==metric].copy(); f[unitcol]=f[unitcol].astype(str)
        pivot=f.pivot(index=unitcol,columns='method',values='value')
        ref=old[(old.dataset==dataset)&(old.endpoint==endpoint)&(old.matrix=='corrupted_raw')].set_index('unit').reindex(pivot.index)
        assert ref.unthinned.notna().all()
        for name in ['corrupted_raw']+[frozen.name(method,p) for p in pcts]:
            frames.append(pd.DataFrame(dict(dataset=dataset,endpoint=endpoint,unit=pivot.index,stratum=ref.stratum.to_numpy(),matrix=name,value=pivot[name].to_numpy(),unthinned=ref.unthinned.to_numpy(),thinned_truth_pipeline=ref.thinned_truth_pipeline.to_numpy())))
    values=pd.concat(frames,ignore_index=True)

    gate=values.matrix.isin(['corrupted_raw',frozen.name(method,1),frozen.name(method,10)])
    gate_values=pd.read_csv(io.StringIO(values.loc[gate].to_csv(index=False)),dtype={'unit':str})
    return pd.concat([gate_values,values.loc[~gate]],ignore_index=True),[]

def summarize(values,method,pcts):
    frozen.METHODS=[method]
    result=frozen.summarize(values,pcts)
    for column in ['error_reduction','ci_low','ci_high']:result[column+'_x100']=100*result[column]
    result['fill_fraction']=result.fill_pct/100; result['method_label']=LABELS[method]
    gaps=[]; draw_checks=[]
    saved_draws=np.load(ROUND/'fill_grid_dense/bootstrap_draws.npz')
    diagnostics=json.loads((ROUND/'fill_grid_dense/bootstrap_diagnostics.json').read_text())
    for (dataset,endpoint),frame in values.groupby(['dataset','endpoint'],sort=False):
        pivot=frame.pivot(index='unit',columns='matrix',values='value')
        ref=frame.drop_duplicates('unit').set_index('unit').reindex(pivot.index)
        reference=ref.unthinned.to_numpy(); index=frozen.thin.unit_draws(ref.stratum.to_numpy(),2000,1729)
        raw_error=np.abs(pivot.corrupted_raw.to_numpy()-reference)
        denominator=np.nanmean(raw_error[index],axis=1)
        assert np.isfinite(denominator).all() and (denominator>0).all()
        prefix=next(d['array_prefix'] for d in diagnostics if d['dataset']==dataset and d['endpoint']==endpoint)
        np.testing.assert_array_equal(pivot.index.to_numpy(dtype=str),saved_draws[prefix+'_units'])
        np.testing.assert_array_equal(index,saved_draws[prefix+'_indices'])
        np.testing.assert_array_equal(denominator,saved_draws[prefix+'_denominator'])
        draw_checks.append(dict(dataset=dataset,endpoint=endpoint,method=method,draws=2000,units_exact=True,indices_exact=True,denominator_exact=True))
        for pct in pcts:
            reduction=raw_error-np.abs(pivot[frozen.name(method,pct)].to_numpy()-reference)
            numerator=np.nanmean(reduction[index],axis=1)
            low,high=np.nanquantile(numerator/denominator*100,[.025,.975])
            gaps.append(dict(dataset=dataset,endpoint=endpoint,method=method,fill_pct=pct,
                gap_closed_pct=np.nanmean(reduction)/np.nanmean(raw_error)*100,gap_ci_low_pct=low,gap_ci_high_pct=high))
    saved_draws.close()
    pd.DataFrame(draw_checks).to_csv(OUT/f'draw_identity_{method}_{values.dataset.iloc[0]}.csv',index=False)
    return result.merge(pd.DataFrame(gaps),on=KEYS,validate='one_to_one',sort=False)

def main():
    assert os.environ.get('SLURM_JOB_ID')
    frames=[]; unit_frames=[]
    for method in METHODS:
        for dataset in frozen.SPECS:
            values,bad=collect(method,dataset,PCTS)
            assert not bad,bad
            values.to_csv(OUT/f'unit_values_{method}_{dataset}.csv',index=False)
            unit_frames.append(values.assign(evaluation_method=method))
            frames.append(summarize(values,method,PCTS))
    columns=pd.read_csv(ROUND/'published_downstream/fill_grid_published_long.csv',nrows=0).columns.tolist()
    frame=pd.concat(frames,ignore_index=True)[columns]
    assert len(frame)==266 and not frame.duplicated(KEYS).any()
    assert np.isfinite(frame.select_dtypes('number')).all().all()
    assert all(len(g)==19 and sorted(g.fill_pct)==PCTS for _,g in frame.groupby(KEYS[:3]))
    frame.to_csv(OUT/'fill_grid_allcells_long.csv',index=False)
    pd.concat(unit_frames,ignore_index=True).to_parquet(OUT/'new_unit_values.parquet',index=False)
    frame=pd.read_csv(OUT/'fill_grid_allcells_long.csv')
    old=pd.read_csv(ROUND/'published_downstream/fill_grid_published_long.csv')
    comparisons=[]
    for row in frame.loc[frame.fill_pct.isin([1,10,100])].to_dict('records'):
        method=row['method'].removesuffix('_all_cells')
        previous=old[(old.dataset==row['dataset'])&(old.endpoint==row['endpoint'])&(old.method==method)&(old.fill_pct==row['fill_pct'])]
        assert len(previous)==1
        previous=previous.iloc[0]
        item={k:row[k] for k in ['dataset','endpoint','fill_pct']}; item['method']=method
        for col in ['error_reduction_x100','ci_low_x100','ci_high_x100','gap_closed_pct','gap_ci_low_pct','gap_ci_high_pct']:
            item['training_'+col]=previous[col];item['allcells_'+col]=row[col]
        item['delta_error_reduction_x100']=row['error_reduction_x100']-previous['error_reduction_x100']
        item['delta_gap_closed_pct']=row['gap_closed_pct']-previous['gap_closed_pct']
        comparisons.append(item)
    pd.DataFrame(comparisons).to_csv(OUT/'training_vs_allcells.csv',index=False)
    assert len(comparisons)==42
    checks=pd.concat([pd.read_csv(p) for p in OUT.glob('draw_identity_*.csv')],ignore_index=True)
    assert len(checks)==14 and checks[['units_exact','indices_exact','denominator_exact']].all().all()
    checks.to_csv(OUT/'draw_identity_all.csv',index=False)
    receipt=dict(rows=len(frame),comparison_rows=len(comparisons),methods=METHODS,fractions=PCTS,bootstrap_draws=2000,seed=1729,draw_identity_checks=len(checks),all_draw_checks_exact=True,job_id=os.environ['SLURM_JOB_ID'])
    (OUT/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
