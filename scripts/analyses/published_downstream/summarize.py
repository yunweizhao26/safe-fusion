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
LABELS=dict(zip(OLD,frozen.LABELS))|dict(saver='SAVER',dca='DCA',scimpute='scImpute',screcover='scRecover',enimpute='EnImpute',scgpt='scGPT')
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
    with (ROUND/'fill_grid_dense/fill_grid_dense_long.csv').open() as f:
        reader=csv.DictReader(f); columns=reader.fieldnames; saved=list(reader)
    def key(row):return tuple(row[k] for k in KEYS[:-1])+(float(row['fill_pct']),)
    saved_map={key(r):r for r in saved}; rows=saved.copy(); failures=[]; reproduced=[]; unit_frames=[]
    for method in METHODS:
        pcts=PCTS if method in NEW or method=='svd' else PCTS[13:]
        for dataset,endpoints in frozen.SPECS.items():
            values,bad=collect(method,dataset,pcts)
            if bad:
                for error in bad: failures.append(dict(analysis='downstream',dataset=dataset,method=method)|error)
                for endpoint,_ in endpoints:
                    for pct in pcts:
                        if method in OLD and pct in PCTS[:13]:continue
                        rows.append(dict(dataset=dataset,endpoint=endpoint,method=method,fill_pct=pct,method_label=LABELS[method],fill_fraction=pct/100,verdict='unavailable; see failures.csv'))
                continue
            values.to_csv(OUT/f'unit_values_{method}_{dataset}.csv',index=False)
            unit_frames.append(values.assign(evaluation_method=method))
            result=summarize(values,method,pcts)
            for row in result.to_dict('records'):
                if method=='svd' and row['fill_pct'] in PCTS[:13]:
                    previous=saved_map[key(row)]
                    for col in columns:
                        if col in KEYS or col in ('method_label','verdict'):continue
                        old=float(previous[col]); new=float(row[col])
                        reproduced.append(dict(**{k:row[k] for k in KEYS},field=col,saved=previous[col],rerun=str(row[col]),exact_csv_match=previous[col]==str(row[col]),absolute_difference=abs(old-new),match_2dp=f'{old:.2f}'==f'{new:.2f}',match_12dp=f'{old:.12f}'==f'{new:.12f}'))
                else: rows.append(row)
    assert len(rows)==len(METHODS)*7*19 and len({key(r) for r in rows})==len(rows)
    order={m:i for i,m in enumerate(METHODS)}
    rows.sort(key=lambda r:(list(frozen.SPECS).index(r['dataset']),[x[0] for x in frozen.SPECS[r['dataset']]].index(r['endpoint']),order[r['method']],float(r['fill_pct'])))
    with (OUT/'fill_grid_published_long.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    with (OUT/'fill_grid_published_long.csv').open() as f: written={key(r):r for r in csv.DictReader(f)}
    preserved=all(written[key(r)]==r for r in saved);assert preserved
    pd.DataFrame(failures,columns=['analysis','dataset','method','unit','status','error','traceback','job_id']).to_csv(OUT/'failures.csv',index=False)
    pd.DataFrame(reproduced).to_csv(OUT/'svd_reproduction.csv',index=False)
    if unit_frames:pd.concat(unit_frames,ignore_index=True).to_parquet(OUT/'new_unit_values.parquet',index=False)
    frame=pd.read_csv(OUT/'fill_grid_published_long.csv')
    peaks=[]
    for (dataset,endpoint,method),g in frame.groupby(['dataset','endpoint','method'],sort=False):
        if endpoint not in ('Marker AUPRC','Dynamics'):continue
        valid=g.dropna(subset=['gap_closed_pct'])
        if len(valid):
            best=valid.gap_closed_pct.max(); top=valid.loc[valid.gap_closed_pct==best]
            peaks.append(dict(dataset=dataset,endpoint=endpoint,method=method,peak_fill_pct=';'.join(f'{x:g}' for x in top.fill_pct),peak_gap_closed_pct=best,complete_grid=len(valid)==19))
        else:peaks.append(dict(dataset=dataset,endpoint=endpoint,method=method,peak_fill_pct='NA',peak_gap_closed_pct=np.nan,complete_grid=False))
    pd.DataFrame(peaks).to_csv(OUT/'peak_fractions.csv',index=False)
    receipt=dict(fractions=PCTS,rows=len(rows),retained_rows=len(saved),retained_all_columns_exact_csv=preserved,svd_checked_values=len(reproduced),svd_exact_csv_matches=sum(r['exact_csv_match'] for r in reproduced),svd_match_12dp=all(r['match_12dp'] for r in reproduced),svd_match_2dp=all(r['match_2dp'] for r in reproduced),svd_max_absolute_difference=max([r['absolute_difference'] for r in reproduced],default=None),failed_units=len(failures),unavailable_rows=int(frame.error_reduction.isna().sum()),draws=2000,seed=1729)
    (OUT/'reproduction.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
