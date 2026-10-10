import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT=Path.cwd(); OUT=ROOT/'artifacts/paper_evidence/review_round4/published_protein'; ROUND=OUT.parent
METHODS=['saver','dca','scimpute','screcover','enimpute','scgpt','alra']
LABELS=dict(safe_fusion='Safe Fusion',svd='SVD',svd_all_cells='SVD, all cells',weighted_knn='Weighted kNN',weighted_knn_all_cells='Weighted kNN, all cells',scvi='scVI',magic='MAGIC',p_nonzero='scVI nonzero probability',fused_value='Fused value',target_baseline='Target baseline',ifng_score='Interferon-gamma score',library_size='Library size',cell_zero_score='Cell zero score',saver='SAVER',dca='DCA',scimpute='scImpute',screcover='scRecover',enimpute='EnImpute',scgpt='scGPT',alra='ALRA')

def pick(frame,name,metric,paired=False):
    col='comparator' if paired else 'ranking'
    scope='control' if metric=='control' else 'all'
    resampling='cells_within_replicate' if metric=='control' else 'target_clusters'
    stat={'pooled':'pooled_spearman','within':'within_target_spearman','control':'pooled_spearman','partial':'partial_spearman'}[metric]
    f=frame[(frame[col]==name)&(frame.scope==scope)&(frame.resampling==resampling)&(frame.statistic==stat)]
    if metric=='partial':f=f[f.covariates=='target_ifng_library']
    assert len(f)<=1,(name,metric,len(f))
    return f.iloc[0] if len(f) else None

def interval(row,field='estimate'):
    if row is None or pd.isna(row[field]):return 'NA'
    return f'{row[field]:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]'

def main():
    assert os.environ.get('SLURM_JOB_ID')
    associations=[]; pairs=[]; estimates=[]; failures=[]; receipts=[]
    for block in ['small','full']:
        for method in ['existing']+METHODS:
            dest=OUT/'evaluation'/method/block
            status=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else dict(status='missing',error='No completed evaluation receipt')
            if status['status']!='completed':
                detail_path=OUT/'fits'/method/block/'failure_details.json'
                if detail_path.exists():
                    status['error']=json.loads(detail_path.read_text())['summary']
                    status['failure_details']=str(detail_path)
                failures.append(dict(block=block,method=method)|status);continue
            receipts.append(status)
            a=pd.read_csv(dest/'association.csv');p=pd.read_csv(dest/'paired_differences.csv');s=pd.read_csv(dest/'estimates.csv')
            if method!='existing':a=a[a.ranking==method];s=s[s.ranking==method]
            associations.append(a.assign(block=block));pairs.append(p.assign(block=block));estimates.append(s)
    association=pd.concat(associations,ignore_index=True);paired=pd.concat(pairs,ignore_index=True);se=pd.concat(estimates,ignore_index=True)

    for failure in failures:
        if failure['method']=='existing':continue
        block,method=failure['block'],failure['method']
        template=association[(association.block==block)&(association.ranking=='safe_fusion')].copy()
        template['ranking']=method
        template[['estimate','ci_low','ci_high']]=np.nan;template['defined_draws']=0
        association=pd.concat([association,template],ignore_index=True)
        t=template.drop(columns=['defined_draws']).rename(columns={'ranking':'comparator','estimate':'safe_fusion_minus_comparator'})
        paired=pd.concat([paired,t],ignore_index=True)
    failed_keys={(r['block'],r['method']) for r in failures}
    undefined=association[association.ranking.isin(METHODS)&association.estimate.isna()]
    undefined=undefined.loc[[key not in failed_keys for key in zip(undefined.block,undefined.ranking)]]
    undefined.to_csv(OUT/'undefined_associations.csv',index=False)
    association.to_csv(OUT/'association.csv',index=False);paired.to_csv(OUT/'paired_differences.csv',index=False);se.to_csv(OUT/'estimates.csv',index=False)
    pd.DataFrame(failures,columns=['block','method','status','error','traceback','job_id','failure_details']).to_csv(OUT/'failures.csv',index=False)
    table=[];s23=[];paired_rows=[]
    for block in ['small','full']:
        a=association[association.block==block];p=paired[paired.block==block]
        names=['safe_fusion','svd' if block=='small' else 'svd_all_cells','weighted_knn' if block=='small' else 'weighted_knn_all_cells','scvi','magic','p_nonzero','fused_value',*METHODS,'target_baseline','ifng_score','library_size']
        for name in names:
            row=dict(block=block,ranking=name,method_label=LABELS[name],fit_cells=3400 if block=='small' else 20156,evaluated_zeros=655 if block=='small' else 4083)
            for metric,stat in [('pooled','pooled_spearman'),('within_target','within_target_spearman')]:
                v=se[(se.block==block)&(se.ranking==name)&(se.statistic==stat)]
                if len(v):
                    r=v.iloc[0];row.update({metric+'_estimate':r.estimate,metric+'_se':r.se,metric:f'{r.estimate:.3f} +/- {r.se:.3f}' if np.isfinite([r.estimate,r.se]).all() else 'NA'})
                else:row.update({metric+'_estimate':np.nan,metric+'_se':np.nan,metric:'NA'})
            table.append(row)
            s23.append(dict(block=block,ranking=name,method_label=LABELS[name],**{m:interval(pick(a,name,m)) for m in ['pooled','within','control','partial']}))
            if name!='safe_fusion':paired_rows.append(dict(block=block,comparator=name,method_label=LABELS[name],**{m:interval(pick(p,name,m,True),'safe_fusion_minus_comparator') for m in ['pooled','within','control','partial']}))
    pd.DataFrame(table).to_csv(OUT/'table3_rows.csv',index=False);pd.DataFrame(s23).to_csv(OUT/'s23_rows.csv',index=False);pd.DataFrame(paired_rows).to_csv(OUT/'paired_rows.csv',index=False)
    tex=(Path(os.environ['SAFE_FUSION_MANUSCRIPT_DIR'])/'supplementary_results.tex').read_text().split('\\label{tab:s_protein_state}',1)[1].split('\\end{table}',1)[0]
    lines=[x for x in tex.splitlines() if x.startswith('Safe Fusion &')]
    assert len(lines)==2
    checks=[]
    for block,line in zip(['small','full'],lines):
        cells=line.split('&')[1:]
        expected=next(r for r in s23 if r['block']==block and r['ranking']=='safe_fusion')
        for metric,cell in zip(['pooled','within','control','partial'],cells):
            clean=re.sub(r'\\textbf\{([^{}]*)\}',r'\1',cell).replace('\\\\','').strip()
            checks.append(dict(block=block,metric=metric,manuscript=clean,rerun=expected[metric],exact_match=clean==expected[metric]))
    pd.DataFrame(checks).to_csv(OUT/'s23_safe_fusion_parity.csv',index=False)
    assert all(c['exact_match'] for c in checks)
    saved=pd.read_csv(ROUND/'protein_main_table/estimates.csv')
    keys=['block','statistic','ranking'];joined=saved.merge(se,on=keys,suffixes=('_saved','_new'),validate='one_to_one')
    assert len(joined)==len(saved)
    comparisons=[]
    for _,r in joined.iterrows():
        for c in ['estimate','se','ci_low','ci_high']:
            np.testing.assert_allclose(r[c+'_saved'],r[c+'_new'],rtol=0,atol=1e-12)
            comparisons.append(dict(**{k:r[k] for k in keys},field=c,old=r[c+'_saved'],new=r[c+'_new'],printed_exact=f'{r[c+"_saved"]:.3f}'==f'{r[c+"_new"]:.3f}'))
    pd.DataFrame(comparisons).to_csv(OUT/'table3_existing_parity.csv',index=False)
    receipt=dict(s23_safe_fusion_cells_exact=sum(c['exact_match'] for c in checks),s23_expected_cells=8,table3_existing_values_checked=len(comparisons),table3_printed_exact=all(c['printed_exact'] for c in comparisons),completed_evaluations=len(receipts),failed_evaluations=len(failures),draws=2000,seed=1729)
    (OUT/'reproduction.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
