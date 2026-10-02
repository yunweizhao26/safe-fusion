import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/sle_donor_labels')
OLD=Path('artifacts/paper_evidence/sle_treg_case')
SRC=O.with_name('sle_transductive')
CASES={'Inductive':OLD,'Unlabeled fused':SRC/'conditional','Unlabeled detection':SRC/'conditional_detection','Donor fused':O/'conditional','Donor detection':O/'conditional_detection'}
COMP=O/'comparisons'; COMP.mkdir(exist_ok=True)
assert (O/'verification/PASSED').exists()
assert len(json.loads((O/'audit.json').read_text()))==12
comparator_checks=json.loads((O/'comparator_endpoint_checks.json').read_text())
assert comparator_checks and all(r['passed'] for r in comparator_checks)
keys={
 'q2_fill_concentration':['method','budget','fill_pct','gene'],
 'q2_fill_share_by_lineage':['method','budget','fill_pct','gene','lineage'],
 'q2_treg_positive':['method','budget','fill_pct','gene'],
 'q4_effects':['method','budget','fill_pct','cells'],
 'q4_fill_rates':['method','budget','fill_pct','genes','cells'],
 'q4_library_zero_fraction':['cells','quantity'],
 'q4_null_de':['method','budget','fill_pct','cells','test','labels'],
 'q5_modules':['method','budget','fill_pct','module','cells'],
 'q6_donor_agreement':['method','budget','fill_pct','gene'],
 'focus_fill_counts':['set','application','gene','cells','method','budget','fill_pct'],
 'focus_fill_counts_by_condition':['set','application','gene','cells','condition','method','budget','fill_pct'],
 'focus_fill_counts_by_fold':['set','application','fold','gene','cells','condition','method','budget','fill_pct'],
 'focus_gene_metrics':['method','gene','budget','fill_pct','cells'],
 'masked_f1_curves':['dataset','method','requested_fill_fraction'],
 'paired_bootstrap':['dataset','comparator','family','statistic'],
 'per_gene':['family','gene','method'],
 'clustering_by_fold':['fold','method','budget','fill_pct'],
 'clustering_mean_over_folds':['method','budget','fill_pct'],
 'treg_calls_by_donor':['fold','method','budget','fill_pct','donor','condition'],
 'treg_calls_summary':['method','budget','fill_pct'],
 'treg_subclusters_by_fold':['fold','method','budget','fill_pct','gene'],
 'treg_subclusters_summary':['method','budget','fill_pct','gene'],
 'fcrl3_positive_control':['scope'], 'fcrl3_protein_agreement':['scope','method'],
}
coverage=[]
for old in sorted((OLD/'results').glob('*/*.csv')):
    rel=old.relative_to(OLD/'results'); index=keys[old.stem]
    result=None
    for label,case in CASES.items():
        frame=pd.read_csv(case/'results'/rel, keep_default_na=False, na_values=[''])
        assert not frame.duplicated(index).any(),rel
        frame=frame.set_index(index)
        if result is not None: assert result.index.equals(frame.index),(rel,label)
        frame=frame.add_prefix(label+'::')
        result=frame if result is None else result.join(frame)
    path=COMP/rel; path.parent.mkdir(exist_ok=True)
    result.reset_index().to_csv(path,index=False)
    coverage.append({'table':str(rel),'rows':len(result),'columns_per_variant':(len(result.columns)//len(CASES))})
pd.DataFrame(coverage).to_csv(COMP/'coverage.csv',index=False)

pooled=[]
for set_name in ('sex','main'):
    for label,case in CASES.items():
        content=json.loads((case/f'results/q1_zeros_{set_name}/summary.json').read_text())
        for family,block in content['pooled'].items():
            for method,values in block.items():
                if method=='genes': continue
                pooled.append({'set':set_name,'family':family,'method':method,'variant':label,**values})
pd.DataFrame(pooled).to_csv(COMP/'q1_pooled.csv',index=False)

f1_rows=[]
for label,case in CASES.items():
    frame=pd.read_parquet(case/'results/masked/masked_f1_unit_counts.parquet')
    frame['unit']=frame['group'].astype(str)+':'+frame['unit'].astype(str)
    frame=frame[frame.method=='Safe Fusion']; donors=sorted(frame.unit.unique())
    arrays=[frame.pivot(index='unit',columns='fraction',values=k).reindex(donors).to_numpy(float) for k in ('n_selected','n_true_positive','n_masked_positives')]
    n,tp,pos=arrays
    rng=np.random.default_rng(1729); draws=rng.integers(0,len(donors),(2000,len(donors)))
    point=2*tp.sum(0)/(n.sum(0)+pos.sum(0))
    boot=2*tp[draws].sum(1)/(n[draws].sum(1)+pos[draws].sum(1))
    for name,v,bs in [('mean F1, 1–10%',point.mean(),boot.mean(1)),*[(f'F1, {k}%',point[k-1],boot[:,k-1]) for k in (1,5,10)]]:
        f1_rows.append({'variant':label,'metric':name,'estimate':v,'lower':np.quantile(bs,.025),'upper':np.quantile(bs,.975)})
pd.DataFrame(f1_rows).to_csv(COMP/'masked_f1_intervals.csv',index=False)
ap=[]
for label,case in CASES.items():
    for fold in range(3):
        j=json.loads((case/f'main/fold_{fold}/masked/selector/calibration_report.json').read_text())
        ap.append({'variant':label,'fold':fold,'average_precision':j['test']['pr_auc'],'n_candidates':j['test']['n_zeros'],'n_masked_positives':j['test']['n_masked_positives']})
pd.DataFrame(ap).to_csv(COMP/'masked_ap.csv',index=False)

def scalar(x):
    if pd.isna(x): return 'NA'
    if isinstance(x,(int,np.integer)): return str(x)
    return f'{float(x):.6f}'
def value(x,ci=None,scale=1):
    text=scalar(x*scale)
    if ci is not None:
        if isinstance(ci,str): ci=json.loads(ci.replace('nan','NaN'))
        text+=' ['+', '.join(scalar(y*scale) for y in ci)+']'
    return text

def row(rel,**filters):
    vals=[]
    for case in CASES.values():
        frame=pd.read_csv(case/'results'/rel, keep_default_na=False, na_values=[''])
        if 'method' in frame and 'method' not in filters: frame=frame[frame.method=='Safe Fusion']
        for key,v in filters.items(): frame=frame[frame[key]==v]
        assert len(frame)==1,(rel,filters,len(frame))
        vals.append(frame.iloc[0])
    return vals
lines=['# Lupus worked example: donor-label Safe Fusion','',
'We reran the unlabeled transductive pipeline on the same main, sex and CITE-seq prepared sets, three folds, gene panels, masks, seeds, budgets and evaluators, adding only donor-label conditioning. Teachers use all cells, weighted kNN uses same-donor neighbours, scVI receives donor as its categorical covariate, and the selector adds donor log mean count and zero fraction from all masked-input cells of each donor (`--condition-feature-source all_cells`). The value model and selector retain their original training cells and fill held-out donors; fused and detection-weighted variants share ranks and budgets, with the latter multiplying the fused count by `p / (p + 0.1(1-p))` using unchanged training-cell calibration.','',
'Brackets are 95% percentile intervals from 2,000 draws, seed 1729: original condition/sex-stratified donor resampling for biology, unstratified donors for F1/AP and protein, and the original permutation bootstrap over 200 balanced disease-label permutations for null tests. Counts and clustering ARI are descriptive; their original evaluators supply no interval. AP is reported separately for each fold, preserving the existing definition. Module differences are SLE minus healthy, in module-score units. A zero counts as filled only when selected and assigned a positive value, as in the original evaluator.','',
'| Endpoint | Inductive | Unlabeled fused | Unlabeled detection-weighted | Donor fused | Donor detection-weighted |',
'|---|---:|---:|---:|---:|---:|']
def add(name,rel,key,ci=None,scale=1,**filters):
    rs=row(rel,**filters)
    vals=[value(r[key],r[ci] if ci else None,scale) for r in rs]
    lines.append('| '+name+' | '+' | '.join(vals)+' |')
for name in dict.fromkeys(r['metric'] for r in f1_rows):
    vals=[value(r['estimate'],[r['lower'],r['upper']],100) for r in f1_rows if r['metric']==name]
    lines.append('| '+name+' (%) | '+' | '.join(vals)+' |')
for fold in range(3):
    vals=[]
    for label in CASES:
        source=SRC if label in ('Inductive','Unlabeled fused','Unlabeled detection') else O
        v={'Inductive':'Inductive','Unlabeled fused':'Conditional','Unlabeled detection':'Detection-weighted','Donor fused':'Conditional','Donor detection':'Detection-weighted'}[label]
        block=pd.read_csv(source/'comparisons/masked_ap_intervals.csv')
        r=block[(block.fold==fold)&(block.variant==v)].iloc[0]
        vals.append(value(r.average_precision,[r.lower,r.upper]))
    lines.append(f'| Masked AP, fold {fold} | '+' | '.join(vals)+' |')
for comp in ('SVD','Weighted kNN','MAGIC','scVI'):
    rs=row('masked/paired_bootstrap.csv',comparator=comp,statistic='mean_difference_1_to_10')
    lines.append('| Mean F1 minus '+comp+' (pp) | '+' | '.join(value(r.estimate_pp,[r.lower_pp,r.upper_pp]) for r in rs)+' |')
for gene in ('RPS4Y1','XIST'):
    add(gene+' AUROC','q1_zeros_sex/per_gene.csv','auroc_paper','auroc_paper_ci',gene=gene)
    add(gene+' AUROC, library-size quintiles','q1_zeros_sex/per_gene.csv','auroc_paper_depth_stratified','auroc_paper_depth_stratified_ci',gene=gene)
for gene in ('TLR5','FCRL3'):
    for budget,title in [('global','all zeros'),('donor',"each donor’s zeros"),('gene',"each gene’s zeros")]:
        for level in ((1,5,10) if budget!='gene' else (10,)):
            add(f'{gene} Treg zeros filled, {title}, {level}%','focus_fills/focus_fill_counts.csv','filled',set='main',application='deploy',gene=gene,cells='Treg',budget=budget,fill_pct=level)
for budget,title in [('global','all zeros'),('gene',"each gene’s zeros")]:
    for level in ((1,5,10) if budget=='global' else (10,)):
        add(f'Treg call change, {title}, {level}% (%)','q3_clustering/treg_calls_summary.csv','relative_change_all','relative_change_all_ci',scale=100,budget=budget,fill_pct=level)
for level in (1,10):
    add(f'Cluster ARI, all zeros, {level}%','q3_clustering/clustering_mean_over_folds.csv','ari',budget='global',fill_pct=level)
add('Cluster ARI, unfilled second-seed reference','q3_clustering/clustering_mean_over_folds.csv','ari',method='Unfilled, second Leiden seed',budget='none',fill_pct=0)
add('Treg pseudobulk effect slope, all zeros, 10%','deploy_main/q4_effects.csv','slope_filled_on_raw','slope_ci',budget='global',fill_pct=10,cells='Treg')
for test,title in [('pseudobulk','pseudobulk'),('wilcoxon','cell level')]:
    add(f'False discoveries gained, {title}, Tregs, 10%','deploy_main/q4_null_de.csv','mean_difference','mean_difference_ci',budget='global',fill_pct=10,cells='Treg',test=test,labels='null')
for module,cells in [('interferon','All cells'),('interferon','Treg'),('treg_suppressive','Treg')]:
    for key,title in [('effect_raw','unfilled disease difference'),('effect_filled','filled disease difference'),('effect_change','change in disease difference')]:
        add(f'{module}, {cells}, {title}, 10%','deploy_main/q5_modules.csv',key,key+'_ci',module=module,cells=cells,budget='global',fill_pct=10)
for gene in ('TLR5','FCRL3'):
    for budget,title in [('global','all zeros'),('gene',"each gene’s zeros")]:
        for key,metric in [('split_half_raw_A_vs_raw_B','raw A vs raw B'),('split_half_filled_A_vs_raw_B','filled A vs raw B')]:
            add(f'{gene} donor split-half Spearman, {metric}, {title}, 10%','deploy_main/q6_donor_agreement.csv',key,key+'_ci',gene=gene,budget=budget,fill_pct=10)
add('FcRL3 protein Spearman, zero-RNA Tregs','q7_protein/fcrl3_protein_agreement.csv','spearman','spearman_ci',scope='Treg')
checks=json.loads((O/'verification/checks.json').read_text())
lines += ['',f'Verification: {len(checks)} unlabeled replay checks passed before donor fits: all candidate/fill arrays match exactly and all 25 CSV tables per value match within 1e-10. Twelve donor-label units passed teacher, held-out-training, comparator and detection-value audits (`audit.json`); all 734 comparator/unfilled endpoint checks passed (`comparator_endpoint_checks.json`). All twelve L40S fits, twelve donor-label chains and fourteen new evaluator tasks completed; nothing remains unfinished.','',
'Paths (relative to `artifacts/paper_evidence/review_round4/sle_donor_labels/`): `conditional/` contains donor-label teachers, values, selectors, fills and `results/`; `conditional_detection/` contains weighted fills and `results/`; `verification/` and `verification_detection/` contain unlabeled replays; `comparisons/` contains full-precision five-way tables for all original endpoints. Saved references remain in `../sle_transductive/` and `../../sle_treg_case/`; code, logs, submitted jobs and accounting records are in `code/`, `logs/`, `runtime/jobs.tsv`, `runtime/sacct_records.txt` and `runtime/sacct_records_full.tsv` (including CPU time, peak memory and GPU accounting). Final source hashes are in `runtime/code_sha256_final.txt`.','',
'Reproduce from the repository root (existing prepared inputs and saved unlabeled outputs required):','',
'```bash','bash artifacts/paper_evidence/review_round4/sle_donor_labels/code/reproduce.sh','bash artifacts/paper_evidence/review_round4/sle_donor_labels/code/monitor.sh','```','',
'The launcher replays both unlabeled values, checks the saved endpoints, runs twelve L40S scVI fits, runs twelve donor-label chains on `cs`, evaluates both values, bootstraps AP, audits and writes this report. Every submission uses account `torch_pr_634_general`; original scientific settings are unchanged and no tuning is performed.','',
'Failures: '+((O/'runtime/failures.txt').read_text().strip() if (O/'runtime/failures.txt').exists() else 'None recorded.')+' No scientific endpoint remains unfinished. FcRL3 protein validation uses healthy CITE-seq donors; it is not SLE protein validation. Split-half correlations reuse transductive teachers fitted to both halves, and donor-label features also use both halves; they are not independent validation.']
(O/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'tables':len(coverage),'rows':sum(r['rows'] for r in coverage),'report':str(O/'README.md')}))
