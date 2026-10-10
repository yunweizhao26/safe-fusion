import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/sle_transductive')
OLD=Path('artifacts/paper_evidence/sle_treg_case')
CASES={'Inductive':OLD,'Conditional':O/'conditional','Detection-weighted':O/'conditional_detection'}
COMP=O/'comparisons'; COMP.mkdir(exist_ok=True)
assert (O/'verification/PASSED').exists()
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
    coverage.append({'table':str(rel),'rows':len(result),'columns_per_variant':(len(result.columns)//3)})
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
lines=['# SLE Treg case study: transductive Safe Fusion','',
'All outputs are isolated here. Original prepared sets, fold gene panels, masks, donor splits, comparator contracts, scores and results are unchanged. No paper, docs or repository README was edited.','',
'The five teachers use every cell in each existing fold input: gene median, SVD and weighted kNN were refitted without cross-fitting; the original standard all-cell MAGIC and L40S scVI fits were reused and converted to counts. The production boosted value learns only from training-cell masked positives. The production MLP selector learns from development-cell masked positives versus recorded zeros and scores held-out donors. All selector jobs use `cs`, eight threads, seed 1729. Fits use `.venv` and biology evaluators use `.venv-sle`, matching the original launcher.','',
'Conditional inserts the production fused count. Detection-weighted inserts that same count times `d = p / (p + 0.1(1-p))`; `p` uses five-fold, training-cell isotonic calibration with negative-subsampling correction. Both use identical selector scores, tie breaking, folds and budgets. The detection-rule output generated during calibration is not used for selection. Comparator SVD and weighted kNN retain their original inductive fits; MAGIC and scVI retain their original standard all-cell fits.','',
'Three folds per set; main 42,665 cells/32 donors, sex 21,635 cells/16 prepared donors (14 analyzed after the original sex exclusions), Hao 18,828 cells/8 donors. Deploy masks training cells only; masked recovery masks all cells. Budgets remain global, per donor and per gene; 1%, 5%, 10% for biology, 1–10% for masked F1. Every original evaluator runs unchanged through a path-routing wrapper.','',
'Intervals are original 95% percentile intervals from 2,000 draws, with the original seed and donor ordering. Biological endpoints stratify donors by condition or sex; protein resamples donors without strata. The original paired masked-F1 evaluator uses unstratified donor draws, which are preserved here. The disease null uses the same 200 balanced donor-label permutations and original permutation bootstrap. No tuning was done.','',
'Numbers below show six decimals; full-precision values, every budget, all genes, cell groups, comparators and existing intervals are side by side in `comparisons/` (see `comparisons/coverage.csv`). Brackets denote 95% intervals. Rates, F1, AP and relative changes are fractions unless marked as percentages. Counts and clustering summaries have no interval in the original evaluator; they remain descriptive. AP intervals extend the original exact fold AP with the same 2,000 donor draws used for paired F1, without reranking; every AP point is checked against its saved selector report to 1e-12. The source README labels Q1–Q7 and has no separate Q8: its masked-recovery section plus the requested AP are reported as Q8.','',
'**Verification.** '+(O/'verification/PASSED').read_text().strip()+'. Every score-derived fill array was replayed exactly, and every original CSV endpoint was compared at tolerance 1e-10 before releasing new evaluations. Details: `verification/checks.json`.','']
def table(title,entries):
    lines.extend([f'**{title}**','', '| Endpoint | Original inductive | Transductive conditional | Transductive detection-weighted |','|---|---:|---:|---:|'])
    for name,values in entries: lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines.append('')
def metric(rel,metric,ci=None,scale=1,**filters):
    return [value(r[metric],r[ci] if ci else None,scale) for r in row(rel,**filters)]
entries=[]
for gene in ('XIST','RPS4Y1'):
    entries.append((gene+' AUROC',metric('q1_zeros_sex/per_gene.csv','auroc_paper','auroc_paper_ci',gene=gene)))
    entries.append((gene+' depth-stratified AUROC',metric('q1_zeros_sex/per_gene.csv','auroc_paper_depth_stratified','auroc_paper_depth_stratified_ci',gene=gene)))
    for kind in ('likely_dropout','biological'):
        entries.append((gene+' '+kind.replace('_',' ')+' fill, global 10%',metric('q1_zeros_sex/per_gene.csv',f'fill_{kind}_10pct',f'fill_{kind}_10pct_ci',gene=gene)))
for s,f in [('sex','sex|paper'),('main','lineage|paper')]:
    items=[json.loads((c/f'results/q1_zeros_{s}/summary.json').read_text())['pooled'][f]['Safe Fusion'] for c in CASES.values()]
    entries.append((s+' pooled AUROC',[value(r['auroc'],r['ci']) for r in items]))
table('Q1. Sex-linked and lineage zeros',entries)
entries=[]
for gene in ('TLR5','FCRL3'):
    for budget in ('global','donor','gene'):
        for cells in ('Treg','other cells'):
            entries.append((f'{gene}, {cells}, {budget} 10% fills',metric('focus_fills/focus_fill_counts.csv','filled',set='main',application='deploy',gene=gene,cells=cells,budget=budget,fill_pct=10)))
    entries.append((gene+' Treg per-gene inserted mean log1p',metric('focus_fills/focus_fill_counts.csv','mean_log1p_inserted',set='main',application='deploy',gene=gene,cells='Treg',budget='gene',fill_pct=10)))
    entries.append((gene+' Treg-positive increase, SLE minus healthy',metric('deploy_main/q2_treg_positive.csv','increase_SLE_minus_healthy','increase_SLE_minus_healthy_ci',gene=gene,budget='gene',fill_pct=10)))
    entries.append((gene+' raw donor expression vs fill-rate Spearman',metric('deploy_main/q2_fill_concentration.csv','spearman_raw_treg_expression_vs_treg_fill_rate','ci',gene=gene,budget='gene',fill_pct=10)))
table('Q2. Focus-gene fills and Treg detection',entries)
entries=[]
for budget in ('global','gene'):
    for level in (1,5,10):
        entries.append((f'Treg-call relative change, {budget} {level}%',metric('q3_clustering/treg_calls_summary.csv','relative_change_all','relative_change_all_ci',budget=budget,fill_pct=level)))
    for key in ('treg_calls_after_all','into_treg_all','out_of_treg_all'):
        entries.append((f'{key}, {budget} 10%',metric('q3_clustering/treg_calls_summary.csv',key,budget=budget,fill_pct=10)))
    for key in ('ari','nmi','treg_subcluster_ari','treg_cluster_size'):
        entries.append((f'{key}, {budget} 10%',metric('q3_clustering/clustering_mean_over_folds.csv',key,budget=budget,fill_pct=10)))
    entries.append((f'FCRL3 maximum subcluster detection, {budget} 10%',metric('q3_clustering/treg_subclusters_summary.csv','max_detection_mean',gene='FCRL3',budget=budget,fill_pct=10)))
table('Q3. Treg calls and clustering',entries)
entries=[]
for cells in ('All cells','Treg'):
    for quantity in ('total UMI (all genes)','zero fraction in fold panel'):
        entries.append((cells+' '+quantity+', SLE minus healthy',metric('deploy_main/q4_library_zero_fraction.csv','SLE_minus_healthy','SLE_minus_healthy_ci',cells=cells,quantity=quantity)))
for budget in ('global','donor'):
    entries.append((f'Treg fill-rate difference, {budget} 10%',metric('deploy_main/q4_fill_rates.csv','SLE_minus_healthy','SLE_minus_healthy_ci',budget=budget,fill_pct=10,genes='all genes',cells='Treg')))
    entries.append((f'Treg pseudobulk effect r, {budget} 10%',metric('deploy_main/q4_effects.csv','pearson_effects','pearson_effects_ci',budget=budget,fill_pct=10,cells='Treg')))
    entries.append((f'Treg pseudobulk slope, {budget} 10%',metric('deploy_main/q4_effects.csv','slope_filled_on_raw','slope_ci',budget=budget,fill_pct=10,cells='Treg')))
    for key in ('significant_raw','significant_filled','significance_changed'):
        entries.append((f'Treg DE {key}, {budget} 10%',metric('deploy_main/q4_effects.csv',key,budget=budget,fill_pct=10,cells='Treg')))
    for test in ('wilcoxon','pseudobulk'):
        entries.append((f'Treg null {test} mean DE increase, {budget} 10%',metric('deploy_main/q4_null_de.csv','mean_difference','mean_difference_ci',budget=budget,fill_pct=10,cells='Treg',test=test,labels='null')))
table('Q4. Disease effects and permutation nulls',entries)
entries=[]
for module,cells in [('interferon','All cells'),('interferon','Treg'),('treg_suppressive','Treg')]:
    for key,ci in [('per_cell_pearson_raw_vs_filled',None),('effect_raw','effect_raw_ci'),('effect_filled','effect_filled_ci'),('effect_change','effect_change_ci')]:
        entries.append((module+', '+cells+', '+key,metric('deploy_main/q5_modules.csv',key,ci,module=module,cells=cells,budget='global',fill_pct=10)))
table('Q5. Modules at global 10%',entries)
entries=[]
for gene in ('TLR5','FCRL3'):
    for key in ('spearman_raw_vs_filled','split_half_raw_A_vs_raw_B','split_half_filled_A_vs_raw_B','split_half_added_A_vs_raw_B'):
        entries.append((gene+' '+key,metric('deploy_main/q6_donor_agreement.csv',key,key+'_ci',gene=gene,budget='gene',fill_pct=10)))
table('Q6. Donor reproducibility at per-gene 10%',entries)
entries=[]
for scope in ('Treg','All cells','B'):
    for key,ci in [('spearman','spearman_ci'),('mean_auroc_q30_q70','mean_auroc_q30_q70_ci')]:
        entries.append((scope+' '+key,metric('q7_protein/fcrl3_protein_agreement.csv',key,ci,scope=scope)))
for budget in ('global','gene'):
    for level in (1,5,10):
        entries.append((f'Treg fills, {budget} {level}%',metric('q7_protein/fcrl3_protein_agreement.csv',f'{budget}_filled_{level}pct',scope='Treg')))
    entries.append((f'Treg protein enrichment, {budget} 10%',metric('q7_protein/fcrl3_protein_agreement.csv',f'{budget}_protein_enrichment_10pct',f'{budget}_protein_enrichment_10pct_ci',scope='Treg')))
entries.append(('Treg detected RNA/protein positive-control Spearman',metric('q7_protein/fcrl3_positive_control.csv','spearman_rna_vs_protein','ci',scope='Treg')))
table('Q7. Hao FcRL3 protein',entries)
entries=[]
for name in dict.fromkeys(r['metric'] for r in f1_rows):
    entries.append((name,[value(r['estimate'],[r['lower'],r['upper']]) for r in f1_rows if r['metric']==name]))
for fold in range(3):
    if (COMP/'masked_ap_intervals.csv').exists():
        block=pd.read_csv(COMP/'masked_ap_intervals.csv').query('fold == @fold').set_index('variant')
        values=[value(block.loc[label,'average_precision'],[block.loc[label,'lower'],block.loc[label,'upper']]) for label in CASES]
    else:
        values=[value(r['average_precision']) for r in ap if r['fold']==fold]
    entries.append((f'AP, fold {fold}',values))
for comp in ('SVD','Weighted kNN','MAGIC','scVI'):
    rs=row('masked/paired_bootstrap.csv',comparator=comp,statistic='mean_difference_1_to_10')
    entries.append(('Mean F1 minus '+comp+', percentage points',[value(r.estimate_pp,[r.lower_pp,r.upper_pp]) for r in rs]))
for gene in ('TLR5','FCRL3'):
    for cells in ('Treg','non-Treg'):
        entries.append((gene+' '+cells+' per-gene 10% masked F1',metric('masked/focus_gene_metrics.csv','f1','f1_ci',gene=gene,cells=cells,budget='gene',fill_pct=10)))
table('Q8 / masked recovery and AP',entries)
lines.extend(['**Outputs and scope.** `conditional/` and `conditional_detection/` contain the three prepared-set views, held-out fill arrays and all seven evaluator outputs under `results/`. `conditional/<set>/fold_<k>/<app>/teachers/`, `methods/safe_fusion/` and `selector/` contain new teacher/value contracts, exact selector scores and saved isotonic knots. Detection-weighted inserted values live in `conditional_detection/.../fills/Safe_Fusion.npz`; its `methods/safe_fusion` symlink deliberately identifies the shared conditional base value. All comparisons use its detection-weighted fill arrays. `verification/` contains the inductive replay. `code/`, `logs/` and `runtime/` contain scripts, logs and Slurm records.','',
'**Limitations and failures.** No separate Q8 endpoint is defined by the original README. No TLR5 protein endpoint can be computed from Hao, and its FcRL3 check is in healthy donors, not SLE. Sparse focus-gene masked positives and the original cell cap, gene panels and annotation limitations remain. Original fixed-count and clustering summaries have no bootstrap interval. AP intervals are newly computed for both original and transductive scores with matched donor draws; finite-draw counts and reproduction errors are recorded in `comparisons/masked_ap_intervals.csv`. See `runtime/failures.txt` for execution failures, if any.','',
'**Reproduce from the repository root.** Existing prepared inputs and original comparator fits are prerequisites; scVI is reused, so no new GPU fit is needed. The launcher submits eight-thread jobs on `cs` with the required account.','',
'```bash',
'bash scripts/analyses/sle_transductive/reproduce.sh',
'```','',
'The launcher runs `setup.py`; twelve original-score fill replays; all seven original evaluations; `validate.py`; twelve transductive chains; all seven evaluations for each value; the matched AP bootstrap; `audit.py`; `check_comparators.py`; and `report.py`. Individual stage command:','',
'```bash',
'O=artifacts/paper_evidence/review_round4/sle_transductive',
'sbatch -A torch_pr_634_general -p cs --array=0-11 --export=ALL,STAGE=chain "scripts/analyses/sle_transductive/pipeline.sh"',
'sbatch -A torch_pr_634_general -p cs --array=0-6 --mem=128G --export=ALL,STAGE=evaluate,VARIANT=conditional "scripts/analyses/sle_transductive/pipeline.sh"',
'sbatch -A torch_pr_634_general -p cs --array=0-6 --mem=128G --export=ALL,STAGE=evaluate,VARIANT=conditional_detection "scripts/analyses/sle_transductive/pipeline.sh"',
'# Run evaluations only after chain completion and verification/PASSED.',
'sbatch -A torch_pr_634_general -p cs --cpus-per-task=2 --mem=8G --time=00:30:00 --wrap=".venv-sle/bin/python scripts/analyses/sle_transductive/report.py"',
'```',''])
calls=row('q3_clustering/treg_calls_summary.csv',budget='gene',fill_pct=10)
headline='At per-gene 10%, Treg-call expansion is '+', '.join(label+' '+value(r['relative_change_all'],r['relative_change_all_ci'],100)+'%' for label,r in zip(CASES,calls))+'. Detection weighting reduces the expansion but does not remove it. Ranking-only endpoints are identical between the two transductive values.'
lines[2:2]=[headline,'']
lines.extend(['**Execution.** Twelve transductive chains and fourteen new evaluation tasks completed; twelve original-score fill replays, seven original evaluations, 97 replay checks, twelve unit audits and six exact-AP bootstrap computations also completed. All four comparator fill arrays matched the original arrays exactly in every unit; 734 comparator/unfilled endpoint checks also passed (`comparator_endpoint_checks.json`). The 25 side-by-side tables contain 14,133 rows. No new scVI fit was needed.','',
'One slow value fit was cancelled after 41 minutes and rerun unchanged; the replacement completed in 55 minutes 32 seconds. One report-only failure misread the literal CSV label `null`; parsing was corrected without rerunning scientific analyses. Pending dependents were replaced, not duplicated. Scheduler dependency updates, requeue and node exclusion were unavailable; successful jobs used the original cs request. Details: `runtime/failures.txt`, `runtime/jobs.tsv`, `runtime/sacct_records.txt`. No scientific endpoint remains unfinished.','',
'Detection weighting gives exactly zero to 19,017 selected entries at the per-gene 10% budget across all twelve units; no selected entry becomes zero at global or per-donor 10%. Evaluators retain the original definition of a fill as selected with a positive inserted value. See `audit.json`.',''])
(O/'README.md').write_text('\n'.join(lines))
print(json.dumps({'compared_tables':len(coverage),'compared_rows':sum(r['rows'] for r in coverage),'report':str(O/'README.md')}))
