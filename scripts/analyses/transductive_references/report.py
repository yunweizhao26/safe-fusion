import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()/'scripts'))
from evaluate_knockdown_zero_analyses import SCOPES
O=Path('artifacts/paper_evidence/review_round4/transductive_references'); OLD=Path('artifacts/paper_evidence/review_round2')
def js(p): return json.loads(Path(p).read_text())
def ci(values,scale=1):
 return f'{values[0]*scale:.4f} [{values[1]*scale:.4f}, {values[2]*scale:.4f}]'
def cell(row,estimate='estimate',scale=1): return ci([row[estimate],row.ci_low,row.ci_high],scale)
def table(head,rows): return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def mix_intervals(root):
 targets=pd.read_csv(root/'targets.csv'); targets=targets[targets.dataset=='papalexi_eccite'].gene.tolist()
 frame=pd.read_csv(root/'mixscape.csv'); draws=np.random.default_rng([1729,SCOPES.index('papalexi_eccite')]).integers(0,len(targets),size=(2000,len(targets)))
 results=[]
 for (method,kind),part in frame.groupby(['method','mixscape_class'],sort=False):
  part=part.set_index('gene').reindex(targets).fillna(0); n=part.n_zeros.to_numpy(); denom=n[draws].sum(axis=1)
  for pct in (1,5,10):
   filled=part[f'filled_{pct}pct'].to_numpy(); boot=np.divide(filled[draws].sum(axis=1),denom,out=np.full(len(denom),np.nan),where=denom>0)
   lo,hi=np.nanquantile(boot,[.025,.975]); results.append(dict(method=method,kind=kind,pct=pct,n_zeros=int(n.sum()),estimate=filled.sum()/n.sum(),ci_low=lo,ci_high=hi))
 return pd.DataFrame(results)
lines=['# Transductive Safe Fusion: external-reference analyses','',
'Output directory: `artifacts/paper_evidence/review_round4/transductive_references/`. Paths below are relative to it.','',
'Fits and available-gene evaluations completed on the Torch cluster. Full-cohort pancreas XIST is unavailable: XIST is absent from rebuilt folds 1 and 2; its reported estimates cover fold 0 only. Entries are estimates [95% bootstrap intervals]. AUROC and correlations use the 0–1 or −1–1 scale; F1 and fill rates are percentages. F1 is masked-entry recovery, not biological accuracy. Fold changes are log2 ratios of mean counts per ten thousand (pseudocount 0.001).', '',
'All-cell teachers see only the supplied masked/deployment RNA counts. Gene median, SVD and weighted kNN use transductive fits; standard MAGIC and scVI are converted to the input count scale. The fused value and selector fit only the production training cells. Settings, seed 1729, selection rules and bootstrap draws remain fixed. Label-aware screens restrict kNN to the same perturbation and give scVI the perturbation covariate.', '',
'Tissue inputs are the rebuilt pancreas folds and all three 34-donor colon folds. Norman always uses the rebuilt benchmark. Deployment inputs restore recorded counts only in test cells. Crossmodal Papalexi uses the existing RNA input and the same 655 CD274 recorded zeros. Comparator scores remain the original scores; only Safe Fusion changes.', '',
'The donor variant uses same-donor all-cell kNN, donor-covariate scVI, and two donor gene features from observed counts of all same-donor cells. This requires the additive `--condition-feature-source all_cells` option: the production fitting-only features are undefined for held-out donors. Selector fitting labels and fitting cells remain unchanged. The default remains `fitting`. No test-result tuning was performed.', '',
'Validation: all 28 saved inductive evaluator CSV tables reproduce to absolute/relative tolerance 1e-12. See `verification/comparison.json`. The rebuilt tissue inductive F1 counts are checked against the saved unit counts. New tissue deployment inductive fits provide matching comparators; the older 9-donor colon results are not substituted. All selectors run in partition `cs`; scVI uses L40S. Existing ALRA results are reused unchanged.', '',
'## 1. Perturbation zeros and masked F1','',
'Unadjusted and five-library-stratum AUROC use the unchanged eligibility rule and 2,000 target bootstrap draws. Zero-analysis rows below use label-free Safe Fusion; complete label-aware, depth-matched and depth-normalized results remain in the output tables. F1 averages the ten fill fractions 1%–10%, with 2,000 perturbation-label draws. Mixscape intervals use those same eligible-target draws and pooled counts.', '']
old=js(OLD/'knockdown/evaluation/report.json'); new=js(O/'knockdown/evaluation_matched/report.json'); rows=[]
names={'adamson_crispri':'Adamson','papalexi_eccite':'Papalexi','norman_crispra':'Norman rebuilt','dixit_ko':'Dixit'}
for screen in ('adamson_crispri','papalexi_eccite','norman_crispra'):
 a=old['screens'][screen]; b=new['screens'][screen]
 for score in ('fill_order','continuous'):
  for adjustment in ('none','depth_strata'):
   rows.append([names[screen],f'{score} AUROC / {adjustment}',ci(a['auroc']['safe_fusion|'+score][adjustment]),ci(b['auroc']['safe_fusion|'+score][adjustment])])
 for pct in (1,5,10):
  rows.append([names[screen],f'log2 fold change after {pct}% fill',ci(a['knockdown_effect_shift'][f'safe_fusion|{pct}']['log2fc_filled']),ci(b['knockdown_effect_shift'][f'safe_fusion|{pct}']['log2fc_filled'])])
mi=mix_intervals(OLD/'knockdown/evaluation'); mt=mix_intervals(O/'knockdown/evaluation_matched'); mi.to_csv(O/'knockdown/mixscape_inductive_intervals.csv',index=False); mt.to_csv(O/'knockdown/mixscape_transductive_intervals.csv',index=False)
for kind in ('KO','NP'):
 for pct in (1,5,10):
  a=mi[(mi.method=='safe_fusion')&(mi.kind==kind)&(mi.pct==pct)].iloc[0]; b=mt[(mt.method=='safe_fusion')&(mt.kind==kind)&(mt.pct==pct)].iloc[0]
  rows.append(['Papalexi Mixscape',f'{kind} fill rate at {pct}% (n={int(a.n_zeros)})',cell(a,scale=100),cell(b,scale=100)])
a=js(OLD/'knockdown/masked_f1/masked_f1_by_screen.json'); b=js(O/'knockdown/masked_f1/masked_f1_by_screen.json')
for screen in ('adamson_crispri','papalexi_eccite','norman_crispra','dixit_ko'):
 for method,label in [('safe_fusion','no labels'),('safe_fusion_condition','perturbation labels')]: rows.append([names[screen],f'mean F1 %, {label}',ci(a[screen]['mean_f1_percent'][method]),ci(b[screen]['mean_f1_percent'][method])])
lines += [table(['Screen','Metric','Inductive','Transductive'],rows),'',
'Outputs: `knockdown/evaluation_matched/` (AUROC, effects, fills, targets, Mixscape); `knockdown/masked_f1/` (every fill fraction, per-label counts, intervals); `knockdown/mixscape_*_intervals.csv`. Comparators are unchanged through read-only `knockdown/evaluation_inputs/` views.','',
'## 2. Sex-linked zeros and the donor variant','',
'Pancreas XIST covers only fold 0 (9 donors); RPS4Y1 covers all 24 pancreas donors, and both genes cover all 34 colon donors. Sex is assigned from the common fold-0 recorded reference, which contains both genes for every cell. The original 24/34-donor bootstrap draws are retained; absent genes contribute no observations. See `gene_availability.csv`. Fill percentages are global budgets over test-cell zeros, not gene-specific budgets. Sex is inferred by the original rule: pooled donor XIST counts exceed RPS4Y1 counts for female donors, otherwise male. These are biological proxy labels. Intervals resample donors within inferred sex, 2,000 draws, seed 1729. Expressing-sex recorded zeros are positive; other-sex zeros are negative. F1 uses the original fold-prefixed donor order and paired 2,000-draw donor bootstrap.','']
rows=[]
for tissue in ('pancreas','colon'):
 root=O/'sex_zeros/evaluation'/tissue; auc=pd.read_csv(root/'auroc.csv'); fill=pd.read_csv(root/'fill_rates.csv')
 methods=['safe_fusion_inductive','safe_fusion','safe_fusion_donor']
 for gene in ('RPS4Y1','XIST'):
  rows.append([tissue,gene+' AUROC',*[cell(auc[(auc.gene==gene)&(auc.method==m)].iloc[0],'auroc') for m in methods]])
  for frac in (.01,.05,.1):
   for kind,label in [('recorded zero, expressing sex','expressing sex'),('absent, other sex','other sex')]:
    rows.append([tissue,f'{gene} {label}, {frac*100:g}% fill',*[cell(fill[(fill.gene==gene)&(fill.method==m)&(fill.fraction==frac)&(fill.kind==kind)].iloc[0],'fill_rate',100) for m in methods]])
f1=js(O/'sex_zeros/masked_f1/masked_f1_report.json')
for tissue in ('Pancreas','Colon'): rows.append([tissue,'mean masked F1 % (1%–10%)',*[ci(f1[tissue]['mean_f1_percent'][m]) for m in methods]])
lines += [table(['Tissue','Metric','Inductive','Transductive','Transductive + donor'],rows),'',
'Paired donor-minus-label-free F1 differences (percentage points): '+ '; '.join(t+': '+ci(f1[t]['donor_minus_labelfree_pp']) for t in ('Pancreas','Colon'))+'.','',
'Outputs: `sex_zeros/evaluation/{pancreas,colon}/` (AUROC, fill rates, counts, inferred sex and feature checks); `sex_zeros/masked_f1/` (all ten fractions and paired intervals). Correct deployment inputs and transductive fits: `sex_zeros/deployment_rebuilt/`; matching inductive fits: `sex_zeros/inductive_rebuilt/`.','',
'## 3. PD-L1 among 655 CD274 RNA zeros','',
'Primary protein scale is CLR. Mean AUROC averages 41 thresholds from the 30th to 70th percentile. Intervals below use 2,000 target-cluster draws. Within-target correlations weight eligible targets by cell count; partial correlations regress ranked variables on the listed controls. Raw-antibody-count and cell-within-replicate intervals are retained in the full tables.','']
A=OLD/'protein/evaluation'; B=O/'protein_cs/evaluation_within_state'
def protein_rows(root,file):
 x=pd.read_csv(root/file).fillna({'covariates':''})
 return x[(x.protein=='PD-L1')&(x.scale=='clr')&(x.cells=='recorded_rna_zero')&(x.scope=='all')&(x.resampling=='target_clusters')]
a=protein_rows(A,'association.csv'); b=protein_rows(B,'association.csv'); rows=[]
for stat,cov,label in [('pooled_spearman','','pooled Spearman'),('within_target_spearman','','within-target Spearman'),('pooled_mean_auroc','','mean AUROC'),('partial_spearman','target_ifng_library','partial: target + IFNG + library'),('partial_spearman','baseline_ifng_library','partial: target baseline + IFNG + library'),('partial_spearman','target_ifng_library_replicate','partial: target + IFNG + library + replicate'),('partial_spearman','target_ifng_library_cellscore','partial: target + IFNG + library + cell score')]:
 rows.append([label,cell(a[(a.ranking=='safe_fusion')&(a.statistic==stat)&(a.covariates==cov)].iloc[0]),cell(b[(b.ranking=='safe_fusion')&(b.statistic==stat)&(b.covariates==cov)].iloc[0])])
a=protein_rows(A,'paired_differences.csv'); b=protein_rows(B,'paired_differences.csv')
for method,label in [('svd','SVD'),('weighted_knn','kNN'),('scvi','scVI'),('target_baseline','target baseline'),('ifng_score','IFNG state'),('library_size','library size'),('cell_zero_score','cell-zero score')]:
 for stat,short in [('pooled_spearman','Spearman'),('pooled_mean_auroc','AUROC')]: rows.append([f'Safe Fusion − {label}, {short}',cell(a[(a.comparator==method)&(a.statistic==stat)].iloc[0],'safe_fusion_minus_comparator'),cell(b[(b.comparator==method)&(b.statistic==stat)].iloc[0],'safe_fusion_minus_comparator')])
lines += [table(['Metric','Inductive','Transductive'],rows),'',
'Outputs: `protein_cs/evaluation_within_state/association.csv`, `paired_differences.csv`, `pdl1_pooled_rankings.csv`, and `global_fill.csv`. Figure 2 panels: `protein_cs/figures/pdl1_range_validation.{pdf,png}`. SVD, kNN, scVI and state controls retain their original scores; the all-other-gene cell-score control also retains the original scores.','',
'## Failures and reproduction','',
"Recovered failures: the Papalexi label-aware value fit timed out after two hours; the inductive pancreas_2 value fit timed out after three hours. Unchanged six-hour retries finished in 2:36 and 2:30, respectively. The initial knockdown evaluator lacked an unchanged Dixit input link; this was restored. Failed logs remain in `logs/`.", '',
"Excluded prior partial results used older pancreas panels, the wrong pancreas truth file, a protein selector in `cpu_short`, or changed protein comparator scores. These artifacts remain preserved; corrected outputs are reported above. Missing round-2 README files were replaced as provenance by saved results and their launchers. Slurm rejected time/partition updates and node exclusion; pending GPU fits were preserved without duplicates.", '',
"Remaining limitation: full-cohort pancreas XIST cannot be evaluated without adding a gene to two fixed panels. No other requested computation remains pending. The evaluator records this absence rather than treating an absent panel gene as a recorded zero.", '',
'Exact refit commands and stage order: `bash scripts/analyses/transductive_references/reproduce.sh` (run in an isolated restoration of the workspace and input links; the script writes the same result paths). Verification: `sbatch -A torch_pr_634_general -p cs -c 2 --mem=8G --wrap=".venv/bin/python scripts/analyses/transductive_references/verify_results.py"`. Report regeneration: `.venv/bin/python scripts/analyses/transductive_references/report.py`. Job states and logs are under `logs/`, `queue.tsv`, and `job_accounting.txt`. Key source snapshots and hashes are in `code_snapshot/`. Existing paper files, docs and root README were not edited; no commit or push was made.','']
(O/'README.md').write_text('\n'.join(lines))
print(O/'README.md')
