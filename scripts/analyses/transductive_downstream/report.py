from pipeline import *
import numpy as np, pandas as pd
out=R/'summary'; out.mkdir(exist_ok=True)
labels={'safe_fusion':'Inductive SF','transductive':'Transductive conditional','detection':'Transductive detection-weighted','svd':'SVD','magic':'MAGIC','scvi':'scVI'}
disease_names={'Safe Fusion':'safe_fusion','Transductive conditional':'transductive','Transductive detection':'detection','SVD':'svd','MAGIC':'magic','scVI':'scvi'}
endpoint_labels={'canonical_marker_pr_auc':'marker AUPRC','ectopic_marker_fill_rate':'off-target fill','cell_identity_macro_f1':'mapping F1','annotation_ari':'clustering ARI','dynamic_gene_spearman':'dynamics','stage_rank_mae':'stage error','edge_pr_auc_q10':'edge AUPRC'}
lines=['# Transductive downstream analyses','', 'Report status and limitations appear below. Values are estimates [95% interval]. README rounds to six significant digits; linked CSVs retain exact evaluator values. Table 3 uses filled-minus-unfilled changes on recorded counts. Only the thinning analysis measures error against an independent unthinned reference. Each table cell lists 1% / 5% / 10% in that order unless stated otherwise. Probabilities and AUPRC remain on the 0–1 scale.','']
lines += ['Transductive teachers use the masked or recorded counts of all cells: gene median, SVD, weighted kNN, standard MAGIC and standard scVI on the count scale. The fused value and selector retain their production training cells and settings. Conditional means the production fused value. Detection-weighted multiplies that value by d = p / (p + 0.1 × (1 − p)), where p comes from five-fold isotonic calibration on the selector-fitting cells. Selected entries and fill budgets are unchanged.','']
def fmt(v): return 'NA' if not np.isfinite(v) else f'{v:.6g}'
def value(v,lo,hi): return f'{fmt(v)} [{fmt(lo)}, {fmt(hi)}]'
def table(frame,keycols,val='difference',low='ci_low',high='ci_high',method='source',fraction='pct'):
 if frame.empty: lines.extend(['No completed results.','']); return
 assert not frame.duplicated(keycols+[method,fraction]).any(), 'Duplicate table entries'
 lines.append('| Endpoint | '+' | '.join(labels.values())+' |')
 lines.append('|---|'+'---|'*len(labels))
 for key,g in frame.groupby(keycols,sort=False,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  cells=[]
  for name in labels:
   f=g[g[method]==name].sort_values(fraction)
   cells.append(' / '.join(value(float(r[val]),float(r[low]),float(r[high]))+str(r.get('extra','')) for _,r in f.iterrows()) or 'pending')
  lines.append('| '+'; '.join(str(k) for k in key if str(k))+' | '+' | '.join(cells)+' |')
 lines.append('')
checks=[pd.read_csv(p) for p in (R/'reproduction').glob('*/check.csv')]
if checks:
 c=pd.concat(checks,ignore_index=True); c.to_csv(out/'reproduction.csv',index=False)
 lines += [f"Reproduction: {c.dataset.nunique()}/4 biological evaluators checked; {int(c.n_rows.sum())} estimate/interval comparisons; maximum absolute difference {c.max_abs_difference.max():g}; {'all passed' if c.passed.all() else 'FAILURES'}. See `summary/reproduction.csv`.",'']
ready=[d for d in D if (Path(d['dir'])/'READY').exists()]
if (R/'audit.csv').exists():
 a=pd.read_csv(R/'audit.csv'); covered=len(a[['design','unit']].drop_duplicates())
 lines += [f"Artifact audit: {covered}/{len(ready)} completed chains covered; {int(a.passed.sum())}/{len(a)} recorded checks passed. See `audit.csv`.",'']
lines += [f"Completed model chains: {len(ready)}/{len(D)}. Every selector uses partition cs. New scVI fits use L40S. Base seed 1729 (original fold and endpoint offsets retained); fractions 1%, 5%, 10%; 2000 bootstrap draws; five calibration folds; up to 200 label permutations. No tuning.",'']
lines += ['Inputs: rebuilt pancreas folds; all three colon folds covering 34 donors; the zebrafish trajectory benchmark; rebuilt Norman CRISPRa. The production Norman file is never used. All paths below are relative to `artifacts/paper_evidence/review_round4/transductive_downstream/`.','']
lines += ['The rebuilt pancreas and colon folds select different genes. Donor metrics are evaluated within each fold and pooled with the original bootstrap. Gene-wise disease effects, modules and annotation checks are reported per fold. **These fold-level analyses do not replace the originally pooled disease endpoints.** A pooled gene-wise analysis requires an agreed evaluation gene universe; none was invented. See `DESIGN.md`.','']
changes=pd.read_csv(out/'endpoint_changes.csv') if (out/'endpoint_changes.csv').exists() else pd.DataFrame()
if len(changes):
 changes[['source','pct']]=changes.method.str.extract(r'^(.*)_(\d+)pct$'); changes['pct']=changes.pct.astype(int)
 changes['endpoint']=changes.metric.map(endpoint_labels); changes=changes[changes.endpoint.notna()]
lines += ['Table 3 — recorded counts. Exact results: `summary/endpoint_changes.csv`; per-unit artifacts: `deployment/evaluation/`; fold-specific disease correlations: `summary/disease_fold_endpoints.csv`.','']
recorded=changes[changes.design=='deployment'].copy() if len(changes) else changes
fold_path=out/'disease_fold_endpoints.csv'
if fold_path.exists():
 fd=pd.read_csv(fold_path)
 fd=fd[(fd.design=='deployment')&(fd.reference=='corrupted_raw')&(fd.metric=='disease_logfc_spearman')&fd.method.str.match(r'^(safe_fusion|transductive|detection|svd|magic|scvi)_(1|5|10)pct$')].copy()
 fd[['source','pct']]=fd.method.str.extract(r'^(.*)_(\d+)pct$');fd['pct']=fd.pct.astype(int)
 fd['dataset']=fd.unit;fd['endpoint']='disease correlation '+fd.contrast
 recorded=pd.concat([recorded,fd],ignore_index=True)
table(recorded,['dataset','endpoint'])
lines += ['Table S17 decomposition — main biological endpoints at 10% fills, change from masked input; full, masked-positive-only and recorded-zero-only fills. Absolute endpoint estimates are also in each `masked/evaluation/*/*/bootstrap_summary.parquet`. Exact changes: `summary/endpoint_changes.csv`; exact absolute values: `summary/s17_absolute.csv`; fill counts: `summary/decomposition_counts.csv`.','']
table(changes[(changes.design=='masked')&(changes.pct==10)&changes.metric.isin(['canonical_marker_pr_auc','dynamic_gene_spearman','edge_pr_auc_q10'])] if len(changes) else changes,['dataset','endpoint','part'])

disease=[]; concise=[]
for d in D:
 if d['design']!='deployment' or d['key'].split('_')[0] not in ['pancreas','colon']:continue
 key=d['key'];tissue=key.split('_')[0]; base=R/'disease'/key
 for typ,name in [('disease_effects','observed_summary.csv'),('disease_effects','permutation_null.csv'),('annotation','overall.csv')]:
  path=base/'deployment'/typ/tissue/name
  if not path.exists():continue
  f=pd.read_csv(path).assign(unit=key,analysis=typ,artifact=name); disease.append(f)
  f=f[np.isclose(f.fraction,.1)]
  for _,r in f.iterrows():
   source=disease_names.get(r['method'])
   if not source:continue
   metrics=[('slope','effect_slope','effect_slope_ci_low','effect_slope_ci_high')] if name=='observed_summary.csv' else [('permutation false-discovery change','false_discoveries_difference_mean','false_discoveries_difference_low','false_discoveries_difference_high')] if name=='permutation_null.csv' else [('mapping changed share','changed_share','changed_share_ci_low','changed_share_ci_high')]
   for title,val,lo,hi in metrics:
    extra=f"; D {int(r.n_discoveries_unfilled)}→{int(r.n_discoveries_filled)}" if name=='observed_summary.csv' else ''
    concise.append(dict(unit=key,endpoint=title,contrast=r.get('contrast',''),test=r.get('test',''),source=source,pct=10,difference=r[val],ci_low=r[lo],ci_high=r[hi],extra=extra))
 module=base/'module_scores'/tissue/'disease_effect.csv'
 if module.exists():
  f=pd.read_csv(module).assign(unit=key,analysis='module_scores',artifact='disease_effect.csv');disease.append(f)
  for _,r in f[np.isclose(f.fraction,.1)&f.module.str.startswith('disease')].iterrows():
   concise.append(dict(unit=key,endpoint=r.module,contrast=r.contrast,test='',source=disease_names[r['method']],pct=10,difference=r.effect_change,ci_low=r.effect_change_ci_low,ci_high=r.effect_change_ci_high,extra=''))
if disease: pd.concat(disease,ignore_index=True).to_csv(out/'disease_all.csv',index=False)
lines += ['Disease and annotation — fold-level results at 10%. D gives unfilled→filled discoveries (FDR 0.05); counts have no confidence interval in the unchanged evaluator. Permutation intervals describe the null distribution, not uncertainty in its mean. Full 1%, 5%, 10% results, permutation means, all modules and intervals: `summary/disease_all.csv`, `disease/<unit>/`.','']
table(pd.DataFrame(concise),['unit','contrast','test','endpoint'])
lines += ['Independent reference — 10% fills; reduction in absolute error against unthinned truth; positive values indicate improvement. Exact values and intervals: `summary/thinning_reference.csv`, `summary/thinning_fold_slopes.csv`, `summary/disease_fold_endpoints.csv`; per-fold disease slopes: `thinning/disease_slopes/`.','']
f=pd.read_csv(out/'thinning_reference.csv') if (out/'thinning_reference.csv').exists() else pd.DataFrame()
if len(f): f=f.rename(columns={'method':'source','fill_pct':'pct'})
slopes=out/'thinning_fold_slopes.csv'
if slopes.exists():
 sf=pd.read_csv(slopes).rename(columns={'method':'source','fill_pct':'pct'})
 sf['dataset']=sf.unit
 f=pd.concat([f,sf],ignore_index=True)
if fold_path.exists():
 fd=pd.read_csv(fold_path)
 fd=fd[(fd.design=='thinning')&(fd.reference=='corrupted_raw')&(fd.metric=='disease_logfc_spearman')&fd.method.str.match(r'^(safe_fusion|transductive|detection|svd|magic|scvi)_(1|5|10)pct$')].copy()
 fd[['source','pct']]=fd.method.str.extract(r'^(.*)_(\d+)pct$');fd['pct']=fd.pct.astype(int)
 fd['dataset']=fd.unit;fd['endpoint']='Disease correlation '+fd.contrast;fd['error_reduction']=fd.difference
 f=pd.concat([f,fd],ignore_index=True)
table(f[f.pct==10] if len(f) else f,['dataset','endpoint'],val='error_reduction')
null=[]
for design in ['deployment','null']:
 p=R/'correlation'/design/'counts/summary.csv'
 if not p.exists():continue
 f=pd.read_csv(p)
 for _,r in f[np.isclose(f.fraction,.1)&f.scope.isin(['colon','pancreas'])].iterrows():
  if r['method'] not in labels:continue
  for title,stat in [('correlated-pair change','n_correlated'),('mean absolute correlation change','mean_abs_rho')]:
   null.append(dict(design=design,tissue=r.scope,endpoint=title,source=r['method'],pct=10,difference=r[stat+'_difference'],ci_low=r[stat+'_difference_low'],ci_high=r[stat+'_difference_high']))
lines += ['Correlation inflation — count-scale changes at 10%. Full fractions, absolute correlations, quantiles, pair counts and 95% intervals: `correlation/{deployment,null}/{counts,log_cp10k}/summary.csv`.','']
table(pd.DataFrame(null),['design','tissue','endpoint'])
missing=[]
for d in D:
 p=Path(d['dir'])
 if not (p/'READY').exists(): missing.append(f"{d['design']}/{d['key']}: model/fill chain unfinished")
 elif d['design']!='null':
  for part in (['full','masked_only','zeros_only'] if d['design']=='masked' else ['full']):
   if not (R/d['design']/'evaluation'/d['key']/part/'paired_comparisons.parquet').exists():missing.append(f"{d['design']}/{d['key']}/{part}: evaluator unfinished")
for design in ['deployment','null']:
 for scale in ['counts','log_cp10k']:
  if not (R/'correlation'/design/scale/'summary.csv').exists():missing.append(f'{design}/{scale}: correlation evaluator unfinished')
for tissue in ['pancreas','colon']:
 for i in range(3):
  base=R/'disease'/f'{tissue}_{i}'
  for relative in [f'deployment/disease_effects/{tissue}/observed_summary.csv',f'deployment/disease_effects/{tissue}/permutation_null.csv',f'deployment/annotation/{tissue}/overall.csv',f'module_scores/{tissue}/disease_effect.csv']:
   if not (base/relative).exists():missing.append(f'{tissue}_{i}/{relative}: unfinished')
for i in range(3):
 if not (R/'thinning/disease_slopes'/f'pancreas_{i}'/'slopes.csv').exists():missing.append(f'thinning/pancreas_{i}: disease slopes unfinished')
lines += ['Failures and remaining work:','']
lines += ['- '+x for x in missing] or ['- All model and main evaluator chains completed.']
lines += ['- Pancreas fold 2 has no testable T1D contrast under the unchanged evaluator: slopes and their intervals are NA, with zero tested pairs and discoveries. No replacement interval is invented.', '- Pooled gene-wise disease endpoints remain unresolved because rebuilt folds use different genes. Fold-specific outputs are labeled explicitly.','- Shared thinning fusion jobs 18824153_17 and 18824153_19, and shared inductive selector 18823913_1, reached their two-hour limits. This task did not alter those outputs. Its failed dependencies were replaced after the main analysis submitted replacement jobs; see `thinning_dependency_repair.json`.', '- Adapter check job 18826632 initially rejected 45 extra all-NaN rows for a metric omitted by the original deployment evaluator. The revised check excludes only those undefined rows; all 315 shared donor metrics match exactly. No model values changed.', '- Pending six-hour GPU requests were cancelled and replaced by 30-minute L40S requests (18828520) after Slurm estimated starts beyond the parent deadline. Model settings are unchanged. Scheduler updates were rejected; only this task’s pending jobs were resubmitted. Historical logs are retained. No previous analysis output was overwritten.','', 'Reproduce from repository root (all real work through Slurm):','', '```bash', 'R=artifacts/paper_evidence/review_round4/transductive_downstream', '# Run after earlier downstream jobs finish. Reuse completed models.', 'bash "scripts/analyses/transductive_downstream/reproduce_all.sh"', '```','', 'Exact expanded commands and original paths are in `logs/*.out`; `manifest.json` records reused fits, `source_sha256.json` records 18 unchanged source versions, `environment.*.json` records package versions, `resubmissions.json` records replacement jobs. Resume only after existing jobs finish; `scripts/analyses/transductive_downstream/reproduce_all.sh` waits for each stage.']
(R/'README.md').write_text('\n'.join(lines)+'\n')
print('README written;',len(lines),'lines')
