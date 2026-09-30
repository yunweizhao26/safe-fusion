from pipeline import *
import numpy as np, pandas as pd
import evaluate_colon_donor_biology as col
import evaluate_pancreas_crossfit_biology as pan
import r3_downstream_thinning as thin
out=R/'summary'; out.mkdir(exist_ok=True)
for design in ['masked','deployment','thinning']:
 for tissue in ['pancreas','colon']:
  for part in (['full','masked_only','zeros_only'] if design=='masked' else ['full']):
   paths=[R/design/'evaluation'/f'{tissue}_{i}'/part/'donor_metrics.parquet' for i in range(3)]
   if not all(p.exists() for p in paths): continue
   frame=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
   module=pan if tissue=='pancreas' else col
   c='condition' if tissue=='pancreas' else 'disease'; donors=frame[['donor',c]].drop_duplicates().sort_values('donor')
   assert not donors.donor.duplicated().any()
   draws=pan.stratified_bootstrap_indices(donors,2000,1729) if tissue=='pancreas' else col.bootstrap_indices(donors.donor.to_numpy(),2000,1729)
   summary,paired=module.summarize_unit_metrics(frame,draws)
   dest=R/design/'evaluation'/tissue/part; dest.mkdir(parents=True,exist_ok=True)
   for name,data in [('donor_metrics',frame),('bootstrap_summary',summary),('paired_comparisons',paired)]: data.to_parquet(dest/f'{name}.parquet',index=False)
   clusters=[p.parent/'clustering' for p in paths]
   if all((p/'unit_seed_metrics.parquet').exists() for p in clusters) and not (dest/'clustering/report.json').exists():
    run('scripts/combine_clustering_folds.py',*sum((['--fold',p] for p in clusters),[]),'--output-dir',dest/'clustering','--bootstrap',2000,'--seed',1729)
absolute=[]
for key in ['colon','pancreas','zebrafish','norman_crispra']:
 path=R/'masked/evaluation'/key/'full/bootstrap_summary.parquet'
 if path.exists(): absolute.append(pd.read_parquet(path).assign(dataset=key))
if absolute: pd.concat(absolute,ignore_index=True).to_csv(out/'s17_absolute.csv',index=False)
counts=[]
for d in D:
 if d['design']=='masked':
  path=Path(d['dir'])/'decomposition/decomposition_counts.csv'
  if path.exists(): counts.append(pd.read_csv(path).assign(unit=d['key']))
if counts: pd.concat(counts,ignore_index=True).to_csv(out/'decomposition_counts.csv',index=False)
rows=[]
for design in ['masked','deployment','thinning']:
 for key in ['colon','pancreas','zebrafish','norman_crispra']:
  for part in (['full','masked_only','zeros_only'] if design=='masked' else ['full']):
   base=R/design/'evaluation'/key/part
   for path in [base,base/'clustering']:
    paired=path/'paired_comparisons.parquet'
    if not paired.exists(): continue
    frame=pd.read_parquet(paired)
    frame=frame[(frame.reference=='corrupted_raw') & frame.method.str.match(r'^(safe_fusion|transductive|detection|svd|magic|scvi)_(1|5|10)pct$')].copy()
    frame['dataset']=key; frame['design']=design; frame['part']=part
    rows.append(frame)
if rows: pd.concat(rows,ignore_index=True).to_csv(out/'endpoint_changes.csv',index=False)

values=[]
metrics={'colon':[('Marker AUPRC','canonical_marker_pr_auc'),('Reference mapping F1','cell_identity_macro_f1')], 'pancreas':[('Marker AUPRC','canonical_marker_pr_auc'),('Reference mapping F1','cell_identity_macro_f1')], 'zebrafish':[('Dynamics','dynamic_gene_spearman'),('Stage error','stage_rank_mae')], 'norman_crispra':[('Edge AUPRC','edge_pr_auc_q10')]}
for key, endpoints in metrics.items():
 path=R/'thinning/evaluation'/key/'full'/('donor_metrics.parquet' if key in ['colon','pancreas'] else 'unit_metrics.parquet')
 if not path.exists(): continue
 frame=pd.read_parquet(path); unit='donor' if key in ['colon','pancreas'] else 'unit'; stratum='condition' if key=='pancreas' else 'disease' if key=='colon' else None
 for endpoint,metric in endpoints:
  f=frame[frame.metric==metric]; pivot=f.pivot(index=unit,columns='method',values='value')
  reference=pivot['reference_truth']
  if endpoint=='Stage error':
   deployed=pd.read_parquet(R/'deployment/evaluation/zebrafish/full/unit_metrics.parquet')
   reference=deployed[(deployed.metric==metric)&(deployed.method=='corrupted_raw')].set_index(unit).value.reindex(pivot.index)
  strata=f.drop_duplicates(unit).set_index(unit)[stratum].reindex(pivot.index).astype(str) if stratum else pd.Series('all',index=pivot.index)
  for name in ['corrupted_raw']+[f'{n}_{p}pct' for n in NAMES for p in PCTS]:
   values.append(pd.DataFrame(dict(dataset=key,endpoint=endpoint,unit=pivot.index.astype(str),stratum=strata.to_numpy(),matrix=name,value=pivot[name].to_numpy(),unthinned=reference.to_numpy(),thinned_truth_pipeline=pivot.reference_truth.to_numpy())))
if values:
 thin.METHODS={n:n for n in NAMES}
 v=pd.concat(values,ignore_index=True); v.to_csv(out/'thinning_unit_values.csv',index=False)
 thin.summarize_units(v,2000,1729).to_csv(out/'thinning_reference.csv',index=False)
fold_disease=[]
for design in ['masked','deployment','thinning']:
 for path in (R/design/'evaluation').glob('pancreas_*/*/disease_fold_comparisons.csv'):
  frame=pd.read_csv(path).assign(design=design,unit=path.parent.parent.name,part=path.parent.name)
  fold_disease.append(frame)
if fold_disease: pd.concat(fold_disease,ignore_index=True).to_csv(out/'disease_fold_endpoints.csv',index=False)
slopes=[pd.read_csv(p).assign(unit=p.parent.name) for p in (R/'thinning/disease_slopes').glob('*/slopes.csv')]
if slopes: pd.concat(slopes,ignore_index=True).to_csv(out/'thinning_fold_slopes.csv',index=False)
print('summary complete',flush=True)
