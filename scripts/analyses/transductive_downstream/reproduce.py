from pipeline import *
import numpy as np, pandas as pd
E=Path('artifacts/paper_evidence'); old=E/'review_round3/downstream/deployment/evaluation'
idx=int(sys.argv[1]); dataset=['colon','pancreas','zebrafish','norman_crispra'][idx]
out=R/'reproduction'/dataset
if dataset=='pancreas':
 base=E/'downstream_deployment/pancreas'; pan=Path('artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets')
 methods=sum((['--extra-method',f'safe_fusion_{pct}pct=safe_fusion_{pct}pct'] for pct in PCTS),[])
 run('scripts/evaluate_pancreas_crossfit_biology.py','--truth',pan/'preprocessed.h5ad','--corrupted',base/'fold_0/recorded.h5ad','--coordinates',base/'fold_0/empty_coordinates.parquet','--crossfit-dir',base,'--safe-fusion-subdir','safe_fusion',*methods,'--output-dir',out,'--bootstrap',2000,'--seed',1729,'--marker-panel','source')
 published=old/'pancreas_biology'
else:
 base=E/'downstream_deployment'/dataset
 if dataset=='colon':
  script='scripts/evaluate_colon_donor_biology.py'; truth='artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/preprocessed.h5ad'; extra=['--coordinates',base/'empty_coordinates.parquet','--marker-panel','source']; published=old/'markers/colon'
 elif dataset=='zebrafish':
  script='scripts/evaluate_trajectory_preservation.py'; truth='external_data/prepared/zebrafish_trajectory.h5ad'; extra=[]; published=old/'trajectory/zebrafish'
 else:
  base=E/'review_round2/norman_rebuilt/deployment/norman_crispra'; truth=E/'review_round2/leakage_free/norman_crispra/prepared.h5ad'; script='scripts/evaluate_interventional_grn.py'; extra=['--dataset','norman_crispra','--intervention','gain_of_function','--publication-doi','10.1126/science.aax4438']; published=old/'grn/norman_crispra'
 methods=sum((['--method',f'safe_fusion_{pct}pct={base}/safe_fusion_{pct}pct'] for pct in PCTS),[])
 run(script,'--truth',truth,'--corrupted',base/'recorded.h5ad','--splits',base/'splits.parquet',*methods,*extra,'--output-dir',out,'--bootstrap',2000,'--seed',1729)
checks=[]
for name,value in [('bootstrap_summary.parquet','estimate'),('paired_comparisons.parquet','difference')]:
 new=pd.read_parquet(out/name); orig=pd.read_parquet(published/name)
 keys=[k for k in ['scope','contrast','method','reference','metric'] if k in new and k in orig]
 merged=new.merge(orig,on=keys,suffixes=('_new','_old')); merged=merged[merged.method.str.match(r'^safe_fusion_(1|5|10)pct$')]
 assert len(merged)>0
 for col in [value,'ci_low','ci_high']:
  a,b=merged[col+'_new'].to_numpy(),merged[col+'_old'].to_numpy(); same=np.allclose(a,b,rtol=0,atol=0,equal_nan=True)
  checks.append(dict(dataset=dataset,file=name,column=col,n_rows=len(merged),max_abs_difference=float(np.nanmax(np.abs(a-b))),passed=bool(same)))
pd.DataFrame(checks).to_csv(out/'check.csv',index=False)
assert all(c['passed'] for c in checks),checks
