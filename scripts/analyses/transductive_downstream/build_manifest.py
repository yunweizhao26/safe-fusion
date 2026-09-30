import json
from pathlib import Path
R=Path('artifacts/paper_evidence/review_round4/transductive_downstream')
E=Path('artifacts/paper_evidence'); LF=E/'review_round2/leakage_free'; C=E/'review_round3/colon_crossfit'; R3=E/'review_round3/downstream'
(R/'logs').mkdir(parents=True,exist_ok=True)
units=[u for u in json.loads((LF/'units_manifest.json').read_text()) if u['key']!='colon']+json.loads((C/'units_manifest.json').read_text())
z=Path('artifacts/external_trajectory/zebrafish')
units.append(dict(key='zebrafish',truth='external_data/prepared/zebrafish_trajectory.h5ad',corrupted=str(z/'corrupted.h5ad'),coordinates=str(z/'coordinates.parquet'),splits=str(z/'splits.parquet'),fit_split='development',fit_cells=None,selector_dir=str(z/'selector_mlp_biology_range'),contracts={n:str(z/p) for n,p in [('Gene median','gene_median'),('SVD','svd_impute'),('Weighted kNN','graph_smooth'),('MAGIC (inductive)','magic_inductive'),('scVI (inductive)','scvi_inductive')]}))
units[-1]['contracts'].update({n:str(R3/'masked/fits/zebrafish'/p) for n,p in [('MAGIC','magic'),('scVI','scvi')]})
entries=[]
for design in ['masked','deployment','thinning','null']:
 for u in units:
  key=u['key']; tissue=key.split('_')[0]
  if design=='null' and tissue not in ['colon','pancreas']: continue
  d=dict(u,design=design,dir=str(R/design/key),reuse={},standard={})
  if design=='masked':
   m=Path(u['contracts']['SVD']).parent
   d['reuse']['inductive']=str(m)
   d['reuse']['transductive']=str((C/'fusion_value/transductive' if tissue=='colon' else E/'review_round2/fusion_value/transductive')/key) if key!='zebrafish' else ''
   d['standard']={n:str(u['contracts']['MAGIC' if n=='magic' else 'scVI']) for n in ['magic','scvi']}
  elif design=='deployment':
   if tissue=='colon':
    old=E/'review_round4/transductive_references/sex_zeros/deployment'/key
    d['input_source']=str(old); d['reuse']['transductive']=str(old)
    d['standard']={n:str(old/(n+'_standard')) for n in ['magic','scvi']}
   elif key in ['zebrafish','norman_crispra']:
    old=E/'downstream_deployment/zebrafish' if key=='zebrafish' else E/'review_round2/norman_rebuilt/deployment/norman_crispra'
    d['input_source']=str(old); d['reuse']['inductive']=str(old/'methods'); d['selector_source']=str(old/'selector')
    d['standard']={n:str(R3/'deployment'/key/'fits'/n) for n in ['magic','scvi']}
  elif design=='thinning':
   tk=key.replace('pancreas_','pancreas_thinning_050_').replace('colon_','colon_thinning_050_') if tissue in ['pancreas','colon'] else ('norman_thinning_050' if key=='norman_crispra' else 'zebrafish_thinning_050')
   root=C/'thinning' if tissue=='colon' else E/'review_round4/transductive_main/pancreas_rebuilt_thinning' if tissue=='pancreas' else E/'review_round2/thinning_transfer'
   models=root/'mask_trained'/tk
   if key=='zebrafish': models=R3/'thinning_reference/models'/tk
   d['input_source']=str(models/'input'); d['reuse']['inductive']=str(models)
   d['reuse']['transductive']=str(E/'review_round4/transductive_main/thinning'/tk/'mask-trained') if key!='zebrafish' else ''
   d['standard']={n:str(models/n) for n in ['magic','scvi']}
   d['recorded_source']=str((R3/'thinning_reference/data'/tk if key=='zebrafish' else root/'data'/tk)/'corrupted.h5ad')
   d['selector_source']=str(models/'selector')
  entries.append(d)
(R/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
print([(i,d['design'],d['key']) for i,d in enumerate(entries)])
