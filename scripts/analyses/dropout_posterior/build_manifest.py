import json
from pathlib import Path
R=Path.cwd(); E=Path('artifacts/paper_evidence'); O=E/'review_round4/dropout_posterior'; V=E/'review_round3/value_v2_ablations/value/contracts'
(O/'logs').mkdir(parents=True,exist_ok=True)
units=[u for f in [E/'review_round2/leakage_free/units_manifest.json',E/'review_round3/colon_crossfit/units_manifest.json'] for u in json.loads(f.read_text()) if u['key']!='colon']
entries=[]
old_names=['Gene median','SVD','Weighted kNN','MAGIC (inductive)','scVI (inductive)']
names=['gene_median','svd_impute','graph_smooth','magic','scvi']
for u in units:
 key=u['key']; cc=key.startswith('colon_'); fr=E/('review_round3/colon_crossfit/fusion_value' if cc else 'review_round2/fusion_value')
 for chain in ['inductive','transductive']:
  root=Path(u['contracts']['Gene median']).parent if chain=='inductive' else fr/'transductive'/key
  entries.append(dict(id=f'masked/{key}/{chain}',analysis='masked',dataset=u['dataset'],key=key,chain=chain,input=u['corrupted'],coordinates=u['coordinates'],splits=u['splits'],truth=u['truth'],fit_split=u['fit_split'],unit_column=u['unit_column'],tie_seed=u['tie_seed'],seed=1729,teachers=[u['contracts'][n] for n in old_names] if chain=='inductive' else [str(root/n) for n in names],fusion=str(root/'safe_fusion'),selector=u['selector_dir'] if chain=='inductive' else '',baseline=str(fr/'selectors'/key/('safe_fusion' if chain=='inductive' else 'safe_fusion_transductive')/'test_scores.npy'),cache=str(V/key) if chain=='inductive' else '',unit=u,probability=str(fr/'nonzero_probability/scvi'/key/'mean.npy'),standard_scvi=u['contracts']['scVI']))
 for level in (['050','025'] if cc else ['050']):
  key='norman_thinning_050' if u['dataset']=='CRISPRa' else f"{u['dataset'].lower()}_thinning_{level}_{u['key'][-1]}"
  parent=E/('review_round3/colon_crossfit/thinning' if cc else 'review_round4/transductive_main/pancreas_rebuilt_thinning' if u['dataset']=='Pancreas' else 'review_round2/thinning_transfer')
  ir=parent/'mask_trained'/key
  for chain in ['inductive','transductive']:
   root=ir if chain=='inductive' else E/'review_round4/transductive_main/thinning'/key/'mask-trained'
   teacher_names=['gene_median','svd_impute','graph_smooth','magic_inductive','scvi_inductive'] if chain=='inductive' else names
   entries.append(dict(id=f'thinning/{key}/{chain}',analysis='thinning',dataset=u['dataset']+' '+level,key=key,chain=chain,input=str(ir/'input/hybrid.h5ad'),coordinates=str(ir/'input/coordinates.parquet'),splits=u['splits'],truth=u['truth'],fit_split=u['fit_split'],unit_column=u['unit_column'],tie_seed=1729,seed=1729,teachers=[str(root/n) for n in teacher_names],fusion=str(root/'safe_fusion'),selector=str(root/'selector'),baseline='',cache=str(V/key) if chain=='inductive' and u['dataset']!='Pancreas' else '',thinning_parent=str(parent),standard_scvi=str(ir/'scvi')))
(O/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
print(len(entries),'chains')
