exec((__import__('pathlib').Path(__file__).parent/'build_manifest.py').read_text().split('(O/\'manifest.json\')')[0])

entries=json.loads((O/'manifest.json').read_text()); ids={s['id'] for s in entries}
def add(analysis,key,chain,root,input,coordinates,splits,truth,fit='development',teachers=None,fusion=None,selector=None,cache='',**extra):
 root=Path(root); names=['gene_median','svd_impute','graph_smooth','magic_inductive','scvi_inductive'] if chain=='inductive' else ['gene_median','svd_impute','graph_smooth','magic','scvi']
 s=dict(id=f'{analysis}/{key}/{chain}',analysis=analysis,dataset=key,key=key,chain=chain,input=str(input),coordinates=str(coordinates),splits=str(splits),truth=str(truth),fit_split=fit,unit_column='donor',tie_seed=1729,seed=1729,teachers=teachers or [str(root/n) for n in names],fusion=str(fusion or root/'safe_fusion'),selector=str(selector or root/'selector'),baseline='',cache=str(cache),**extra)
 if s['id'] not in ids: entries.append(s); ids.add(s['id'])
refs=E/'review_round4/transductive_references'; kd=E/'review_round2/knockdown'
for key in ['adamson_crispri','papalexi_eccite','norman_crispra']:
 norm=key=='norman_crispra'; old=kd/'norman_rebuilt/deployment' if norm else E/'downstream_deployment'/key
 new=refs/'knockdown/norman_rebuilt/deployment' if norm else refs/'knockdown/deployment'/key
 truth=E/'review_round2/leakage_free/norman_crispra/prepared.h5ad' if norm else Path('external_data/prepared')/(key+'.h5ad')
 for chain,base in [('inductive',old),('transductive',new)]:
  add('known',key,chain,base/'methods',old/'hybrid.h5ad',old/'coordinates.parquet',old/'splits.parquet',truth,selector=base/'selector',deploy=str(old))
for u in units:
 if u['dataset'] not in ['Pancreas','Colon']: continue
 key=u['key']; base=refs/'sex_zeros/deployment_rebuilt'/key
 for chain,root in [('inductive',refs/'sex_zeros/inductive_rebuilt'/key),('transductive',base)]:
  add('sex',key,chain,root,base/'hybrid.h5ad',base/'coordinates.parquet',base/'splits.parquet',u['truth'],u['fit_split'])
old=E/'papalexi_crossmodal/benchmark'; new=refs/'protein_cs'
for chain,root in [('inductive',old),('transductive',new)]:
 names=['gene_median','svd_impute','graph_smooth','magic_inductive','scvi_counts'] if chain=='inductive' else ['gene_median','svd_impute','graph_smooth','magic','scvi_counts']
 add('protein','papalexi',chain,root,old/'corrupted.h5ad',old/'coordinates.parquet',old/'splits.parquet','external_data/prepared/papalexi_eccite_crossmodal.h5ad',teachers=[str(root/n) for n in names],selector=root/'mlp_selector')
for s in entries:
 if s['id']=='thinning/norman_thinning_050/inductive': s['exact_calibration']=True
 if s['analysis']=='known':
  base=kd/'norman_rebuilt/standard_imputers' if s['key']=='norman_crispra' else kd/'standard_imputers'
  s['standard_scvi']=str(base/'scvi') if s['key']=='norman_crispra' else str(base/'scvi'/s['key'])
  if s['chain']=='transductive': s['teachers']=[x.replace('/magic','/magic_inductive').replace('/scvi','/scvi_inductive') for x in s['teachers']]
 elif s['analysis']=='sex': s['standard_scvi']=str(refs/'sex_zeros/deployment_rebuilt'/s['key']/'scvi_standard')
 elif s['analysis']=='protein':
  s['standard_scvi']=str(E/'papalexi_crossmodal/benchmark/scvi')
  if s['chain']=='inductive': s['teachers'][-1]=str(E/'papalexi_crossmodal/benchmark/scvi_inductive')
(O/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
print(len(entries),'total chains')
