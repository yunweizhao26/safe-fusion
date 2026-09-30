import json, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import anndata as ad
import numpy as np
import pandas as pd
import disease_control_sex_zeros as e
from disease_control_common import dense, stratified_draws
from masked_f1_units import Unit, load_unit, fraction_name, UNIT_FRACTIONS, unit_counts
import evaluate_donor_label_masked_f1 as f
R=Path.cwd(); O=R/'artifacts/paper_evidence/review_round4/transductive_references/sex_zeros'
MANIFESTS=[R/'artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json',R/'artifacts/paper_evidence/review_round3/colon_crossfit/units_manifest.json']
units=[u for p in MANIFESTS for u in json.loads(p.read_text()) if u['dataset'] in ('Pancreas','Colon') and u['key']!='colon']
METHODS=('safe_fusion','safe_fusion_donor','safe_fusion_inductive')
f.METHODS=METHODS

def selector(u,method,setting):
 if setting=='recorded':
  return O/('inductive_rebuilt' if method=='safe_fusion_inductive' else 'deployment_rebuilt')/u['key']/('selector_donor' if method=='safe_fusion_donor' else 'selector')
 return R/u['selector_dir'] if method=='safe_fusion_inductive' else O/'masked_donor'/u['key']/('selector_donor' if method=='safe_fusion_donor' else 'selector')

def recorded():
 for tissue in ('Pancreas','Colon'):
  tables=[]; sex_rows=[]; audited=[]; availability=[]
  first=next(u for u in units if u['dataset']==tissue)
  reference=ad.read_h5ad(O/'deployment_rebuilt'/first['key']/'recorded.h5ad')
  reference_counts=dense(reference.layers['corrupted_counts']).astype(np.float32)
  reference_index={g:list(reference.var['feature_name'].astype(str)).index(g) for g in e.SEX_GENES}
  sex_reference={}
  for donor in reference.obs['donor'].astype(str).unique():
   members=reference.obs['donor'].astype(str).to_numpy()==donor
   x=float(reference_counts[members,reference_index['XIST']].sum()); y=float(reference_counts[members,reference_index['RPS4Y1']].sum())
   sex_reference[donor]=(x,y,'female' if x>y else 'male')
  for u in [u for u in units if u['dataset']==tissue]:
   base=O/'deployment_rebuilt'/u['key']; a=ad.read_h5ad(base/'recorded.h5ad')
   counts=dense(a.layers['corrupted_counts']).astype(np.float32)
   obs=a.obs.copy().reset_index(drop=True)
   symbols=list(a.var['feature_name'].astype(str)); ix={g:symbols.index(g) for g in e.SEX_GENES if g in symbols}
   assert list(a.obs_names)==list(reference.obs_names)
   split=pd.read_parquet(base/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy()
   test=np.flatnonzero(split=='test')
   for donor, members in obs.groupby('donor',observed=True).groups.items():
    rows=np.asarray(members); x,y,sex=sex_reference[str(donor)]; obs.loc[rows,'sex']=sex
    if np.any(split[rows]=='test'): sex_rows.append({'donor':str(donor),'sex':sex,'unit':u['key'],'n_cells':len(rows),'XIST_counts':float(x),'RPS4Y1_counts':float(y)})
   for gene in e.SEX_GENES: availability.append({'unit':u['key'],'gene':gene,'present':gene in ix,'test_cells':len(test),'test_donors':obs.iloc[test]['donor'].nunique()})
   all_genes=e.SEX_GENES
   try:
    e.SEX_GENES={g:sex for g,sex in all_genes.items() if g in ix}
    table=e.zero_table(obs.iloc[test].reset_index(drop=True),counts[test],ix,None)
   finally: e.SEX_GENES=all_genes
   rows=test[table['row'].to_numpy()]; cols=table['col'].to_numpy()
   for method in METHODS:
    s=selector(u,method,'recorded'); scores=pd.read_parquet(s/'selected_gene_scores.parquet')
    scores=scores[scores['split']=='test'].set_index(['cell_index','gene_index'])
    table[f'score:{method}']=scores.loc[list(zip(rows,cols)),'selector_score'].to_numpy()
    for fraction in e.FRACTIONS:
     values=np.load(s/f'safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}'/'mean.npy',mmap_mode='r')
     table[f'filled:{method}:{fraction}']=values[rows,cols]!=0
    if method=='safe_fusion_donor':
     hybrid=ad.read_h5ad(base/'hybrid.h5ad'); observed=dense(hybrid.layers['corrupted_counts']).astype(np.float32)

     full=pd.read_parquet(s/'selected_gene_scores.parquet')
     feature_columns=[c for c in full if 'condition' in c]
     assert len(feature_columns)==2,feature_columns
     for donor,members in obs.groupby('donor',observed=True).groups.items():
      block=full[full['cell_index'].isin(members)]
      if block.empty: continue
      gene=block['gene_index'].to_numpy(); expected=[np.log1p(observed[np.asarray(members)].mean(axis=0))[gene],(observed[np.asarray(members)]<=0).mean(axis=0)[gene]]
      for name,values_ in zip(feature_columns,expected): np.testing.assert_allclose(block[name],values_,rtol=1e-6,atol=1e-7)
     assert json.loads((s/'calibration_report.json').read_text())['condition_feature_source']=='all_cells'
     audited.append(u['key'])
   tables.append(table.assign(unit=u['key']))
  table=pd.concat(tables,ignore_index=True); sex=pd.DataFrame(sex_rows)
  assert not sex['donor'].duplicated().any(),'donor tested in multiple folds'
  donors=sorted(sex['donor'].astype(str).unique()); groups=sex.set_index('donor').loc[donors,'sex'].to_numpy()
  draws=stratified_draws(groups,2000,1729)
  fills,auc=e.summarize(table,list(METHODS),['recorded zero, expressing sex'],draws,donors,'recorded counts')
  out=O/('evaluation' if sys.argv[1]=='recorded' else sys.argv[1])/tissue.lower(); out.mkdir(parents=True,exist_ok=True)
  pd.DataFrame(fills).to_csv(out/'fill_rates.csv',index=False); pd.DataFrame(auc).to_csv(out/'auroc.csv',index=False)
  pd.DataFrame(e.zero_counts(table,'recorded counts')).to_csv(out/'zero_counts.csv',index=False)
  sex.to_csv(out/'donor_sex.csv',index=False)
  pd.DataFrame(availability).to_csv(out/'gene_availability.csv',index=False)
  (out/'audit.json').write_text(json.dumps({'donor_feature_checks':audited,'donors':len(donors),'draws':2000,'seed':1729},indent=2))

def masked():
 reports={}; records=[]; frames=[]
 for u in units:
  fields={**u,'contracts':{}}
  for k in ('corrupted','coordinates','splits','truth','selector_dir'): fields[k]=R/fields[k]
  data=load_unit(Unit(**fields)); cells=np.flatnonzero(data.split=='test'); rr,cols=np.where(data.counts[cells]==0); rows=cells[rr]; labels=data.masked[rows,cols]
  for method in METHODS:
   for fraction in UNIT_FRACTIONS:
    path=selector(u,method,'masked')/f'safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}'/'mean.npy'
    chosen=np.load(path,mmap_mode='r')[rows,cols]!=0
    counts=unit_counts(chosen,labels,data.unit_labels[rows]).assign(dataset=u['dataset'],fold=u['key'],method=method,fraction=fraction)
    counts['unit']=u['key']+':'+counts['unit'].astype(str)
    frames.append(counts)
    records.append({'tissue':u['dataset'],'unit':u['key'],'method':method,'fraction':fraction,'n_selected':int(chosen.sum()),'n_true_positive':int((chosen&labels).sum()),'n_masked_positives':int(labels.sum()),'masked_f1':2*(chosen&labels).sum()/(chosen.sum()+labels.sum())})
 all_counts=pd.concat(frames,ignore_index=True)
 for tissue,frame in all_counts.groupby('dataset',sort=False):
  source=MANIFESTS[0 if tissue=='Pancreas' else 1].parent/'masked_f1_unit_counts.parquet'
  old=pd.read_parquet(source); old=old[(old['dataset']==tissue)&(old['method']=='Safe Fusion')].copy()
  old['unit']=old['group'].astype(str)+':'+old['unit'].astype(str)
  keys=['unit','fraction']; columns=['n_selected','n_true_positive','n_masked_positives']
  new=frame[frame['method']=='safe_fusion_inductive']
  pd.testing.assert_frame_equal(old.set_index(keys)[columns].sort_index(),new.set_index(keys)[columns].sort_index(),check_dtype=False)
  reports[tissue]=f.tissue_report(frame,2000,1729) if 'safe_fusion_donor' in METHODS else plain_report(frame)
  reports[tissue]['inductive_unit_counts_match_existing']=True
 out=O/('masked_f1' if sys.argv[1]=='masked' else sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(records).to_csv(out/'masked_f1_by_fraction.csv',index=False); all_counts.to_csv(out/'masked_f1_unit_counts.csv',index=False)
 (out/'masked_f1_report.json').write_text(json.dumps(reports,indent=2)+'\n')
def plain_report(frame):
 donors=sorted(frame.unit.unique()); draws=np.random.default_rng(1729).integers(0,len(donors),size=(2000,len(donors)))
 values={}
 for method,part in frame.groupby('method',sort=False):
  a=f.unit_arrays(part,donors)
  def mean(idx): return f.pooled_f1(a['n_selected'][idx],a['n_true_positive'][idx],a['n_masked_positives'][idx]).mean()
  values[method]=f.interval(mean(slice(None)),np.array([mean(d) for d in draws]))
 return {'n_donors':len(donors),'mean_f1_percent':values}
if __name__=='__main__':
 if sys.argv[1]=='masked_pancreas': units=[u for u in units if u['dataset']=='Pancreas']
 if sys.argv[1]=='masked_plain': METHODS=('safe_fusion','safe_fusion_inductive')
 if sys.argv[1]=='recorded_transductive': METHODS=('safe_fusion','safe_fusion_donor')
 if sys.argv[1]=='recorded_inductive': METHODS=('safe_fusion_inductive',)
 (recorded if sys.argv[1].startswith('recorded') else masked)()
