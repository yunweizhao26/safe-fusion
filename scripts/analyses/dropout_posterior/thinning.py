from posterior import *
import anndata as ad
import evaluate_thinning_transfer as t

def run(dataset,only=None):
 specs=[s for s in json.loads((OUT/'manifest.json').read_text()) if s['analysis']=='thinning' and s['dataset']==dataset and (only is None or s['chain']==only)]
 tables=[]; gates=[];reference_tables=[]
 oldpath=Path(specs[0]['thinning_parent'])/'evaluation/transfer_summary.csv'
 if not oldpath.exists():
  import inspect
  namespace=dict(t.__dict__);namespace.update(DESIGNS=('mask-trained',),SELECTORS={'Safe Fusion':None})
  source=inspect.getsource(t.unit_table);a=source.index('    scores = {');b=source.index('    selections:',a);source=source[:a]+'    scores = {}\n'+source[b:]
  exec(compile(source,'<original thinning evaluator, baseline only>','exec'),namespace)
  exec(compile(inspect.getsource(t.analyze),'<original thinning summary>','exec'),namespace)
 for key in sorted(set(s['key'] for s in specs)):
  spec=next(s for s in specs if s['key']==key); parent=Path(spec['thinning_parent']); counts=ad.read_h5ad(spec['input']); truth=ad.read_h5ad(spec['truth'])
  x=t.dense(counts.layers['corrupted_counts']).astype(np.float32); y=t.dense(truth.layers['counts']).astype(np.float32)
  assert np.array_equal(counts.obs_names,truth.obs_names) and np.array_equal(counts.var_names,truth.var_names)
  split=pd.read_parquet(spec['splits']).set_index('cell_id').loc[counts.obs_names,'split'].to_numpy()
  if not oldpath.exists():
   unit={'key':key,'splits':spec['splits'],'thin_root':parent/'thinning_trained'/key,'mask_root':Path(spec['selector']).parent}
   reference_tables.append(namespace['unit_table']({},unit,y,x,counts.obs_names.astype(str).to_numpy(),truth.obs[spec['unit_column']].astype(str).to_numpy()))
  r,c=np.nonzero((x==0)&(split=='test')[:,None]); original=y[r,c]; positive=original>0; one=original==1
  bio=truth.obs[spec['unit_column']].astype(str).to_numpy()[r]; unique,codes=np.unique(bio,return_inverse=True)
  def agg(a):return np.bincount(codes,weights=a,minlength=len(unique))
  table=dict(units=[key+':'+u for u in unique],positives=agg(positive),count_one=agg(one),counts={},count_one_hits={},n=len(r),original=original[positive])
  for chain in ([only] if only else ['inductive','transductive']):
   sp=next(s for s in specs if s['key']==key and s['chain']==chain)
   if (OUT/sp['id']/'blocked.json').exists():
    selections=t.selector_selections(Path(sp['selector']),r,c,len(r));method=chain+':baseline'
    table['counts'][method]={b:(agg(sel),agg(sel&positive)) for b,sel in selections.items()};table['count_one_hits'][method]=agg(selections[.05]&one)
    continue
   z=np.load(OUT/sp['id']/'scores.npz')
   assert np.array_equal(r,z['rows']) and np.array_equal(c,z['cols'])
   for name in ['baseline','posterior','posterior_fitted_rate','posterior_nb']:
    if name not in z or (name=='posterior_nb' and chain=='inductive'):continue
    score=z[name]; order=np.argsort(-score.astype(float),kind='stable');rank=np.empty(len(r),int);rank[order]=np.arange(len(r))
    selections={b:rank<max(1,round(b*len(r))) for b in t.FRACTIONS};method=chain+':'+name
    table['counts'][method]={b:(agg(sel),agg(sel&positive)) for b,sel in selections.items()}
    table['count_one_hits'][method]=agg(selections[.05]&one)
  score=scvi_comparator(spec,r,c)
  if score is not None:
   from compute_matched_baseline_f1_curves import tie_broken_order
   order=tie_broken_order(score,1729);rank=np.empty(len(r),int);rank[order]=np.arange(len(r));selections={b:rank<max(1,round(b*len(r))) for b in t.FRACTIONS};method='scVI P(X>0)'
   table['counts'][method]={b:(agg(sel),agg(sel&positive)) for b,sel in selections.items()};table['count_one_hits'][method]=agg(selections[.05]&one)
  tables.append(table)
 names=sorted(set.intersection(*(set(x['counts']) for x in tables)));n=sum(len(x['units']) for x in tables)
 draws=np.vstack([np.arange(n),np.random.default_rng(7).integers(0,n,size=(2000,n))]);pos=np.concatenate([x['positives'] for x in tables]);one=np.concatenate([x['count_one'] for x in tables])
 stats={}
 for name in names:
  curves=[]
  for b in t.FRACTIONS:
   sel=np.concatenate([x['counts'][name][b][0] for x in tables]);hit=np.concatenate([x['counts'][name][b][1] for x in tables]);curves.append(200*hit[draws].sum(1)/(sel[draws].sum(1)+pos[draws].sum(1)))
  hits=np.concatenate([x['count_one_hits'][name] for x in tables]);stats[name]={'mean_f1_1_10':np.mean(curves,axis=0),'count_one_recall_at_5':100*hits[draws].sum(1)/one[draws].sum(1)}

 label=('norman' if dataset.startswith('CRISPRa') else dataset.split()[0].lower())+'_thinning_'+dataset.split()[1]
 if oldpath.exists():
  old=pd.read_csv(oldpath);old=old[(old.dataset==label)&(old.method=='Safe Fusion, mask-trained')].iloc[0]
 else:
  baseline=namespace['analyze'](reference_tables,2000,7);old=pd.Series(baseline['rows'][0]);dest=OUT/'baseline_reproduction'/label;dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(baseline['rows']).to_csv(dest/'transfer_summary.csv',index=False);(dest/'source.json').write_text(json.dumps({'reason':'upstream summary pending','kernel':'evaluate_thinning_transfer.unit_table and analyze','method':'Safe Fusion, mask-trained','no_comparator_fits_required':True},indent=2))
 for metric,col in [('mean_f1_1_10','mean_f1_1_10'),('count_one_recall_at_5','recall_at_5_count_1')]:
  got=stats['inductive:baseline'][metric][0];np.testing.assert_allclose(got,old[col],atol=1e-8,rtol=0);gates.append(dict(metric=metric,difference=float(got-old[col])))
 rows=[]
 for name,metrics in stats.items():
  for metric,values in metrics.items():
   for chain in ([only] if only else ['inductive','transductive']):
    diff=values-stats[chain+':baseline'][metric];lo,hi=np.quantile(values[1:],[.025,.975]);dl,dh=np.quantile(diff[1:],[.025,.975])
    rows.append(dict(dataset=dataset,method=name,metric=metric,reference=chain+':baseline',estimate=values[0],low=lo,high=hi,difference=diff[0],difference_low=dl,difference_high=dh,n_units=n))
 dest=OUT/('evaluation/thinning_inductive' if only else 'evaluation/thinning')/dataset.replace(' ','_');dest.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(dest/'results.csv',index=False);(dest/'audit.json').write_text(json.dumps(dict(reproduction=gates,draws=2000,seed=7),indent=2))
 np.savez(dest/'bootstrap.npz',**{name+'|'+m:a for name,d in stats.items() for m,a in d.items()})
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
