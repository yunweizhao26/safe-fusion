from posterior import *
from types import SimpleNamespace
import evaluate_knockdown_zero_analyses as e
from compute_matched_baseline_f1_curves import tie_broken_order

def run(dataset,only=None):
 specs=[s for s in json.loads((OUT/'manifest.json').read_text()) if s['analysis']=='known' and s['key']==dataset and (only is None or s['chain']==only)]
 args=SimpleNamespace(deploy_root='artifacts/paper_evidence/downstream_deployment',review_root='artifacts/paper_evidence/review_round2/knockdown',norman_benchmark='artifacts/paper_evidence/review_round2/leakage_free/norman_crispra',norman_root='artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt',min_detection=.2,depth_strata=5,draws=2000,seed=1729)
 def methods(screen,gene_columns,inputs,seed):
  r,c=np.nonzero((screen.recorded==0)&screen.test[:,None]);lib=screen.recorded.sum(1,dtype=np.float64);out={}
  def add(name,score,value,random_ties=False):
   order=tie_broken_order(score,seed) if random_ties else np.argsort(-score.astype(float),kind='stable');rank=np.empty(len(r),int);rank[order]=np.arange(len(r));first=np.full(len(r),np.inf)
   for pct in reversed(e.FRACTIONS):first[rank<max(1,round(pct/100*len(r)))]=pct
   first[value<=0]=np.inf; matrix_order=np.full(screen.recorded.shape,np.inf,dtype=np.float32);matrix_order[r,c]=first
   matrix_score=np.full(screen.recorded.shape,np.nan,dtype=np.float64);matrix_score[r,c]=score
   def source(pct):
    hit=first<=pct;sums=lib+np.bincount(r[hit],weights=value[hit],minlength=len(lib));columns=screen.recorded[:,gene_columns].astype(float).copy();keep=hit&np.isin(c,gene_columns);local={int(g):j for j,g in enumerate(gene_columns)}
    columns[r[keep],np.array([local[int(g)] for g in c[keep]],dtype=int)]=value[keep]
    return sums,columns
   out[name]=e.Method(name,matrix_order,matrix_score,None,source)
  for sp in specs:
   z=np.load(OUT/sp['id']/'scores.npz');assert np.array_equal(r,z['rows']) and np.array_equal(c,z['cols'])
   add(sp['chain']+':baseline',z['baseline'],z['conditional'])
   for score in ['posterior','posterior_fitted_rate','posterior_nb']:
    if score not in z:continue
    for val in ['conditional','weighted','rate']:add(sp['chain']+':'+score+':'+val,z[score],z[val])
  prob=scvi_comparator(specs[0],r,c)
  if prob is not None:
   value=e.count_scale_contract(ROOT/specs[0]['standard_scvi'],screen,lib)[r,c];add('scVI P(X>0)',prob,value,True)
  for method in ['scimpute','screcover','dca_zinb','scvi_zinb']:
   root=ROOT/'artifacts/paper_evidence/review_round3/comparators/fits'/method/('deploy_'+dataset);p=root/'dropout_probability.npy'
   if p.exists():
    score=np.load(p,mmap_mode='r')[r,c];value=e.count_scale_contract(root,screen,lib)[r,c];add(method,score,value,True)
  return out
 e.build_methods=methods
 tables=e.evaluate_screen(dataset,args,None)
 dest=OUT/('evaluation/known_inductive' if only else 'evaluation/known')/dataset;dest.mkdir(parents=True,exist_ok=True)
 for name,frame in tables.items():frame.to_csv(dest/(name+'.csv'),index=False)
 old=pd.read_csv(ROOT/'artifacts/paper_evidence/review_round2/knockdown/evaluation/auroc.csv');old=old[(old.dataset==dataset)&(old.method=='safe_fusion')]
 got=tables['auroc'];got=got[got.method=='inductive:baseline']
 merged=old.merge(got,on=['dataset','gene','score_type'],suffixes=('_old','_new'))
 assert len(merged)==len(old)
 err=max(float(np.nanmax(np.abs(merged[k+'_old']-merged[k+'_new']))) for k in ['none','depth_strata','depth_matched']);assert err<1e-7,err

 rows=[]; genes=list(tables['targets'].gene);n=len(genes);draws=np.vstack([np.arange(n),np.random.default_rng([1729,e.SCOPES.index(dataset)]).integers(0,n,size=(2000,n))])
 for table,group,metrics in [('auroc',['score_type'],['none','depth_strata','depth_matched']),('effects',['fill_pct'],['log2fc_filled','shift']),('fills',['fill_pct'],['fill_control','fill_perturbed'])]:
  f=tables[table]
  for grouping,part in f.groupby(group):
   for metric in metrics:
    wide=part.pivot(index='gene',columns='method',values=metric).reindex(genes);stats={name:np.nanmean(wide[name].to_numpy()[draws],axis=1) for name in wide}
    for name,values in stats.items():
     for chain in ([only] if only else ['inductive','transductive']):
      diff=values-stats[chain+':baseline'];lo,hi=np.nanquantile(values[1:],[.025,.975]);dl,dh=np.nanquantile(diff[1:],[.025,.975]);rows.append(dict(dataset=dataset,analysis=table,group=str(grouping),metric=metric,method=name,reference=chain+':baseline',estimate=values[0],low=lo,high=hi,difference=diff[0],difference_low=dl,difference_high=dh,n_targets=n))
 pd.DataFrame(rows).to_csv(dest/'results.csv',index=False);(dest/'audit.json').write_text(json.dumps({'inductive_auroc_max_error':err,'draws':2000,'seed':1729},indent=2))
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
