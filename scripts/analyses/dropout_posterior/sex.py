from posterior import *
import anndata as ad
import disease_control_sex_zeros as e
from disease_control_common import stratified_draws
from sle_common import pair_auroc

def run(tissue):
 specs=[s for s in json.loads((OUT/'manifest.json').read_text()) if s['analysis']=='sex' and s['key'].startswith(tissue.lower())];frames=[];sexes={};availability=[]
 first=min(specs,key=lambda x:x['key']);reference=ad.read_h5ad(Path(first['input']).parent/'recorded.h5ad');refcounts=v.dense(reference.layers['corrupted_counts']).astype(np.float32);refsymbols=reference.var['feature_name'].astype(str).tolist();refindex={g:refsymbols.index(g) for g in e.SEX_GENES};refdonors=reference.obs.donor.astype(str).to_numpy();sex_reference={}
 for donor in np.unique(refdonors):
  keep=refdonors==donor;sex_reference[donor]='female' if refcounts[keep,refindex['XIST']].sum()>refcounts[keep,refindex['RPS4Y1']].sum() else 'male'
 for key in sorted(set(s['key'] for s in specs)):
  spec=next(s for s in specs if s['key']==key);a=ad.read_h5ad(spec['truth']);counts=v.dense(a.layers['counts']).astype(np.float32);obs=a.obs.copy().reset_index(drop=True)
  symbols=a.var['feature_name'].astype(str).tolist();ix={g:symbols.index(g) for g in e.SEX_GENES if g in symbols};assert list(a.obs_names)==list(reference.obs_names)
  split=pd.read_parquet(spec['splits']).set_index('cell_id').loc[a.obs_names,'split'].to_numpy();test=np.flatnonzero(split=='test')
  for donor,members in obs.groupby('donor',observed=True).groups.items():
   rr=np.array(members);sex=sex_reference[str(donor)];obs.loc[rr,'sex']=sex
   if np.any(split[rr]=='test'):sexes[str(donor)]=sex
  for gene in e.SEX_GENES:availability.append(dict(unit=key,gene=gene,present=gene in ix,test_cells=len(test),test_donors=obs.iloc[test].donor.nunique()))
  all_genes=e.SEX_GENES
  try:
   e.SEX_GENES={g:sex for g,sex in all_genes.items() if g in ix};f=e.zero_table(obs.iloc[test].reset_index(drop=True),counts[test],ix,None)
  finally:e.SEX_GENES=all_genes
  r=test[f.row.to_numpy()];c=f.col.to_numpy();lib=counts.sum(1);edges=np.quantile(lib[test],np.linspace(0,1,6)[1:-1]);f['depth']=np.searchsorted(edges,lib[r],side='right');f['unit']=key
  for chain in ['inductive','transductive']:
   sp=next(s for s in specs if s['key']==key and s['chain']==chain);z=np.load(OUT/sp['id']/'scores.npz');keys=z['rows'].astype(np.int64)*a.n_vars+z['cols'];indices=np.searchsorted(keys,r*a.n_vars+c);assert np.array_equal(keys[indices],r*a.n_vars+c)
   for name in ['baseline','posterior','posterior_fitted_rate','posterior_nb',*['teacher_'+n for n in v.TEACHERS]]:
    if name not in z or (name=='posterior_nb' and chain=='inductive'):continue
    method=chain+':'+name;f['score:'+method]=z[name][indices]
    if name in ['baseline','posterior','posterior_fitted_rate','posterior_nb']:
     order=np.argsort(-z[name].astype(float),kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
     for b in e.FRACTIONS:f[f'filled:{method}:{b}']=(rank[indices]<max(1,round(b*len(order))))&(z['conditional'][indices]>0)
  if 'scvi_probability' in z:
   method='scVI probability';f['score:'+method]=z['scvi_probability'][indices];order=np.lexsort((np.random.default_rng(1729).random(len(z['rows'])),-z['scvi_probability'].astype(float)));rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
   for b in e.FRACTIONS:f[f'filled:{method}:{b}']=(rank[indices]<max(1,round(b*len(order))))&(z['teacher_scvi_inductive'][indices]>0)
  frames.append(f)
 table=pd.concat(frames,ignore_index=True);donors=sorted(sexes);n=len(donors);draws=stratified_draws(np.array([sexes[x] for x in donors]),2000,1729);w=np.vstack([np.ones(n),np.array([np.bincount(d,minlength=n) for d in draws])]);rows=[];stats={}
 methods=[s[6:] for s in table if s.startswith('score:')]
 for gene,frame in table.groupby('gene'):
  positive=frame.kind.to_numpy()=='recorded zero, expressing sex';code=pd.Categorical(frame.donor.astype(str),categories=donors).codes;depth=frame.depth.to_numpy()
  for method in methods:
   score=frame['score:'+method].to_numpy();components=[]
   for q in [None,0,1,2,3,4]:
    keep=np.ones(len(frame),bool) if q is None else depth==q
    pos=[score[(code==i)&positive&keep] for i in range(n)];neg=[score[(code==i)&~positive&keep] for i in range(n)]
    u=pair_auroc(pos,neg);numer=np.einsum('ka,ab,kb->k',w,u,w);denom=(w@np.array(list(map(len,pos))))*(w@np.array(list(map(len,neg))))
    with np.errstate(invalid='ignore',divide='ignore'):value=numer/denom
    stats[(gene,method,'AUROC '+('all' if q is None else 'Q'+str(q+1)))]=value
    if q is not None:components.append((numer,denom))
   with np.errstate(invalid='ignore',divide='ignore'):stats[(gene,method,'AUROC within quintiles')]=sum(x[0] for x in components)/sum(x[1] for x in components)
   for b in e.FRACTIONS:
    col=f'filled:{method}:{b}'
    if col not in frame:continue
    filled=frame[col].to_numpy()
    for label,mask in [('dropout',positive),('biological',~positive)]:
     numer=w@np.bincount(code[mask&filled],minlength=n);denom=w@np.bincount(code[mask],minlength=n)
     with np.errstate(invalid='ignore',divide='ignore'):stats[(gene,method,f'fill {b:g} {label}')]=numer/denom
 for (gene,method,metric),values in stats.items():
  for chain in ['inductive','transductive']:
   diff=values-stats[(gene,chain+':baseline',metric)];lo,hi=np.nanquantile(values[1:],[.025,.975]);dl,dh=np.nanquantile(diff[1:],[.025,.975]);rows.append(dict(dataset=tissue,gene=gene,method=method,metric=metric,reference=chain+':baseline',estimate=values[0],low=lo,high=hi,difference=diff[0],difference_low=dl,difference_high=dh))

 fills,auc=e.summarize(table,['inductive:baseline'],['recorded zero, expressing sex'],draws,donors,'recorded counts')
 for row in auc:
  a=stats[(row['gene'],'inductive:baseline','AUROC all')];np.testing.assert_allclose([a[0],*np.nanquantile(a[1:],[.025,.975])],[row['auroc'],row['ci_low'],row['ci_high']],rtol=0,atol=1e-12)
 dest=OUT/'evaluation/sex'/tissue;dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(availability).to_csv(dest/'gene_availability.csv',index=False);table.to_parquet(dest/'zeros.parquet',index=False);pd.DataFrame(rows).to_csv(dest/'results.csv',index=False);(dest/'audit.json').write_text(json.dumps(dict(inductive_evaluator_parity=True,draws=2000,seed=1729,n_donors=n),indent=2))
if __name__=='__main__':run(sys.argv[1])
