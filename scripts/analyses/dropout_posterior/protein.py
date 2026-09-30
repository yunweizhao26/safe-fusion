from posterior import *
import anndata as ad
from evaluate_papalexi_crossmodal import safe_spearman

def run():
 specs=[s for s in json.loads((OUT/'manifest.json').read_text()) if s['analysis']=='protein'];spec=specs[0];a=ad.read_h5ad(spec['truth']);g=list(a.var_names).index('CD274');counts=v.dense(a.layers['counts'])
 panel=pd.read_parquet(ROOT/'artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet');panel=panel[panel.protein=='PDL1'].set_index('cell_id')
 frames=[];f=None
 for sp in specs:
  z=np.load(OUT/sp['id']/'scores.npz');keep=(z['cols']==g)&(counts[z['rows'],z['cols']]==0);indices=np.flatnonzero(keep);r=z['rows'][indices];src=a.obs['source_cell_id'].astype(str).to_numpy()[r];present=np.isin(src,panel.index);indices=indices[present];r=r[present];src=src[present]
  if f is None:f=panel.loc[src,['replicate','adt_count','adt_clr']].reset_index(drop=True)
  assert len(r)==655
  for name in ['baseline','posterior','posterior_fitted_rate','posterior_nb']:
   if name in z:f[sp['chain']+':'+name]=z[name][indices]
  if sp['chain']=='transductive' and 'scvi_probability' in z:f['comparator:scVI probability']=z['scvi_probability'][indices]
 groups=[x.index.to_numpy() for _,x in f.groupby('replicate',observed=True)];rng=np.random.default_rng(1729);draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(1000)];stats={}
 for column in ['adt_clr','adt_count']:
  for name in [x for x in f if ':' in x]:
   stats[(column,name)]=np.array([safe_spearman(f[name].to_numpy()[idx],f[column].to_numpy()[idx]) for idx in [np.arange(len(f)),*draws]])
 old=pd.read_csv(ROOT/'artifacts/paper_evidence/papalexi_crossmodal/benchmark/evaluation_cd274/continuous_protein_association.csv');old=old[old.method=='mlp_safe_fusion'].iloc[0];val=stats[('adt_clr','inductive:baseline')];np.testing.assert_allclose([val[0],*np.nanquantile(val[1:],[.025,.975])],[old.spearman,old.ci_low,old.ci_high],atol=1e-12,rtol=0)
 rows=[]
 for (col,name),values in stats.items():
  for chain in ['inductive','transductive']:
   diff=values-stats[(col,chain+':baseline')];lo,hi=np.nanquantile(values[1:],[.025,.975]);dl,dh=np.nanquantile(diff[1:],[.025,.975]);rows.append(dict(protein=col,method=name,reference=chain+':baseline',estimate=values[0],low=lo,high=hi,difference=diff[0],difference_low=dl,difference_high=dh,n=655))
 dest=OUT/'evaluation/protein';dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(dest/'results.csv',index=False);f.to_csv(dest/'zeros.csv',index=False);(dest/'audit.json').write_text(json.dumps(dict(inductive_point_parity=True,draws=1000,seed=1729,paired_on_original_first_method_draws=True),indent=2))
if __name__=='__main__':run()
