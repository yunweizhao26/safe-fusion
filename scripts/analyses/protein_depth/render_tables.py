from pathlib import Path
import pandas as pd
import numpy as np
O=Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_depth'
LABEL={'safe_fusion':'SF selector','standard_depth':'Standard depth','residual_logit':'Residual logit','fused_value':'Fused value','mu_per_library':'Fused value/library','d_times_mu':'Selector × fused value','p_nonzero':'P(nonzero)','svd':'SVD','weighted_knn':'Weighted kNN','magic':'MAGIC','scvi':'scVI','library_size':'Library size','svd_all_cells':'SVD all cells','weighted_knn_all_cells':'kNN all cells','ifng_score':'IFN-gamma score','target_baseline':'Target baseline','cell_zero_score':'Cell zero score'}
METRIC={'pooled_spearman':'Pooled','partial_depth_spearman':'Depth adjusted','within_l1_spearman':'Within L1','within_l2_spearman':'Within L2','within_target_spearman':'Within target','depth_spearman':'Score–depth','pooled_mean_auroc':'Mean AUROC'}

def fmt(v):return 'NA' if not np.isfinite(v) else f'{v:.3f}'
def ci(row,value='estimate'):return f'{fmt(row[value])} [{fmt(row.ci_low)}, {fmt(row.ci_high)}]' if np.isfinite(row[value]) else 'NA'
def table(headers,rows):return '\n'+'\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def main():
 a=pd.read_csv(O/'tables/association.csv'); d=pd.read_csv(O/'tables/differences.csv'); s=pd.read_csv(O/'tables/summary.csv'); r=pd.read_csv(O/'tables/ranking_summary.csv'); p=pd.read_csv(O/'tables/pairs.csv'); dg=pd.read_csv(O/'tables/diagnostics.csv')
 dest=O/'tables/markdown';dest.mkdir(exist_ok=True)
 def save(name,text):(dest/f'{name}.md').write_text(text)
 metrics=['pooled_spearman','depth_spearman','partial_depth_spearman','within_l1_spearman','within_l2_spearman']
 for phase,methods in [('diagnosis',[x for x in LABEL if x not in ['standard_depth','residual_logit']]),('correction',['safe_fusion','standard_depth','residual_logit','magic','scvi'])]:
  rows=[]
  for method in methods:
   b=r[(r.stratum=='all')&(r.method==method)].set_index('metric')
   if len(b):rows.append([LABEL[method]]+[ci(b.loc[m],'median') for m in metrics])
  save(f'hao_{phase}',table(['Ranking']+[METRIC[m] for m in metrics],rows))
 for ref in ['safe_fusion','standard_depth','residual_logit']:
  rows=[]
  for stratum in ['all','training <200','training >=200','depth negative','depth neutral','depth positive']:
   for metric in ['pooled_spearman','partial_depth_spearman','within_l1_spearman','within_l2_spearman']:
    cells=[]
    for comparator in ['magic','scvi']:
     b=s[(s.stratum==stratum)&(s.ranking==ref)&(s.comparator==comparator)&(s.metric==metric)].iloc[0]
     cells.append(f'{int(b.wins)}/{int(b.defined_pairs)}; {ci(b,"median_delta")}; p={b.sign_test_p:.3g}')
    rows.append([stratum,METRIC[metric],*cells])
  save(f'strata_{ref}',table(['Pairs','Statistic','vs MAGIC: wins; median Δ [95% CI]; sign p','vs scVI: wins; median Δ [95% CI]; sign p'],rows))
 rows=[]
 for pair in p.itertuples():
  def v(ref,comp,metric):return ci(d[(d.case==pair.case)&(d.ranking==ref)&(d.comparator==comp)&(d.metric==metric)].iloc[0])
  depth=dg[(dg.case==pair.case)&(dg.metric=='protein_depth_spearman')].iloc[0]
  rows.append([f'{pair.gene}/{pair.protein}',pair.n_train_detected,ci(depth),v('safe_fusion','magic','pooled_spearman'),v('safe_fusion','magic','partial_depth_spearman'),v('safe_fusion','magic','within_l2_spearman')])
 save('hao_per_pair_diagnosis',table(['Gene/antibody','Training detected','Protein–depth ρ [95% CI]','SF−MAGIC pooled','SF−MAGIC depth adjusted','SF−MAGIC within L2'],rows))
 rows=[]
 for pair in p.itertuples():
  def val(ref,comp,metric):return ci(d[(d.case==pair.case)&(d.ranking==ref)&(d.comparator==comp)&(d.metric==metric)].iloc[0])
  sd=a[(a.case==pair.case)&(a.method=='standard_depth')&(a.metric=='pooled_spearman')].iloc[0]
  rows.append([f'{pair.gene}/{pair.protein}',ci(sd),val('standard_depth','magic','pooled_spearman'),val('standard_depth','scvi','pooled_spearman')])
 save('hao_per_pair_correction',table(['Gene/antibody','Standard-depth pooled ρ','Standard−MAGIC','Standard−scVI'],rows))
 for case in ['papalexi_small','papalexi_full']:
  rows=[]
  for method,b in a[a.case==case].groupby('method',sort=False):
   b=b.set_index('metric');rows.append([LABEL[method]]+[ci(b.loc[m]) for m in ['pooled_spearman','depth_spearman','partial_depth_spearman','within_target_spearman']])
  save(case,table(['Ranking','Pooled ρ','Score–depth ρ','Depth-adjusted ρ','Within-target ρ'],rows))
  for ref in ['safe_fusion','standard_depth','residual_logit']:
   rows=[]
   for comp,b in d[(d.case==case)&(d.ranking==ref)].groupby('comparator',sort=False):
    b=b.set_index('metric');rows.append([LABEL[comp]]+[ci(b.loc[m]) for m in ['pooled_spearman','partial_depth_spearman','within_target_spearman']])
   save(f'{case}_{ref}_differences',table(['Comparator','Δ pooled','Δ depth adjusted','Δ within target'],rows))
 print('Rendered complete main and supporting Markdown tables.')
if __name__=='__main__':main()
