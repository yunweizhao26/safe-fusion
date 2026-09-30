from common import *
import json
import os
import numpy as np
import pandas as pd
import selector_attribution as a

i=int(os.environ['SLURM_ARRAY_TASK_ID'])
u=units()[i//2]
model=['inductive','transductive'][i%2]
root=OUT/'s6_exact'/u.key/model
root.mkdir(parents=True,exist_ok=True)
original=a.fit_scores
def capture(variant,architecture,fit,test,*args):
    result=original(variant,architecture,fit,test,*args)
    dest=root/(variant+'__'+architecture)
    dest.mkdir(exist_ok=True)
    np.save(dest/'test_scores.npy',result[1])
    report=dict(test=dict(pr_auc=float(a.average_precision_score(test.labels,result[1])),
                          n_zeros=len(test.labels),n_masked_positives=int(test.labels.sum())))
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return result
a.fit_scores=capture
teachers=[u.contracts[n] for n in ['Gene median','SVD','Weighted kNN','MAGIC (inductive)','scVI (inductive)']]
fusion=u.contracts['Gene median'].parent/'safe_fusion'
if model=='transductive':
    teachers=[fusion_root(u)/'transductive'/u.key/n for n in ['gene_median','svd_impute','graph_smooth','magic','scvi']]
    fusion=fusion_root(u)/'transductive'/u.key/'safe_fusion'
sys.argv=['selector_attribution.py','--corrupted',str(u.corrupted),'--truth',str(u.truth),'--coordinates',str(u.coordinates),
          '--splits',str(u.splits),'--fusion-contract',str(fusion),'--output-dir',str(root),'--fit-split',u.fit_split,
          '--unit-column',u.unit_column,'--variants','full','teacher_only','context_only','--architectures','mlp',
          '--budgets',*[str(i/100) for i in range(1,11)],'--max-fit-rows','2000000','--bootstrap','2000','--seed','1729']
for p in teachers:sys.argv+=['--teacher-contract',str(p)]
a.main()
if u.key=='norman_crispra' and model=='inductive':
    old=pd.read_parquet(E/'review_round2/norman_rebuilt/selector_mlp_attribution_range/norman_crispra/ranking_metrics.parquet')
    new=pd.read_parquet(root/'ranking_metrics.parquet')
    joined=old.merge(new,on='variant',suffixes=('_old','_new'))
    error=float(np.max(np.abs(joined.test_pr_auc_old-joined.test_pr_auc_new)))
    (root/'parity.json').write_text(json.dumps(dict(max_ap_difference=error,rows=len(joined)),indent=2)+'\n')
    assert error<1e-10,(error,joined[['variant','test_pr_auc_old','test_pr_auc_new']].to_dict('records'))
