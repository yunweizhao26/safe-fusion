from pathlib import Path
import json
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/transductive_references')
pairs=[('knockdown','artifacts/paper_evidence/review_round2/knockdown/evaluation'),('masked_f1','artifacts/paper_evidence/review_round2/knockdown/masked_f1'),('protein','artifacts/paper_evidence/review_round2/protein/evaluation'),('sex_zeros/pancreas','artifacts/paper_evidence/disease_control_checks/sex_zeros/pancreas'),('sex_zeros/colon','artifacts/paper_evidence/disease_control_checks/sex_zeros/colon')]
results=[]
for local,original in pairs:
 for p in sorted(Path(original).glob('*.csv')):
  q=O/'verification'/local/p.name
  try:
   old=pd.read_csv(p); new=pd.read_csv(q)
   pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-12,atol=1e-12)
   results.append({'analysis':local,'file':p.name,'rows':len(old),'match':True})
  except Exception as exc: results.append({'analysis':local,'file':p.name,'match':False,'error':str(exc)[:500]})
(O/'verification/comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'tables':len(results),'matched':sum(x['match'] for x in results),'failures':[x for x in results if not x['match']]},indent=2))
assert all(x['match'] for x in results)
