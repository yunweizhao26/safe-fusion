from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
O=Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_depth';ROOT=O.parents[3]

def main():
 h=hashlib.sha256((O/'hypothesis.md').read_bytes()).hexdigest()
 expected=(O/'runtime/hypothesis.sha256').read_text().split()[0]
 assert h==expected
 checks=[];protected={}
 for kind in ['hao','papalexi_small','papalexi_full']:
  receipt=json.loads((O/f'runtime/preparation_{kind}.json').read_text())
  assert receipt['hypothesis_sha256']==h
  assert max(v['max_abs_difference'] for v in receipt['original_score_checks'])<1e-6
  protected.update({k:v for k,v in receipt['source_sha256'].items() if not k.startswith('artifacts/paper_evidence/review_round4/protein_depth/')})
  checks.append(dict(kind=kind,reference_library=receipt['reference_library'],saved_genes=len(receipt['original_score_checks']),saved_rows=sum(v['n'] for v in receipt['original_score_checks']),max_original_score_error=max(v['max_abs_difference'] for v in receipt['original_score_checks'])))
 for rel,digest in protected.items():
  with (ROOT/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,rel
 cases=[f'hao_{i:03}' for i in range(41)]+['papalexi_small','papalexi_full']
 for case in cases:
  receipt=json.loads((O/f'runtime/{case}_evaluation.json').read_text())
  assert receipt['draws']==2000
  if case.startswith('hao_'):assert receipt['original_hao_2000_draws_replayed']
  else:assert (O/f'runtime/{case}_original_replay.json').exists()
 a=pd.read_csv(O/'tables/association.csv');d=pd.read_csv(O/'tables/differences.csv');diag=pd.read_csv(O/'tables/diagnostics.csv')
 assert set(a.case)==set(cases)
 assert a[(a.method=='library_size')&(a.metric=='partial_depth_spearman')].estimate.isna().all()
 assert a[a.method.isin(['standard_depth','residual_logit'])].query('metric == "pooled_spearman"').defined_draws.eq(2000).all()
 r2=diag[diag.metric.str.startswith('protein_variance_explained')].estimate
 assert r2.between(-1e-12,1+1e-12).all()
 undefined=a[a.estimate.isna()]
 undefined.to_csv(O/'tables/undefined_statistics.csv',index=False)
 partial=a[(a.method!='library_size')&(a.metric=='partial_depth_spearman')&(a.defined_draws<2000)]
 result=dict(passed=True,cases=43,draws=2000,hypothesis_sha256=h,protected_assets_unchanged=len(protected),reconstructions=checks,association_rows=len(a),paired_rows=len(d),undefined_association_rows=len(undefined),nonlibrary_partial_incomplete_draw_rows=len(partial))
 (O/'runtime/final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
