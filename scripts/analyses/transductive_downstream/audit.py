from pipeline import *
import anndata as ad, numpy as np, pandas as pd
from safefusion_benchmark.hashing import sha256_file
checks=[]
for d in D:
 p=Path(d['dir'])
 if not (p/'READY').exists(): continue
 a=ad.read_h5ad(p/'hybrid.h5ad'); counts=a.layers['corrupted_counts'].toarray()
 input_hash=sha256_file(p/'hybrid.h5ad')
 coordinates_hash=sha256_file(p/'coordinates.parquet')
 splits=pd.read_parquet(p/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy(); test=splits=='test'
 det=np.load(p/'transductive/selector/detection.npz'); rows,cols=det['rows'],det['cols']
 assert np.isfinite(det['detection']).all() and ((det['detection']>=0)&(det['detection']<=1)).all()
 for pct in PCTS:
  c=np.load(p/f'transductive_{pct}pct/mean.npy'); w=np.load(p/f'detection_{pct}pct/mean.npy')
  assert np.array_equal(w[rows,cols],(c[rows,cols]*det['detection']).astype(np.float32))
  assert np.array_equal(w[test][counts[test]>0],counts[test][counts[test]>0])
  for n in NAMES:
   values=np.load(p/f'{n}_{pct}pct/mean.npy'); assert np.isfinite(values).all() and (values>=0).all()
   assert np.array_equal(values[test][counts[test]>0],counts[test][counts[test]>0])
   meta=json.loads((p/f'{n}_{pct}pct/metadata.json').read_text()); assert meta['cell_ids']==a.obs_names.tolist() and meta['gene_ids']==a.var_names.tolist()
  checks.append(dict(design=d['design'],unit=d['key'],check='conditional_detection_and_count_contracts',fraction=pct,passed=True,max_abs_difference=0))
 if d['design']=='masked':
  base=Path('artifacts/paper_evidence/review_round3/colon_crossfit/fusion_value') if d['key'].startswith('colon_') else Path('artifacts/paper_evidence/review_round2/fusion_value')
  score_file=base/'selectors'/d['key']/'safe_fusion_transductive/test_scores.npy'
  if score_file.exists():
   old_scores=np.load(score_file)
   checks.append(dict(design=d['design'],unit=d['key'],check='existing_transductive_scores',fraction=0,passed=bool(np.array_equal(det['scores'],old_scores)),max_abs_difference=float(np.max(np.abs(det['scores']-old_scores)))))
  for pct in PCTS:
   old=Path(d['selector_dir'])/f'safe_fusion_calibrated_mlp_topk_{str(pct/100).replace(".","p")}'/'mean.npy'
   if old.exists():
    v=np.load(p/f'safe_fusion_{pct}pct/mean.npy'); ref=np.load(old)
    checks.append(dict(design=d['design'],unit=d['key'],check='existing_inductive_fill',fraction=pct,passed=bool(np.array_equal(v,ref)),max_abs_difference=float(np.max(np.abs(v-ref)))))
 for family in ['inductive','transductive']:
  for n in TEACHERS:
   meta=json.loads((p/family/n/'metadata.json').read_text()); assert meta['cell_ids']==a.obs_names.tolist() and meta['gene_ids']==a.var_names.tolist()
   assert meta.get('input_sha256')==input_hash,(d['key'],d['design'],family,n,'input hash differs')
   assert meta.get('coordinates_sha256')==coordinates_hash,(d['key'],d['design'],family,n,'coordinate hash differs')
   if family=='transductive': assert meta['parameters'].get('transductive') is True
   else: assert meta['parameters'].get('test_used_for_fit') is False
assert checks
pd.DataFrame(checks).to_csv(R/'audit.csv',index=False)
print(pd.DataFrame(checks).groupby(['check','passed']).size().to_string())
assert all(r['passed'] for r in checks)
