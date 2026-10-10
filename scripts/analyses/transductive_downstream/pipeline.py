import json, os, shutil, subprocess, sys
from pathlib import Path
R=Path('artifacts/paper_evidence/review_round4/transductive_downstream')
sys.path[:0]=[str(Path.cwd()/'scripts'),str(Path.cwd()/'src')]
PY='.venv/bin/python'; MAGIC='.conda-magic-current/bin/python'; SCVI='.conda-scvi-current/bin/python'
TEACHERS=['gene_median','svd_impute','graph_smooth','magic_inductive','scvi_inductive']
NAMES=['safe_fusion','transductive','detection','svd','magic','scvi']; PCTS=[1,5,10]
D=json.loads((R/'manifest.json').read_text())
def run(script,*args,python=PY):
 cmd=[python,str(script),*map(str,args)]; print('RUN', ' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
def link(src,dst):
 src=Path(src); dst=Path(dst)
 if not src.exists(): raise FileNotFoundError(src)
 if dst.exists(): return
 dst.parent.mkdir(parents=True,exist_ok=True);
 try: dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
 except FileExistsError: assert dst.resolve()==src.resolve()
def complete(p): return (Path(p)/'metadata.json').exists() and (Path(p)/'mean.npy').exists()
def prep(d):
 p=Path(d['dir']); p.mkdir(parents=True,exist_ok=True)
 if d['design']=='masked':
  for name,k in [('hybrid.h5ad','corrupted'),('coordinates.parquet','coordinates'),('splits.parquet','splits')]: link(d[k],p/name)
 elif d['design']=='null':
  if not (p/'manifest.json').exists(): run('scripts/r3_downstream_permute.py','--deployment-dir',R/'deployment'/d['key'],'--output-dir',p,'--seed',1729)
 elif 'input_source' in d:
  src=Path(d['input_source'])
  for n in ['hybrid.h5ad','coordinates.parquet','splits.parquet']: link(src/n,p/n)
  if d['design']=='thinning':
   link(d['recorded_source'],p/'recorded.h5ad')
   import pandas as pd
   pd.read_parquet(p/'coordinates.parquet').iloc[:0].to_parquet(p/'empty_coordinates.parquet',index=False)
  else:
   for n in ['recorded.h5ad','empty_coordinates.parquet']: link(src/n,p/n)
 elif not (p/'manifest.json').exists():
  run('scripts/build_deployment_inputs.py','--truth',d['truth'],'--corrupted',d['corrupted'],'--coordinates',d['coordinates'],'--splits',d['splits'],'--output-dir',p)
 link(d['truth'],p/'truth.h5ad')
def common(d):
 p=Path(d['dir']); return ['--input',p/'hybrid.h5ad','--coordinates',p/'coordinates.parquet','--splits',p/'splits.parquet','--seed',1729]
def reuse(d,family,name):
 src=d['reuse'].get(family)
 if not src: return False
 if family=='transductive': name={'magic_inductive':'magic','scvi_inductive':'scvi'}.get(name,name)
 src=Path(src)/name
 if complete(src):
  link(src,Path(d['dir'])/family/({'magic':'magic_inductive','scvi':'scvi_inductive'}.get(name,name))); return True
 return False

def cpu(d):
 prep(d); p=Path(d['dir'])
 for family in ['inductive','transductive']:
  for name in TEACHERS[:3]:
   dst=p/family/name
   if complete(dst) or reuse(d,family,name): continue
   run('scripts/run_leakage_safe_method.py','--method',name,*common(d),'--output',dst,*(['--transductive'] if family=='transductive' else []))
 if not complete(p/'inductive/magic_inductive') and not reuse(d,'inductive','magic_inductive'):
  run('scripts/run_inductive_teacher.py','--method','magic',*common(d),'--output',p/'inductive/magic_inductive','--n-jobs',8,python=MAGIC)
 standard(d,'magic')
def standard(d,name):
 p=Path(d['dir']); dst=p/'standard'/name
 if not complete(dst):
  src=d['standard'].get(name)
  if src:
   if not complete(src): raise FileNotFoundError(f'Existing source not ready; do not duplicate: {src}')
   link(src,dst)
  else:
   args=['--corrupted',p/'hybrid.h5ad','--coordinates',p/'coordinates.parquet','--splits',p/'splits.parquet','--output',dst,'--seed',1729]
   run(f'scripts/run_{name}_baseline.py',*args,*(['--n-jobs',8] if name=='magic' else ['--epochs',200]),python=MAGIC if name=='magic' else SCVI)
 dstcount=p/'transductive'/f'{name}_inductive'
 if not complete(dstcount): run('scripts/count_scale_contract.py','--contract',dst,'--corrupted',p/'hybrid.h5ad','--output',dstcount)
def gpu(d):
 prep(d); p=Path(d['dir'])
 if not complete(p/'inductive/scvi_inductive') and not reuse(d,'inductive','scvi_inductive'):
  run('scripts/run_inductive_teacher.py','--method','scvi',*common(d),'--output',p/'inductive/scvi_inductive',python=SCVI)
 src=d['standard'].get('scvi')
 if src and not complete(src): print('Existing standard scVI still pending:',src,flush=True)
 else: standard(d,'scvi')
def select(d):
 import numpy as np
 from safefusion_benchmark.contracts import write_output_contract
 p=Path(d['dir'])
 if not complete(p/'transductive/scvi_inductive'): standard(d,'scvi')
 for family in ['inductive','transductive']:
  teachers=sum((['--teacher-contract',p/family/n] for n in TEACHERS),[])
  if not complete(p/family/'safe_fusion') and not reuse(d,family,'safe_fusion'):
   run('scripts/run_leakage_safe_method.py','--method','safe_fusion',*common(d),'--output',p/family/'safe_fusion',*teachers,*(['--transductive'] if family=='transductive' else []))
  out=p/family/'selector'
  existing=d.get('selector_source') if d['design']!='masked' else d.get('selector_dir')
  if family=='inductive' and existing and not (out/'calibration_report.json').exists():
   old=Path(existing)
   contract_names=[f'safe_fusion_calibrated_mlp_topk_{str(pct/100).replace(".","p")}' for pct in PCTS]
   if all(complete(old/n) for n in contract_names) and (old/'calibration_report.json').exists():
    for n in contract_names: link(old/n,out/n)
    link(old/'calibration_report.json',out/'calibration_report.json')
  if not (out/'calibration_report.json').exists():
   run(Path('scripts/analyses/transductive_downstream/selector.py'),'--corrupted',p/'hybrid.h5ad','--truth',p/'truth.h5ad','--coordinates',p/'coordinates.parquet','--splits',p/'splits.parquet','--fusion-contract',p/family/'safe_fusion',*teachers,'--output-dir',out,'--fit-split',d['fit_split'],*(['--fit-cells',d['fit_cells']] if d.get('fit_cells') else []),'--architecture','mlp','--budget-mode','apply_topk','--budgets',.01,.05,.1,'--curve-points',2,'--seed',1729,*(['--detection-rule-mask-rate',.1] if family=='transductive' else []))
  sources=[]
  for pct in PCTS:
   suffix=str(pct/100).replace('.','p'); src=out/f'safe_fusion_calibrated_mlp_topk_{suffix}'
   name=('safe_fusion' if family=='inductive' else 'transductive')+f'_{pct}pct'
   sources+=['--source',f'{name}={src}']
   if family=='transductive':
    a=np.load(out/'detection.npz'); values=np.load(src/'mean.npy'); meta=json.loads((src/'metadata.json').read_text())
    rows,cols=a['rows'],a['cols']; values[rows,cols]*=a['detection']
    meta['parameters']['inserted_value']='conditional_detection'; meta['parameters']['detection_mask_rate']=.1
    dest=out/f'conditional_detection_{pct}pct'; write_output_contract(dest,values,meta)
    sources+=['--source',f'detection_{pct}pct={dest}']
  if d['design']=='masked':
   for spec in sources[1::2]:
    name,src=spec.split('=',1); link(src,p/name)
  else: run('scripts/finalize_deployment_contracts.py','--recorded',p/'recorded.h5ad','--splits',p/'splits.parquet',*sources,'--output-root',p)
 for name,src in [('svd',p/'inductive/svd_impute'),('magic',p/'transductive/magic_inductive'),('scvi',p/'transductive/scvi_inductive')]:
  run('scripts/apply_fill_fraction.py','--corrupted',p/'hybrid.h5ad','--splits',p/'splits.parquet','--method-contract',src,'--method-name',name,'--output-root',p/'matched','--fractions',.01,.05,.1)
  sources=[]
  for pct in PCTS:
   src=p/'matched'/f'{name}_topk_{str(pct/100).replace(".","p")}'
   if d['design']=='masked': link(src,p/f'{name}_{pct}pct')
   else: sources+=['--source',f'{name}_{pct}pct={src}']
  if sources: run('scripts/finalize_deployment_contracts.py','--recorded',p/'recorded.h5ad','--splits',p/'splits.parquet',*sources,'--output-root',p)
 if d['design']=='masked':
  sources=sum((['--source',f'{n}_{pct}pct={p}/{n}_{pct}pct'] for n in NAMES for pct in PCTS),[])
  run('scripts/decompose_fills.py','--corrupted',p/'hybrid.h5ad','--coordinates',p/'coordinates.parquet','--splits',p/'splits.parquet',*sources,'--output-root',p/'decomposition')
 (p/'READY').write_text('completed\n')

def evaluate(d,part='full'):
 p=Path(d['dir']); design=d['design']; key=d['key']; tissue=key.split('_')[0]
 if design=='null': return
 methods=[]
 for n in NAMES:
  for pct in PCTS:
   name=f'{n}_{pct}pct'; src=p/name if part=='full' else p/'decomposition'/f'{name}__{part}'
   methods+=['--method',f'{name}={src}']
 corrupted=p/('hybrid.h5ad' if design=='masked' else 'recorded.h5ad')
 coords=p/('coordinates.parquet' if design=='masked' else 'empty_coordinates.parquet')
 out=R/design/'evaluation'/key/part
 args=['--truth',p/'truth.h5ad','--corrupted',corrupted,'--splits',p/'splits.parquet',*methods,'--output-dir',out,'--bootstrap',2000,'--seed',1729,'--allow-transductive']
 if tissue=='pancreas':
  run(Path('scripts/analyses/transductive_downstream/pancreas_fold.py'),*args,'--coordinates',coords,'--fold',key[-1])
 elif tissue=='colon': run('scripts/evaluate_colon_donor_biology.py',*args,'--coordinates',coords,'--marker-panel','source')
 elif key=='zebrafish': run('scripts/evaluate_trajectory_preservation.py',*args)
 else: run('scripts/evaluate_interventional_grn.py',*args,'--dataset','norman_crispra','--intervention','gain_of_function','--publication-doi','10.1126/science.aax4438')
 if tissue in ['pancreas','colon']:

  run('scripts/evaluate_unsupervised_clustering.py',*args[:args.index('--output-dir')],'--output-dir',out/'clustering','--label-column','cell_type','--unit-column','donor','--cluster-seeds',20,'--bootstrap',500 if tissue=='pancreas' else 2000,'--seed',1729+int(key[-1]) if tissue=='pancreas' else 1729,'--allow-transductive')

if __name__=='__main__':
 stage=sys.argv[1]; index=int(sys.argv[2]); d=D[index]
 if stage=='select' and (Path(d['dir'])/'READY').exists(): raise SystemExit('Already complete')
 if stage in ['cpu','gpu','select','prep']: globals()[stage](d)
 elif stage=='evaluate': evaluate(d,sys.argv[3] if len(sys.argv)>3 else 'full')
 else: raise ValueError(stage)
