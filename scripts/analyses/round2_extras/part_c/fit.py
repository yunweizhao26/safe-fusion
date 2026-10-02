import os,sys,json,subprocess
from pathlib import Path
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_c')
u=json.loads((O/'units.json').read_text())[7+int(os.environ['SLURM_ARRAY_TASK_ID'])]
dest=Path(u['destination']); dest.mkdir(parents=True,exist_ok=True)
for name in ['NUMBA_CACHE_DIR','XDG_CACHE_HOME','TMPDIR']:
    path=dest/'cache'/os.environ['SLURM_JOB_ID']/name
    path.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(path)
seed=str(u['seed']); mode=sys.argv[1]
def run(python,script,args,done,threads=16):
    if Path(done).exists(): return
    command=[python,script,*map(str,args)]
    with (dest/f'commands_{mode}.jsonl').open('a') as f: f.write(json.dumps(command)+'\n')
    env=dict(os.environ)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        env[name]=str(threads)
    subprocess.run(command,check=True,env=env)
    assert Path(done).exists(),done
common=['--corrupted',u['corrupted'],'--coordinates',u['coordinates'],'--splits',u['splits']]
def selector(method):
    contract=u['extras']['EnImpute'] if method=='EnImpute' else u['contracts'][method]
    output=Path(u['selectors'][method]).parent
    run('.venv/bin/python','scripts/stacked_selector_scores.py',[*common,'--fit-split',u['fit_split'],'--unit-column',u['unit_column'],'--contract',contract,'--name',method,'--output-dir',output,'--seed',seed],output/'test_scores.npz',threads=8)
if mode=='enimpute':
    k=dest/'kcluster_paper.json'
    run('.venv-scanpy/bin/python','scripts/standard_imputers/r3_kcluster.py',['--corrupted',u['corrupted'],'--output',k,'--seed',seed],k,threads=4)
    output=Path(u['extras']['EnImpute'])
    run('.venv/bin/python','scripts/standard_imputers/run_r3_comparator.py',[*common,'--method','enimpute','--output',output,'--kcluster-json',k,'--ncores','16','--seed',seed],output/'metadata.json')
    selector('EnImpute')
elif mode=='fast':
    output=Path(u['extras']['DCA'])
    run('.venv/bin/python','scripts/standard_imputers/run_r3_comparator.py',[*common,'--method','dca','--output',output,'--ncores','16','--seed',seed],output/'metadata.json')
    output=Path(u['extras']['scVI probability'])
    run('.conda-scvi-current/bin/python','scripts/nonzero_probability.py',['--method','scvi','--contract',u['contracts']['scVI'],'--corrupted',u['corrupted'],'--output',output],output/'metadata.json')
    for method in ['scVI','MAGIC']: selector(method)
else: raise ValueError(mode)
(dest/f'{mode}_complete.json').write_text(json.dumps(dict(seed=u['seed'],unit=u['key'],job=os.environ['SLURM_JOB_ID']))+'\n')
