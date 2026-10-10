from common import *
import os, subprocess, time
method=sys.argv[1]; unit=UNITS[int(os.environ['SLURM_ARRAY_TASK_ID'])]; b=base(unit)
out=O/'fits'/method/unit
if method=='screcover':
    audit=json.loads((O/'audit'/f'{unit}.json').read_text())
    if audit['reuse']:
        print('Audited legacy reuse:',unit,flush=True);sys.exit(0)
    cluster=R4/'sex_zero_comparators/kcluster'/f'{unit}.json' if unit not in UNITS[:2] else R3/'kcluster'/f'deploy_{unit}.json'
    assert cluster.exists()
    cmd=[str(ROOT/'.venv/bin/python'),'scripts/standard_imputers/run_r3_comparator.py','--method','screcover','--corrupted',str(b/'hybrid.h5ad'),'--coordinates',str(b/'coordinates.parquet'),'--splits',str(b/'splits.parquet'),'--output',str(out),'--kcluster-json',str(cluster),'--ncores',os.environ['SLURM_CPUS_PER_TASK'],'--seed','1729']
else:
    assert method=='scgpt'
    cmd=[str(ROOT/'.venv-baselines/bin/python'),'scripts/run_scgpt_gene_prediction.py','--corrupted',str(b/'hybrid.h5ad'),'--coordinates',str(b/'coordinates.parquet'),'--ckpt-dir','external_data/baselines/scgpt_human','--output',str(out),'--batch-size','64','--gene-batch-size','512','--max-length','1200','--device','cuda','--seed','1729']
start=time.time(); print(json.dumps(dict(unit=unit,method=method,command=cmd)),flush=True)
result=subprocess.run(cmd)
dump(O/'logs'/f'{method}_{unit}_execution.json',dict(unit=unit,method=method,command=cmd,job_id=os.environ.get('SLURM_JOB_ID'),array_id=os.environ.get('SLURM_ARRAY_JOB_ID'),exit_code=result.returncode,elapsed_seconds=time.time()-start))
sys.exit(result.returncode)
