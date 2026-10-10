from common import *
import json
import subprocess

mode = sys.argv[1]
n = 28 if mode == 'replicate' else 7
records = []
def submit(stage, dependencies=(), gpu=False):
    cmd = ['sbatch','--parsable','-A','torch_pr_634_general','-J',f'sf-r4-{mode}-{stage}',
           f'--array=0-{n-1}', '-o',str(OUT/'logs'/f'{mode}-{stage}-%A_%a.log')]
    if gpu:
        cmd += ['-p','l40s_public','--gres=gpu:l40s:1','--time=00:45:00']
    if dependencies:
        cmd += ['--dependency=afterok:'+':'.join(dependencies)]
    cmd += [str(Path('scripts/analyses/transductive_main/run.sh')),'fit.py',mode,stage]
    job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
    records.append(dict(job=job, command=cmd))
    return job
p = submit('prepare')
c = submit('cpu',[p])
g = submit('gpu',[p],True)
f = submit('fit',[c,g])
(OUT/f'{mode}_jobs.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))
