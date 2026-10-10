#!/usr/bin/env python3

import argparse
from pathlib import Path
import shlex
import subprocess
ROOT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
REPO = ROOT.parents[3]
p=argparse.ArgumentParser()
p.add_argument('stage')
p.add_argument('size', type=int)
p.add_argument('--dependency')
p.add_argument('--memory', type=int)
a=p.parse_args()
mem=a.memory or (128 if a.stage in ('scimpute','screcover','enimpute') else 64)
command=['sbatch','--parsable','--account=torch_pr_634_general','--chdir',str(REPO),
         '-J','scmp-'+a.stage,'--cpus-per-task=8',f'--mem={mem}G','--time=24:00:00',
         '-o',str(ROOT/'logs'/f'{a.stage}-{a.size}-%j.out'),
         '-e',str(ROOT/'logs'/f'{a.stage}-{a.size}-%j.err')]

command += ['-p','l40s_public','--gres=gpu:l40s:1','--time=01:00:00'] if a.stage=='scgpt' else ['-p','cl' if a.stage in ('enimpute','scimpute','screcover') and mem>480 else 'cs']
if a.stage.startswith('evaluate') or a.stage in ('kcluster','reproduce','check_selector','report'):
    command += ['--time=01:00:00']
if a.dependency:
    kind,*ids=a.dependency.split(':')
    raw=subprocess.check_output(['sacct','-n','-P','-j',','.join(ids),'--format=JobID,State'],text=True)
    states={r[0]:r[1].split()[0] for line in raw.splitlines() if len(r:=line.split('|'))>1}
    active={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED'}
    if kind=='afterok' and any(states.get(j,'UNKNOWN') not in active|{'COMPLETED','UNKNOWN'} for j in ids):
        raise RuntimeError('Required predecessor failed: '+a.dependency)

    ids=[j for j in ids if states.get(j,'UNKNOWN') in active|{'UNKNOWN'}]
    if ids: command += ['--dependency',kind+':'+':'.join(ids)]
command += [str(ROOT/'job.sh'),a.stage,str(a.size)]
job=subprocess.check_output(command,text=True).strip().split(';')[0]
with (ROOT/'jobs.tsv').open('a') as f: f.write(f'{a.stage}\t{a.size}\t{job}\n')
with (ROOT/'commands.sh').open('a') as f: f.write(shlex.join(command)+'\n')
print(job,flush=True)
