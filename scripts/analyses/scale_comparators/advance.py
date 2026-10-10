#!/usr/bin/env python3

import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
lock=(ROOT/'.advance.lock').open('w')
try:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
SIZES=(25000,50000,100000,200000)
R_METHODS=('enimpute','scimpute','screcover')
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED'}
rows=list(csv.reader((ROOT/'jobs.tsv').open(),delimiter='\t'))
ids=[r[2] for r in rows]
raw=subprocess.check_output(['sacct','-n','-P','-j',','.join(ids),'--format=JobID,State,ElapsedRaw,MaxRSS,ExitCode,Start,End,AllocCPUS,Timelimit'],text=True)
(ROOT/'sacct.tsv').write_text('JobID|State|ElapsedRaw|MaxRSS|ExitCode|Start|End|AllocCPUS|Timelimit\n'+raw)
records={r[0]:r for line in raw.splitlines() if (r:=line.split('|')) and '.' not in r[0]}
bykey={(stage,int(size)):job for stage,size,job in rows}
def state(job): return records.get(job,['','UNKNOWN'])[1].split()[0]
def submit(stage,size,dependency=None,memory=None):
    if (stage,size) in bykey: return bykey[stage,size]
    command=[sys.executable,str(ROOT/'submit.py'),stage,str(size)]
    if dependency: command+=['--dependency',dependency]
    if memory: command+=['--memory',str(memory)]
    job=subprocess.check_output(command,text=True).strip()
    bykey[stage,size]=job
    print('submitted',stage,size,job,flush=True)
    return job

if not all((ROOT/'reproduction'/str(n)/'PASS.json').exists() and state(bykey['reproduce',n])=='COMPLETED' for n in SIZES):
    failures=[(n,state(bykey['reproduce',n])) for n in SIZES if state(bykey['reproduce',n]) not in ACTIVE|{'COMPLETED','UNKNOWN'}]
    if failures:
        (ROOT/'BLOCKED.json').write_text(json.dumps({'reproduction_failed':failures},indent=2)+'\n')
        sys.exit(2)
    print('Waiting for all S28 reproduction checks',flush=True)
    sys.exit(0)

k=submit('kcluster',25000)
for method in R_METHODS:
    submit(method,25000,'afterok:'+k)
for n in SIZES:
    for method in ('selector_scvi','selector_magic','dca','scgpt'):
        submit(method,n)
submit('check_selector',25000,'afterok:'+bykey['selector_scvi',25000])

gate={}
for method in R_METHODS:
    job=bykey[method,25000]
    st=state(job)
    attempts=[j for stage,size,j in rows if stage==method and int(size)==25000]
    elapsed=sum(int(records[j][2]) for j in attempts if j in records)
    if st=='COMPLETED' and elapsed<=43200 and (ROOT/'cells_25000/fits'/method/'mean.npy').exists():
        gate[method]='25,000 completed within 12 hours; larger sizes eligible'
        for n,mem in ((50000,1024),(100000,2800),(200000,2800)):
            k=submit('kcluster',n)
            submit(method,n,'afterok:'+k,memory=mem)
    elif elapsed>43200:
        gate[method]=f'Larger sizes not run: 25,000 exceeded 12 hours, state={st}, elapsed={elapsed} seconds'
    elif st in ACTIVE|{'UNKNOWN'}:
        gate[method]='Waiting for 25,000 cells'
    else:
        gate[method]=f'Larger sizes not run: 25,000 state={st}, elapsed={elapsed} seconds'
(ROOT/'r_gates.json').write_text(json.dumps(gate,indent=2)+'\n')

if not (ROOT/'selector_sampling_PASS.json').exists() and all(state(bykey[m,n])=='COMPLETED' for n in SIZES for m in ('selector_scvi','selector_magic')):
    subprocess.run([sys.executable,str(ROOT/'check_sampling.py')],check=True)


for n in SIZES:
    if all(state(bykey[m,n])=='COMPLETED' for m in ('selector_scvi','selector_magic','dca')):
        submit('evaluate_core',n)
    if state(bykey['scgpt',n])=='COMPLETED' and ('evaluate_core',n) in bykey and state(bykey['evaluate_core',n])=='COMPLETED':
        submit('evaluate_gpu',n)


for n in SIZES:
    required=['selector_scvi','selector_magic','dca','scgpt']
    if n==25000: required+=list(R_METHODS)
    else:
        if any(v.startswith('Waiting') for v in gate.values()): continue
        required += [m for m in R_METHODS if (m,n) in bykey]
    jobs=[bykey[m,n] for m in required]
    if all(state(j) not in ACTIVE|{'UNKNOWN'} for j in jobs):
        prior=[bykey[stage,n] for stage in ('evaluate_core','evaluate_gpu','evaluate_ready') if (stage,n) in bykey]
        submit('evaluate',n,'afterany:'+':'.join(prior) if prior else None)
if all(('evaluate',n) in bykey and state(bykey['evaluate',n])=='COMPLETED' for n in SIZES) and state(bykey['check_selector',25000]) not in ACTIVE|{'UNKNOWN'}:
    (ROOT/'DONE').write_text('All requested applicable jobs and evaluation have finished.\n')
    print('DONE',flush=True)
print(json.dumps({'completed':sum(state(j)=='COMPLETED' for j in bykey.values()),'unfinished':{f'{s}/{n}':state(j) for (s,n),j in bykey.items() if state(j)!='COMPLETED'}},sort_keys=True),flush=True)

subprocess.run([sys.executable,str(ROOT/"report.py")],check=True)
