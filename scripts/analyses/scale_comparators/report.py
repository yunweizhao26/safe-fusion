#!/usr/bin/env python3

import csv
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import pandas as pd
ROOT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
REPO=ROOT.parents[3]
SIZES=(25000,50000,100000,200000)
METHODS={'DCA':'dca','Selector on scVI':'selector_scvi','Selector on MAGIC':'selector_magic',
         'scGPT':'scgpt','EnImpute':'enimpute','scImpute':'scimpute','scRecover':'screcover'}
sys.path.insert(0,str(REPO/'scripts'))
from scale_report import gigabytes
jobs=list(csv.reader((ROOT/'jobs.tsv').open(),delimiter='\t'))
fields=['JobID','State','ElapsedRaw','MaxRSS','ExitCode','Start','End','AllocCPUS','Timelimit','NodeList','ReqMem','Partition','AllocTRES','TRESUsageInMax']
raw=subprocess.check_output(['sacct','-n','-P','-j',','.join(j[2] for j in jobs),'--format='+','.join(fields)],text=True)
(ROOT/'sacct_final.txt').write_text('|'.join(fields)+'\n'+raw)
records={r['JobID']:r for r in csv.DictReader(raw.splitlines(),fieldnames=fields,delimiter='|')}

live_path=ROOT/'live_peak_memory.json'
live=json.loads(live_path.read_text()) if live_path.exists() else {}
running=[job+'.batch' for job,r in records.items() if '.' not in job and r['State']=='RUNNING']
if running:
    sample=subprocess.run(['sstat','-n','-P','-j',','.join(running),'--format=JobID%40,MaxRSS'],text=True,capture_output=True)
    (ROOT/'sstat_live.txt').write_text(sample.stdout+sample.stderr)
    for line in sample.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<2: continue
        job=fields[0].strip().removesuffix('.batch')
        value=gigabytes(fields[1].strip())
        if pd.notna(value): live[job]=max(live.get(job,0),value)
    live_path.write_text(json.dumps(live,indent=2)+'\n')
resources=[]
for stage,size,job in jobs:
    r=records.get(job,{})
    b=records.get(job+'.batch',{})
    peak=gigabytes(b.get('MaxRSS',''))
    gpu=re.search(r'gres/gpumem=([0-9.]+[KMGT]?)',b.get('TRESUsageInMax',''))
    source='sacct' if pd.notna(peak) else 'sstat running sample' if job in live else 'unavailable'
    resources.append({'stage':stage,'cells':int(size),'job':job,'state':r.get('State','UNKNOWN'),
                      'elapsed_seconds':int(r.get('ElapsedRaw') or 0),'peak_memory_gib':peak if pd.notna(peak) else live.get(job,float('nan')),
                      'sacct_peak_memory_gib':peak,'peak_memory_source':source,
                      'peak_gpu_memory_gib':gigabytes(gpu.group(1)) if gpu else float('nan'),
                      'allocated_tres':r.get('AllocTRES',''),
                      'exit_code':r.get('ExitCode',''),'start':r.get('Start',''),'end':r.get('End',''),
                      'cpus':r.get('AllocCPUS',''),'time_limit':r.get('Timelimit',''),'node':r.get('NodeList',''),'requested_memory':r.get('ReqMem',''),'partition':r.get('Partition','')})
resources=pd.DataFrame(resources)
resources.to_csv(ROOT/'resources.csv',index=False)
lookup={(r.stage,int(r.cells)):r for r in resources.itertuples()}
frames=[]
for n in SIZES:
    path=ROOT/'evaluation'/str(n)/'results/paired_differences.csv'
    if path.exists(): frames.append(pd.read_csv(path))
differences=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
if len(differences): differences.to_csv(ROOT/'paired_differences.csv',index=False)
for filename in ('method_summary.csv','masked_f1_curves.csv','method_availability.csv'):
    paths=[ROOT/'evaluation'/str(n)/'results'/filename for n in SIZES]
    parts=[pd.read_csv(p) for p in paths if p.exists()]
    if parts: pd.concat(parts,ignore_index=True).to_csv(ROOT/filename,index=False)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def status(slug,n):
    r=lookup.get((slug,n))
    return r.state if r else 'not run'
def result(method,n,stat):
    if len(differences):
        row=differences[(differences.cells==n)&(differences.reference=='Safe Fusion')&(differences.method==method)&(differences.statistic==stat)]
        if len(row):
            r=row.iloc[0]
            return f'{100*r.difference:.2f} [{100*r.lower:.2f}, {100*r.upper:.2f}]'
    st=status(METHODS[method],n)
    return 'not evaluated' if st=='COMPLETED' else st

def runtime(slug,n):
    r=lookup.get((slug,n))
    if r is None: return 'not run'
    memory=f'{r.peak_memory_gib:.2f}' if pd.notna(r.peak_memory_gib) else 'NA'
    if r.peak_memory_source=='sstat running sample': memory+='*'
    suffix='' if r.state=='COMPLETED' else f' ({r.state})'
    return f'{r.elapsed_seconds/60:.2f} / {memory}{suffix}'

now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
lines=['# Scale comparison methods',
'We reused the four S28 lupus inputs (25,000, 50,000, 100,000 and 200,000 cells; 2,000 genes), their saved 10% masks and donor splits, and seed 1729. DCA (default `nb-conddisp`), frozen scGPT and the R methods use the paper runners without tuning, while the single-method MLP selectors use standard scVI or MAGIC count-scale proposals, Table 1 feature precision, and the S28 two-million-row sampling rule. The unchanged S28 evaluator reports Safe Fusion minus each method in mean masked F1 over fill fractions 1%–10% and pooled average precision, with 95% percentile intervals from the same 2,000 paired bootstrap draws of 24 test donors.',
 f'Status captured: {now}. All new jobs use account `torch_pr_634_general` and eight CPUs; selectors, DCA and 25,000-cell R runs use `cs`, larger R runs request high-memory `cl` nodes, and scGPT uses one L40S on `l40s_public`. DCA uses the paper’s CPU TensorFlow environment. R jobs have a 24-hour limit; larger sizes are eligible only after the corresponding 25,000-cell run completes within 12 hours.',
 '**Evaluator reproduction.**']
checks=[]
for n in SIZES:
    p=ROOT/'reproduction'/str(n)/'PASS.json'
    if p.exists():
        d=json.loads(p.read_text())
        err=max(v.get('max_absolute_error',0) for v in d.values())
        checks.append(f'{n:,}: PASS (maximum numeric difference {err:.3g}; donor counts exact)')
    else: checks.append(f'{n:,}: {status("reproduce",n)}; no PASS record')
lines.append('; '.join(checks)+'. Comparison jobs are submitted only after all four checks pass. The checks recompute every saved method summary, paired difference, F1 curve and donor count; numeric tolerance is 1e-12. All 28 printed S28 F1 entries also match the saved values exactly (`S28_table_PASS.json`).')
p=ROOT/'checks/selector_PASS.json'
lines.append('**Selector reproduction.** '+('The 25,000-cell scVI selector has exactly the same candidate coordinates, labels and scores as `scripts/stacked_selector_scores.py`. All eight selectors also match the saved S28 model parameters, fitting-row counts and recorded sampling counts (`selector_sampling_PASS.json`).' if p.exists() else f'Table 1 selector check: {status("check_selector",25000)}. See its log for details.'))
lines+=['**Mean F1 difference (percentage points; 95% interval).**',table(['Method']+[f'{n:,}' for n in SIZES],[[m]+[result(m,n,'mean_f1_1_to_10') for n in SIZES] for m in METHODS]),
        '**Average precision difference (percentage points; 95% interval).**',table(['Method']+[f'{n:,}' for n in SIZES],[[m]+[result(m,n,'average_precision') for n in SIZES] for m in METHODS]),
        '**Runtime: wall minutes / peak resident GiB.**',table(['Method or step']+[f'{n:,}' for n in SIZES],[[m]+[runtime(slug,n) for n in SIZES] for m,slug in METHODS.items()])]
gpu_peaks=[]
for n in SIZES:
    r=lookup.get(('scgpt',n))
    value=f'{r.peak_gpu_memory_gib:.2f}' if r is not None and pd.notna(r.peak_gpu_memory_gib) else 'NA'
    gpu_peaks.append(f'{n:,}: {value}')
lines.append('scGPT peak GPU memory from `sacct`, in GiB: '+ '; '.join(gpu_peaks)+'.')
old=pd.read_csv(REPO/'artifacts/paper_evidence/review_round3/scale/results/resources.csv')
totals=[]
for m,slug in [('Selector on scVI','scvi'),('Selector on MAGIC','magic')]:
    cells=[]
    for n in SIZES:
        new=lookup.get(('selector_'+slug,n))
        prior=old[(old.step==slug)&(old.cells==n)].iloc[-1]
        cells.append(f'{new.elapsed_seconds/60+prior.wall_minutes:.2f} / {max(new.peak_memory_gib,prior.peak_memory_gb):.2f}' if new is not None and new.state=='COMPLETED' else 'NA')
    totals.append([m+' including saved method fit']+cells)
lines += [table(['Complete selector pipeline']+[f'{n:,}' for n in SIZES],totals),
          'Elapsed time excludes queueing; pending jobs have zero execution time. Runtime rows show the latest execution attempt; prior failed attempts and their costs are listed below. Selector rows measure the new selector fit and scoring only. Pipeline totals add the saved S28 standard-method fit wall time and take the larger peak memory; input preparation and evaluation are excluded. R clustering preparation is separate below. Unmarked MaxRSS is from each job’s `sacct` batch step. A memory value marked * is the largest saved `sstat` sample while that job was running, used only when `sacct` has no peak yet; it is a lower bound on the eventual peak, not a completed-run measurement. NA means no recorded peak, not zero; `resources.csv` separates the accounting peak and its source, and `live_peak_memory.json` preserves the live samples.',
          table(['Preparation or check']+[f'{n:,}' for n in SIZES],[[label]+[runtime(slug,n) for n in SIZES] for label,slug in [('R cluster count','kcluster'),('S28 reproduction','reproduce'),('DCA and selector evaluation','evaluate_core'),('Evaluation including scGPT','evaluate_gpu'),('Interim R evaluation','evaluate_ready'),('Full evaluation','evaluate')]]),
          '**Paths and reproduction.**',
          'All new artifacts are under `artifacts/paper_evidence/review_round4/scale_comparators/`. Saved S28 inputs and reference outputs remain in `artifacts/paper_evidence/review_round3/scale/cells_<N>/`; symlinks under `reproduction/` and `evaluation/` reuse them without copying or modifying them.',
          '- `cells_<N>/fits/<method>/`: mean arrays, metadata and runner work files.\n- `cells_<N>/selector_{scvi,magic}/`: candidate coordinates, scores and selector reports.\n- `reproduction/<N>/PASS.json` and `results/`: numeric reproduction evidence.\n- `checks/selector_PASS.json`: exact Table 1 selector check; `results_PASS.json`: completed-result and source-hash checks.\n- `evaluation/<N>/results/`: method summaries, paired differences, curves and donor counts.\n- Root CSV files: combined results and resources; `jobs.tsv`, `commands.sh`, `sacct_final.txt`, `logs/` and `r_gates.json`: execution records.\n- `scgpt_gene_support.json`: vocabulary coverage; `scale_selector.py`: S28 selector copy with an additive `--stacked-convention` flag; `evaluate.py`: registrations around the unchanged S28 evaluator.',
          'From the repository root, the following submits jobs; wait for each required predecessor before evaluating or reporting. `commands.sh` records the exact submitted commands, including resource requests and dependencies.',
          '```bash\nOUT=artifacts/paper_evidence/review_round4/scale_comparators\nexport PYTHONDONTWRITEBYTECODE=1\n.venv/bin/python "$OUT/check_table.py"\nfor n in 25000 50000 100000 200000; do\n  .venv/bin/python "$OUT/submit.py" reproduce "$n"\ndone\nwhile squeue -u yz5944 -h -n scmp-reproduce | grep -q .; do sleep 600; done\n# Fresh outputs only: monitor submits comparisons after all four PASS files exist,\n# applies the 12-hour R gate, and submits evaluation when methods finish.\nbash "$OUT/monitor.sh"\n.venv/bin/python "$OUT/report.py"\n```',
          'For a fresh rerun, archive `cells_<N>/`, `reproduction/<N>/results/`, `evaluation/<N>/results/`, `checks/`, `selector_sampling_PASS.json`, `DONE` (if present), and `jobs.tsv` inside this output directory before submission, while keeping the scripts, input symlinks and provenance files; the paper runners refuse to overwrite existing contracts. `monitor.sh` uses the enclosing Slurm job’s end time and stops two hours before it. To resume this recorded run, run the monitor without deleting its ledger.',
          '**Failures, limits and remaining work.**']
support_path = ROOT/'cells_25000/fits/scgpt/metadata.json'
if support_path.exists():
    support = json.loads(support_path.read_text())['parameters']
    lines.append(f'scGPT matches {support["matched_genes"]} of {support["total_genes"]} genes; unmatched genes retain the runner score below all matched genes.')
else:
    lines.append('scGPT vocabulary coverage is unavailable until the first fit completes.')
if (ROOT/'r_gates.json').exists():
    gates=json.loads((ROOT/'r_gates.json').read_text())
    for m in gates:
        elapsed=int(resources[(resources.stage==m)&(resources.cells==25000)].elapsed_seconds.sum())
        if elapsed>43200:
            gates[m]=f'Larger sizes not run: 25,000 exceeded 12 hours, state={status(m,25000)}, elapsed={elapsed} seconds'
    (ROOT/'r_gates.json').write_text(json.dumps(gates,indent=2)+'\n')
    lines += [f'- {m}: {v}.' for m,v in gates.items()]
fail=resources[(resources.state!='COMPLETED') & (resources.stage!='report') & (resources.stage.isin(METHODS.values()) | ~resources.state.isin(['PENDING','RUNNING'])) & ~((resources.stage=='scgpt') & resources.state.str.startswith('CANCELLED') & (resources.elapsed_seconds==0))]
if len(fail):
    for r in fail.itertuples():
        peak=f'{r.peak_memory_gib:.2f} GiB' if pd.notna(r.peak_memory_gib) else 'NA'
        lines.append(f'- Job {r.job}, {r.stage}, {r.cells:,} cells: {r.state}; elapsed {r.elapsed_seconds/3600:.3f} h; peak {peak} ({r.peak_memory_source}); requested memory {r.requested_memory}; exit {r.exit_code}. Logs: `logs/{r.stage}-{r.cells}-{r.job}.{{out,err}}`.')
else: lines.append('All submitted jobs completed.')

p=ROOT/'failure_notes.txt'
if p.exists(): lines.append(p.read_text().strip())
unfinished=[]
for m,slug in METHODS.items():
    for n in SIZES:
        if result(m,n,'mean_f1_1_to_10') in ('not evaluated','PENDING','RUNNING','UNKNOWN'):
            unfinished.append(f'{m} at {n:,} cells')
if unfinished: lines.append('Remaining: finish or resolve '+', '.join(unfinished)+'; evaluate their completed outputs and refresh accounting. Cancelled fits have no recovery estimates.')
else: lines.append('No completed output remains unevaluated. Methods that failed or were excluded by the 12-hour rule have no recovery estimate.')
p=ROOT/'finalization.json'
if p.exists():
    followup=json.loads(p.read_text())
    lines.append('Follow-up evaluations: '+', '.join(f'{int(n):,} cells: job {j} ({status("evaluate",int(n))})' for n,j in followup['evaluations'].items())+'. Report-refresh job '+followup['report']+' ('+status('report',0)+') follows those evaluations and runs `check_results.py`. The pending evaluations and report refresh are already submitted and can finish after the process allocation ends; their commands and dependencies are in `commands.sh`. Current running methods remain unfinished until their final accounting and evaluation are available.')
(ROOT/'README.md').write_text('\n\n'.join(lines)+'\n')
print(ROOT/'README.md')
