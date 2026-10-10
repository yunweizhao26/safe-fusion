import json
from pathlib import Path
import subprocess

round4 = Path('artifacts/paper_evidence/review_round4')
code = Path('scripts/analyses')
ledger = round4 / 'published_downstream/jobs.jsonl'
if ledger.exists():
    raise FileExistsError('Existing submission; inspect its jobs before starting a new run')

def submit(analysis, stage, args=(), cores=2, memory='8G', hours='01:00:00', array=None, dependencies=(), success=True, gpu=False, threads=None):
    out = round4 / analysis
    cmd = ['sbatch', '--parsable', '-A', 'torch_pr_634_general', '-p', 'h200_courant' if gpu else 'cs',
           '-c', str(cores), '--mem', memory, '-t', hours, '--job-name', 'pub-' + analysis.removeprefix('published_') + '-' + stage,
           '-o', str(out / 'logs' / (stage + '-%A_%a.log'))]
    if array:
        cmd += ['--array', array]
    if gpu:
        cmd += ['--gres', 'gpu:h200:1']
    if threads:
        cmd += ['--export', f'ALL,NUM_THREADS={threads}']
    if dependencies:
        cmd += ['--dependency', ('afterok:' if success else 'afterany:') + ':'.join(dependencies)]
    script = 'evaluate' if stage.startswith('evaluate') else stage
    cmd += [str(code / 'published_downstream/job.sh'), str(code / analysis / (script + '.py')), *args]
    job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
    with ledger.open('a') as stream:
        stream.write(json.dumps(dict(analysis=analysis, stage=stage, args=args, job_id=job, command=cmd)) + '\n')
    print(analysis, stage, ' '.join(args), job, flush=True)
    return job

downstream = 'published_downstream'
protein = 'published_protein'
allcells = 'published_downstream_allcells'
kcluster = submit(downstream, 'fit', ['kcluster'], 4, '16G', array='0-9')
fits = {}
for method, cores, memory, hours in [('saver', 8, '120G', '12:00:00'), ('dca', 8, '64G', '04:00:00'),
                                     ('scimpute', 16, '96G', '18:00:00'), ('screcover', 16, '96G', '18:00:00'),
                                     ('enimpute', 16, '96G', '18:00:00'), ('alra', 8, '48G', '02:00:00'),
                                     ('scgpt', 8, '64G', '02:00:00')]:
    fits[method] = submit(downstream, 'fit', [method], cores, memory, hours,
                          array='8-9' if method in ['alra', 'scgpt'] else '0-9',
                          dependencies=[kcluster] if method in ['dca', 'scimpute', 'screcover', 'enimpute'] else [],
                          gpu=method == 'scgpt')
evaluations = []
for method in ['detection', 'transductive', 'svd', 'magic', 'scvi', 'alra', 'selector_scvi', 'selector_magic',
               'saver', 'dca', 'scimpute', 'screcover', 'enimpute']:
    evaluations.append(submit(downstream, 'evaluate', [method], 8, '48G', '04:00:00', array='0-7',
                              dependencies=[fits[method]] if method in fits and method != 'alra' else [],
                              success=False, threads=2))
summary = submit(downstream, 'summarize', dependencies=evaluations)
protein_evaluations = []
for method in ['existing', *fits]:
    for index, block in [(8, 'small'), (9, 'full')]:
        protein_evaluations.append(submit(protein, 'evaluate-' + method + '-' + block, [method, block], 3, '16G',
                                          dependencies=[fits[method] + '_' + str(index)] if method != 'existing' else [],
                                          success=False, threads=1))
submit(protein, 'summarize', dependencies=protein_evaluations, success=False)
refits = []
for method, memory in [('svd_all_cells', '48G'), ('alra_all_cells', '64G')]:
    fit = submit(allcells, 'fit', [method], 8, memory, '04:00:00', array='0-7', threads=8)
    refits.append(submit(allcells, 'evaluate', [method], 8, '48G', '04:00:00', array='0-7', dependencies=[fit], threads=2))
submit(allcells, 'summarize', dependencies=[summary, *refits])
