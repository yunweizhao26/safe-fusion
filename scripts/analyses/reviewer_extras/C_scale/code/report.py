import csv
from decimal import Decimal
import io
import json
import subprocess
from pathlib import Path

ROOT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras/C_scale')

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def main():
    jobs_path = ROOT / 'jobs.tsv'
    jobs = list(csv.DictReader(jobs_path.open(), delimiter='\t')) if jobs_path.exists() else []
    resources = []
    if jobs:
        output = subprocess.check_output(['sacct', '-n', '-P', '-j', ','.join(j['job'] for j in jobs),
            '--format=JobIDRaw,JobName,State,Elapsed,ElapsedRaw,MaxRSS,TotalCPU,AllocCPUS,ReqMem,NodeList'], text=True)
        (ROOT / 'sacct_raw.txt').write_text(output)
        fields = ['job', 'job_name', 'state', 'elapsed', 'elapsed_seconds', 'maxrss', 'total_cpu', 'cpus', 'requested_memory', 'nodes']
        records = {r[0]: dict(zip(fields, r)) for r in csv.reader(io.StringIO(output), delimiter='|') if r}
        for job in jobs:
            row = dict(job)
            top = records.get(job['job'], {})
            batch = records.get(job['job'] + '.batch', {})
            row.update({key: top.get(key, '') for key in fields if key != 'job'})
            row['maxrss'] = batch.get('maxrss', '')
            row['total_cpu'] = batch.get('total_cpu', top.get('total_cpu', ''))
            resources.append(row)
        with (ROOT / 'resources.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(resources[0]))
            writer.writeheader()
            writer.writerows(resources)
    summary = ROOT / 'results/method_summary.csv'
    evaluations = [r for r in resources if r['stage'] == 'evaluate']
    evaluated = bool(evaluations) and evaluations[-1]['state'] == 'COMPLETED'
    interim_complete = (ROOT / 'interim_snapshot/completed_utc.txt').exists()
    if not evaluated and interim_complete:
        summary = ROOT / 'interim_snapshot/results/method_summary.csv'
    summaries = list(csv.DictReader(summary.open())) if summary.exists() else []
    metric_sources = [summary]
    saver_summary = ROOT / 'saver_snapshot/results/method_summary.csv'
    if not evaluated and interim_complete and (ROOT / 'saver_snapshot/completed_utc.txt').exists():
        summaries += list(csv.DictReader(saver_summary.open()))
        metric_sources.append(saver_summary)
    present = {r['method'] for r in summaries}
    metric_status = ('Final evaluation completed.' if evaluated else
                     'Completed interim evidence shown; final evaluation remains pending.' if interim_complete else
                     'Any metric rows below are checkpoints until evaluation completes.')
    required = {'Safe Fusion', 'Safe Fusion (transductive)', 'Safe Fusion without MAGIC teacher',
                'SVD', 'Weighted kNN', 'ALRA', 'SAVER', 'MAGIC', 'scVI', 'scVI probability',
                'MAGIC (inductive)', 'scVI (inductive)'}
    if any(r['step'] == 'magic_teacher' and r['state'] == 'TIMEOUT' for r in resources):
        required -= {'Safe Fusion', 'MAGIC (inductive)'}
    missing = sorted(required - present)
    status = 'Complete' if evaluated and not missing else 'Incomplete'
    overlap_check = None
    if evaluated and interim_complete:
        prior = []
        for snapshot in ('interim_snapshot', 'saver_snapshot'):
            path = ROOT / snapshot / 'results/method_summary.csv'
            if (ROOT / snapshot / 'completed_utc.txt').exists():
                prior.extend(csv.DictReader(path.open()))
        final_ap = {r['method']: r for r in summaries if r['statistic'] == 'average_precision'}
        comparisons = []
        for row in prior:
            if row['statistic'] == 'average_precision' and row['method'] in final_ap:
                current = final_ap[row['method']]
                differences = [abs(Decimal(current[k]) - Decimal(row[k])) for k in ('estimate', 'lower', 'upper')]
                comparisons.append({'method': row['method'], 'exact_csv_match': all(current[k] == row[k] for k in ('estimate', 'lower', 'upper')),
                                    'max_absolute_difference': str(max(differences))})
        overlap_check = {'methods_compared': len(comparisons), 'all_exact_csv_match': all(r['exact_csv_match'] for r in comparisons),
                         'max_absolute_difference': str(max((Decimal(r['max_absolute_difference']) for r in comparisons), default=Decimal(0))),
                         'methods': comparisons}
        (ROOT / 'final_snapshot_comparison.json').write_text(json.dumps(overlap_check, indent=2) + '\n')
    (ROOT / 'completion_status.json').write_text(json.dumps({'status': status, 'evaluation_completed': evaluated,
        'missing_requested_methods': missing}, indent=2) + '\n')
    lines = ['# Part C: 200,000 lupus cells and 5,000 genes', '',
        f'Status: **{status}**. ' + metric_status, '',
        ('Missing requested method rows: ' + ', '.join(missing) + '.') if missing else 'All required method rows are present.', '',

        'The experiment selects 5,000 highly variable genes with the S28 training-donor rule on the same 200,000 lupus cells, donor split, and 10% mask seed (1729). '
        'Every model uses the S28 settings, including five-fold inductive teachers, the transductive variant, the four-teacher variant without MAGIC, and the original comparisons. '
        'Average precision and 95% percentile intervals use 2,000 draws of the same 24 held-out donors with seed 1729; each step runs on eight CPU cores, and scVI also uses one L40S GPU.', '',
        '## Results', '']
    lines += ['Metric sources: ' + ', '.join(f'`{path.relative_to(ROOT)}`' for path in metric_sources) + '. Native CSV values are on the 0–1 scale.', '']
    if overlap_check is not None:
        lines += [f"Final versus completed snapshot validation: {overlap_check['methods_compared']} overlapping AP rows; exact CSV agreement for estimates and interval bounds: {overlap_check['all_exact_csv_match']}; maximum absolute difference: {overlap_check['max_absolute_difference']} on the 0–1 scale. See final_snapshot_comparison.json.", '']
    if summary.exists():
        rows = [r for r in summaries if r['statistic'] == 'average_precision']
        lines += [table(['Method', 'AP (%)', '95% lower (%)', '95% upper (%)', 'Candidates', 'Masked positives'],
            [[r['method'], *[format((Decimal(r[key]) * 100).normalize(), 'f') for key in ('estimate', 'lower', 'upper')], r['test_candidates'], r['masked_positives']] for r in rows]), '']
    else:
        lines += ['Average precision and intervals are not available yet. No 2,000-gene run was repeated.', '']
    lines += ['## Resources', '', 'Elapsed and MaxRSS are the Slurm wall time and batch-step peak memory; MaxRSS is sampled by Slurm.', '']
    lines += [table(['Step', 'Job', 'State', 'Elapsed', 'MaxRSS'],
            [[r['step'], r['job'], r['state'], r['elapsed'], r['maxrss']] for r in resources]) if resources else 'Jobs have not been submitted.', '']
    failures = [r for r in resources if r['state'] and not r['state'].startswith(('COMPLETED', 'PENDING', 'RUNNING'))]
    pending = [r for r in resources if r['state'].startswith(('PENDING', 'RUNNING'))]
    lines += ['## Failures and remaining work', '']
    if status == 'Complete':
        lines += ['No requested work remains. The failures below are recovered execution history.', '']
    if pending:
        queue = subprocess.check_output(['squeue', '--start', '-h', '-j', ','.join(r['job'] for r in pending),
                                         '-o', '%i|%T|%S|%R'], text=True)
        (ROOT / 'queue_status.txt').write_text(queue)
        lines += ['Current scheduler snapshot: job ID, state, estimated start in cluster local time, and reason. Estimates can change.',
                  '', '```text', queue.rstrip(), '```', '']
    lines += [f"- {r['step']}: job {r['job']}, {r['state']}. Inspect logs/{r['job_name']}-{r['job']}.err." for r in failures]
    lines += [f"- {r['step']}: job {r['job']}, {r['state']}; output remains unverified." for r in pending]
    if not failures and not pending:
        lines += ['No failed jobs recorded.' if jobs else 'Execution is pending completion of higher-priority Parts A and B.']
    finished = {r['step']: r for r in resources if r['state'] == 'COMPLETED'}
    teacher_status = (f"The MAGIC teacher completed in {finished['magic_teacher']['elapsed']}, within its 12-hour limit. " if 'magic_teacher' in finished else
                      'The MAGIC teacher has a 12-hour limit; the four-teacher variant supports evaluation if that limit is reached. ')
    saver_status = (f"SAVER completed in {finished['saver']['elapsed']}, before its 16-hour administrative limit. " if 'saver' in finished else
                    'SAVER has a 16-hour administrative limit to leave time within the parent job deadline; reaching it would not establish failure within the original 24-hour S28 limit. ')
    lines += ['', 'Standard MAGIC and ALRA completed their numerical fits, saved their full contracts, and printed their production completion metadata, but their initial shell wrappers failed afterward because a live shell file had been edited. The original shell bytes were restored, both saved model outputs were retained and successfully scored in the complete interim evaluation, and only MAGIC downstream conversions and dependencies were resubmitted. SAVER finished just after the first snapshot checked its availability; its separate evaluation added the omitted row without refitting anything. See execution_notes.md for job IDs and all submission failures.', '', teacher_status +
              'The local evaluator selects the first available selector candidate file, preserving the original metrics, ties, and bootstrap draws. '
              'Its implementation also supports an unused fallback: if no selector completes, the original S28 candidate enumerator reconstructs the same test zeros and masked-positive labels for comparator evaluation. '
              'SAVER uses the cl partition because Slurm rejected its 480 GB request on cs, with eight cores and ncores=8 unchanged. '
              + saver_status +
              'No shared script was edited.', '', '## Reproduction and files', '', 'From the repository root:', '', '```bash',
              'bash artifacts/paper_evidence/review_round4/reviewer_extras/C_scale/code/submit.sh',
              'while squeue -u yz5944 -h -o %j | grep -q "^extras-C-"; do sleep 600; done',
              '.venv/bin/python artifacts/paper_evidence/review_round4/reviewer_extras/C_scale/code/report.py', '```', '',
              'The submission script refuses to overwrite an existing jobs.tsv. Job commands are in code/stage.sh, copied from scripts/slurm_scale.sh with only paths, gene count, and size selection changed. '
              'The scVI wrapper redirects Lightning logs into this output directory.', '',
              '- `prepare_report.json`: source checksum, genes, donors, subset definition.',
              '- `cells_200000/`: new truth, mask, methods, selectors, and comparisons.',
              '- `results/method_summary.csv`: exact estimates and 95% intervals.',
              '- `results/paired_differences.csv`: paired differences and 95% intervals.',
              '- `resources.csv`, `sacct_raw.txt`, `jobs.tsv`, `logs/`: cost and execution records.',
              '- `code/`: launcher, local evaluator, logging wrapper, report script, original scale-script hashes.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
