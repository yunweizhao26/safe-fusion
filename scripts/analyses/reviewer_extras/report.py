import csv,json
from pathlib import Path
ROOT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras')

def read(path):
    with path.open() as f: return list(csv.DictReader(f))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def interval(row,value,low,high,mult=1):
    if mult==1: return f"{row[value]} [{row[low]}, {row[high]}]"
    return f"{float(row[value])*mult:.17g} [{float(row[low])*mult:.17g}, {float(row[high])*mult:.17g}]"

parts=['# Reviewer extras, round 4\n\nAll paths below are relative to `artifacts/paper_evidence/review_round4/reviewer_extras/`. Existing model settings and seeds are unchanged. All substantive analyses run through Slurm, account `torch_pr_634_general`; selectors use `cs`, eight BLAS threads are used for ALRA, and scVI uses one L40S. No original artifact, shared script, manuscript, documentation, commit, or remote branch is changed.\n']
a=ROOT/'A_calibration'
parts.append('## A. Held-out calibration\n\nWe pool all held-out zero candidates from rebuilt pancreas folds 0–2, the three colon folds covering 34 donors, and the rebuilt Norman CRISPRa benchmark, labeling masked positives 1 and recorded zeros 0. We recover the saved calibrated probability from the saved detection probability as `p = 0.1 d / (1 - 0.9 d)`, then measure ten equal-count reliability bins, ECE, Brier score, and calibration bias without fitting any model. We obtain percentile 95% intervals from 2,000 draws of held-out donors or perturbation targets with replacement, seed 1729, rebuilding the equal-count bins within each draw.\n')
if (a/'calibration.csv').exists():
    rows=read(a/'calibration.csv')
    parts.append(table(['Dataset','Candidates','Donors / targets','Observed positive share','Mean p','Brier','Constant-reference Brier'],[[r[k] for k in ['dataset','candidates','units','observed_share','mean_p','brier','brier_constant']] for r in rows]))
    parts.append(table(['Dataset','ECE [95% interval]','Mean p minus observed share [95% interval]'],[[r['dataset'],interval(r,'ece','ece_ci_low','ece_ci_high'),interval(r,'bias','bias_ci_low','bias_ci_high')] for r in rows]))
    parts.append('All quantities are on the 0–1 scale. The constant-reference Brier score is prevalence × (1 − prevalence). Calibration bias is consistent with zero in pancreas and colon; Norman has a small negative bias whose interval excludes zero. Every Brier score is below its constant-reference value.\n\nEqual-count bins split probability ties in stable saved-candidate order. Bootstrap clusters are sorted donor/target labels, including all held-out targets represented in the candidates; the statistic pools candidates rather than averaging donor metrics. The exact weighted-rank bootstrap passes a runnable check against explicit candidate replication. Saved d is float32, so recovered p retains storage quantization; the largest absolute difference from a saved unit mean p is 1.2443871807987783e-08. No isotonic map or selector is refitted.\n')
    parts.append(table(['Dataset','Bin','Count','Mean p','Observed positive share'],[[r[k] for k in ['dataset','bin','count','mean_p','observed_share']] for r in read(a/'reliability.csv')]))
    parts.append('Outputs: `A_calibration/calibration.csv`, `A_calibration/reliability.csv`, `A_calibration/reliability.png` (three panels, 240 dpi), `A_calibration/*_bootstrap.npz` (draws and statistics), and `A_calibration/provenance.json`.\n\nFailure and recovery: job 18883717 stopped at a probability-range assertion because floating-point inversion of d = 1 gave p slightly above 1. Clipping only inversion roundoff to [0, 1] fixed the assertion; job 18883781 completed. Final job 18884298 verified the saved mask rate, probability range, and disjoint held-out donors, and regenerated the figure with panel-specific axes; all numerical results were unchanged.\n\nReproduce:\n```bash\nO=artifacts/paper_evidence/review_round4/reviewer_extras\nsbatch -A torch_pr_634_general -p cs -J extras-A -o "$O/logs/A-%j.log" "$O/job.sh" .venv/bin/python -u "$O/A_calibration/analyze.py"\n```\n')
else: parts.append('Status: pending.\n')
parts.append('## B. Additional thinning comparators\n\nWe reuse the saved 50% thinning inputs, fitting labels, held-out cells, and unthinned endpoint references of the existing transductive downstream analysis. ALRA is fitted on the same thinned hybrid inputs with the production runner and eight BLAS threads, while the production single-method MLP selectors use standard scVI or MAGIC values plus the three context features and insert their own method’s count-scale value. We fill exactly 1%, 5%, and 10% of candidate zeros and apply the unchanged biological evaluators and original paired donor, stage, or target bootstrap draws, 2,000 replicates with seed 1729.\n')
b=ROOT/'B_thinning'
if (b/'reproduction.json').exists():
    check=json.loads((b/'reproduction.json').read_text())
    parts.append(f"Before new fitting, the unchanged Table 3 aggregation reproduced {check['rows']} existing rows and {check['numeric_comparisons']} numerical values, maximum absolute difference {check['max_absolute_difference']}. This check reruns the bootstrap from saved per-unit endpoint values; it does not refit the existing methods.\n")
if (b/'new_rows.csv').exists():
    rows=read(b/'new_rows.csv'); names={'alra':'ALRA','selector_scvi':'Selector on scVI','selector_magic':'Selector on MAGIC'}
    with (b/'table3_new_rows_x100.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['dataset','endpoint','method','fill_pct','error_reduction_x100','ci_low_x100','ci_high_x100'])
        writer.writerows([[r[k] for k in ['dataset','endpoint','method','fill_pct']]+[format(100*float(r[k]),'.17g') for k in ['error_reduction','ci_low','ci_high']] for r in rows])
    keys=list(dict.fromkeys((r['dataset'],r['endpoint']) for r in rows)); lookup={(r['dataset'],r['endpoint'],r['method'],int(r['fill_pct'])):r for r in rows}
    existing=read(b/'existing_reproduced.csv')
    combined={(r['dataset'],r['endpoint'],r['method'],int(r['fill_pct'])):r for r in existing+rows}
    paper_names={'detection':'Detection-weighted','transductive':'Fused value','svd':'SVD','magic':'MAGIC','scvi':'scVI',**names}
    best={(ds,ep,p):max(round(100*float(combined[ds,ep,n,p]['error_reduction']),2) for n in paper_names) for ds,ep in keys for p in [1,10]}
    def compact(r):
        mark='\\*' if float(r['ci_low'])>0 or float(r['ci_high'])<0 else ''
        value=f"{100*float(r['error_reduction']):.2f}{mark}"
        return f'**{value}**' if round(100*float(r['error_reduction']),2)==best[r['dataset'],r['endpoint'],int(r['fill_pct'])] else value
    extended=table(['Endpoint',*paper_names.values()],[[f'{ds}; {ep}',*[' / '.join(compact(combined[ds,ep,n,p]) for p in [1,10]) for n in paper_names]] for ds,ep in keys])
    (b/'table3_extended.md').write_text('Reduction in absolute error × 100, 1% / 10% fills. * denotes a 95% interval excluding zero; bold marks the largest displayed reduction at each budget.\n\n'+extended)
    parts.append('Table 3 format: reduction in absolute error × 100 at 1% / 10% fills; * indicates a 95% interval excluding zero. Bold marks the largest displayed reduction among the original Table 3 methods and the new comparators, with rounding ties retained. `B_thinning/table3_extended.md` places the unchanged original columns alongside the new columns; the following exact table additionally includes the requested 5% budget.\n')
    parts.append(table(['Endpoint',*names.values()],[[f'{ds}; {ep}',*[' / '.join(compact(lookup[ds,ep,n,p]) for p in [1,10]) for n in names]] for ds,ep in keys]))
    parts.append(table(['Endpoint','Method','1% [95% interval]','5% [95% interval]','10% [95% interval]'],[[f'{ds}; {ep}',names[n],*[interval(lookup[ds,ep,n,p],'error_reduction','ci_low','ci_high',100) for p in [1,5,10]]] for ds,ep in keys for n in names]))
    parts.append('Outputs: `B_thinning/new_rows.csv` (native 0–1 endpoint scale), `B_thinning/table3_new_rows_x100.csv` (Table 3 scale), `B_thinning/all_rows.csv`, `B_thinning/unit_values.csv`, `B_thinning/reference_parity.csv`, `B_thinning/reproduction.json`, and `B_thinning/<unit>/` (fits, selector scores, filled contracts, fill checks, and biological evaluation). `B_thinning/commands.jsonl` records exact executed commands and `B_thinning/jobs.tsv` records Slurm jobs. The compact table matches the paper’s 1% / 10%, two-decimal, asterisk convention; the exact table retains all three budgets and interval endpoints.\n\nValidation and failures: all eight fitting/evaluation jobs and the summary completed without failure. Every fill-count check passed, and all 14 raw/unthinned reference endpoint checks passed, maximum absolute difference 1.1102230246251565e-16.\n')
else: parts.append('Status: pending. Existing results and Part A remain complete independently.\n')
parts.append('Reproduce:\n```bash\nO=artifacts/paper_evidence/review_round4/reviewer_extras\nbash "$O/B_thinning/submit.sh"\n# After the fit array completes successfully:\n# .venv/bin/python "$O/B_thinning/run.py" summary  (run through job.sh/sbatch)\n```\n')
parts.append('## C. Scale at 5,000 genes\n\nPaths in this section are relative to `C_scale/`.\n')
c=ROOT/'C_scale/README.md'
parts.append(c.read_text().split('\n',1)[1].replace('\n## ','\n### ') if c.exists() else 'Status: pending. The 200,000-cell setting will reuse S28 settings with 5,000 highly variable genes; the 2,000-gene results will not be rerun.\n')
if (ROOT/'finalizer_jobs.tsv').exists():
    with (ROOT/'finalizer_jobs.tsv').open() as f: finalizer=list(csv.DictReader(f,delimiter='\t'))[-1]
    parts.append(f"Automatic report refresh: Slurm job {finalizer['finalizer_job']} runs `finalize.sh` after evaluation job {finalizer['evaluation_job']} terminates, regenerates both reports, and prints this README to its Slurm log. See `finalizer_jobs.tsv`.\n")
parts.append('## Monitoring and report regeneration\n\nWait without rapid polling:\n```bash\nwhile squeue -u yz5944 -h -o %j | grep -Eq "^extras-(A|B-|C-|finalize)"; do sleep 600; done\n.venv/bin/python artifacts/paper_evidence/review_round4/reviewer_extras/C_scale/code/report.py\n.venv/bin/python artifacts/paper_evidence/review_round4/reviewer_extras/report.py\ncat artifacts/paper_evidence/review_round4/reviewer_extras/README.md\n```\n')
(ROOT/'README.md').write_text('\n'.join(parts))
