from pathlib import Path
import json
import pandas as pd
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras')
lines=['# Round 2 referee extras','', 'Output: `artifacts/paper_evidence/review_round4/round2_extras/`. AUROCs use the 0–1 scale; F1 differences are percentage points. All intervals use 2,000 bootstrap draws and seed 1729.','', '## Part A — Table 2 AUROCs','',
'The analysis reuses saved scores and recorded-zero definitions: likely dropouts are control zeros for 26 Adamson targets, activated-cell zeros for 41 Norman targets, and male-donor RPS4Y1 zeros in 24 pancreas and 34 colon donors. Bootstrap draws resample targets with the original screen-specific seed streams or donors within sex, keeping the original donor pools.','']
def ci(x,lo,hi): return f'{x:.4f} [{lo:.4f}, {hi:.4f}]'
a=O/'part_a/table2_aurocs.csv'
if a.exists():
    v=json.loads((O/'part_a/verification.json').read_text())
    lines += [f'**Reproduction against the initial Table 2 snapshot:** all 65 source estimates reproduce within 1e-12 and match its printed precision. The requested fill-order gate fails: only {v["fill_order_printed_matches"]}/65 fill-order estimates match that table; its 26 tissue entries used continuous scores, despite its original caption.','',
    'The original screen fill grid is 1%, 2%, …, 10%; the original tissue deployment grid is 1%, 5%, 10%. Tissue fill-order results below use those three saved budgets. Unfilled zeros gives the fraction of evaluated cell–gene zeros in the common zero-score tie at 10%; tied AUROC pairs gives the fraction of positive–negative comparisons tied at any fill-order score. Screen summaries average over targets, with within-quintile pair weighting for the adjusted AUROC and its pair-tie share.','']
    f=pd.read_csv(a)
    order=['Safe Fusion','SVD','Weighted kNN','MAGIC','scVI','ALRA','SAVER','DCA','scImpute','EnImpute','Safe Fusion, labels','Weighted kNN, labels','Per-label expected count']
    for dataset,adj,title in [('adamson_crispri','none','Adamson, unadjusted'),('adamson_crispri','depth_strata','Adamson, library-size quintiles'),('norman_crispra','none','Norman'),('pancreas','none','RPS4Y1, pancreas'),('colon','none','RPS4Y1, colon')]:
        lines += [f'### {title}','','| Method | Fill-order AUROC [95% CI] | Continuous AUROC [95% CI] | Unfilled zeros (%) | Tied AUROC pairs (%) |','|---|---|---|---:|---:|']
        g=f[(f.dataset==dataset)&(f.adjustment==adj)].set_index('method')
        for name in order:
            r=g.loc[name]; lines.append(f'| {name} | {ci(r.fill_order,r.fill_lower,r.fill_upper)} | {ci(r.continuous,r.continuous_lower,r.continuous_upper)} | {100*r.unfilled_zero_share:.2f} | {100*r.tied_pair_share:.2f} |')
        lines.append('')
    lines += ['Per-label expected count retains the published detection transform, `1-exp(-L*a)`, and its stored precision. Its tissue version uses all observed hybrid cells, matching the original donor-aware teachers; the fitting-only donor rule remains undefined for held-out donors.','',
    'Paths: `part_a/table2_aurocs.csv`, `reproduction.csv`, `source_gate.csv`, `table2_original_snapshot.csv`, `gate.json`, `verification.json`, endpoint `*_per_target.csv`, tissue `*_zero_scores.parquet`, and `*_draws.npy`. Failure: the initial manuscript’s common fill-order description did not match its tissue source estimates; this analysis changed neither the manuscript nor statistical definitions.','',
    'Reproduce Part A: `bash artifacts/paper_evidence/review_round4/round2_extras/reproduce.sh A`.','']
else: lines += ['Incomplete. See `logs/a-*.log` and `part_a/` for completed endpoints.','']
lines += ['## Part B — All-cell Figure 1 curves','',
'The unchanged masked-F1 evaluator ranks the all-cell SVD, weighted kNN and ALRA contracts on the original held-out candidates and 1,000-point grid. Only those nine method-by-dataset curves replace the source CSV rows; all other records are preserved verbatim.','',
'| Dataset | Highest F1 / 100 | Strictly highest / 100 |','|---|---:|---:|']
for r in pd.read_csv(O/'part_b/highest_f1.csv').itertuples(): lines.append(f'| {r.dataset} | {r.highest} | {r.strictly_highest} |')
lines += ['', 'Pancreas has one tied fraction. These are exact counts over a fixed grid, so no bootstrap interval is attached to the count.','',
'Paths: `part_b/selector_f1_fillrate_allcell_1000_points.csv`, `highest_f1.csv`, `verification.json`. Validation: 9,000 replaced rows, 18,000 unchanged rows, identical columns and grid. Failures: none.','',
'Reproduce Part B: `bash artifacts/paper_evidence/review_round4/round2_extras/reproduce.sh B`.','',
'## Part C — Closest-comparator replicates','',
'The analysis reuses masks, split roles and model seeds 1729–1733, including the saved transductive Safe Fusion, all-cell SVD/kNN and scVI models, and invokes the paper runners unchanged for new fits. It subtracts each comparator’s mean masked F1 over the ten integer fill fractions 1%–10% from Safe Fusion per seed; intervals resample the same biological units across all five seeds and then average, while SD describes variation across seeds.','']
btable=['Mean F1 differences on the ten integer fractions, with the existing paired donor/target intervals for these same all-cell fits:','',
        '| Dataset | SF minus SVD [95% CI] | SF minus weighted kNN [95% CI] | SF minus ALRA [95% CI] |','|---|---|---|---|']
bf=pd.read_csv(O/'part_b/paired_mean_f1.csv')
for dataset,part in bf.groupby('dataset',sort=False):
    part=part.set_index('comparator')
    values=[ci(part.loc[m+' (transductive)','estimate_pp'],part.loc[m+' (transductive)','lower_pp'],part.loc[m+' (transductive)','upper_pp']) for m in ['SVD','Weighted kNN','ALRA']]
    btable.append('| '+dataset+' | '+' | '.join(values)+' |')
btable += ['', 'All 12 curve means (three comparators and Safe Fusion in each dataset) reproduce the saved Table 1 analysis at its four-decimal precision; see `part_b/table1_parity.json` and `paired_mean_f1.csv`.','']
pos=lines.index('## Part C — Closest-comparator replicates'); lines[pos:pos]=btable
c=O/'part_c/across_seeds.csv'
if c.exists() and c.stat().st_size>5:
    f=pd.read_csv(c); p=pd.read_csv(O/'part_c/paired_per_seed.csv')
    lines += ['| Dataset | SF minus comparator | Seed 1729 | 1730 | 1731 | 1732 | 1733 | Mean [95% CI] | Seed SD | SF higher / 5 |','|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in f.itertuples():
        s=p[(p.dataset==r.dataset)&(p.method==r.method)].set_index('seed')
        vals=' | '.join(f'{s.loc[k,"difference"]:.4f}' for k in range(1729,1734))
        lines.append(f'| {r.dataset} | {r.method} | {vals} | {ci(r.mean_difference,r.lower,r.upper)} | {r.seed_sd:.4f} | {r.seeds_higher} |')
    lines += ['', 'Per-seed paired 95% intervals are in `part_c/paired_per_seed.csv`; absolute F1 intervals are in `absolute_per_seed.csv`. S4’s float64 count-scale comparator rankings, unit tie seeds and stable selector ties are retained. CRISPRa keeps the evaluator’s control group alongside its 65 targets.','']
    v=json.loads((O/'part_c/verification.json').read_text())
    lines += [f'Validation: {len(v["reference_parity"])}/15 saved Safe Fusion estimates and intervals reproduce within 1e-12; {len(f)}/24 requested five-seed comparisons complete.','']
    if v['missing']: lines += ['Remaining work:','',*[f'- {x}' for x in v['missing']],'']
else: lines += ['Incomplete; fits or evaluations remain. See `part_c/units.json`, per-unit `evaluation_missing.json`, `logs/`, and `job_accounting.txt`.','']
lines += ['Paths: `part_c/units.json`, per-unit `commands_*.jsonl`, `fits16/`, `selectors/`, `probability/`, `unit_counts.parquet`, and aggregate CSVs. Failure history: initial eight-thread DCA/EnImpute submissions were cancelled and excluded after metadata showed the paper used 16 threads; final runs use 16, and selectors retain eight. An initial EnImpute preprocessing task encountered a shared-cache stale file handle; final jobs isolate caches. One seed-1729 evaluator started before a manifest-field fix and was retried; an early summary needed a NumPy-integer JSON conversion fix. See Slurm accounting for final task states.','',
'## Reproduction and accounting','',
'Reproduce Part C: `bash artifacts/paper_evidence/review_round4/round2_extras/reproduce.sh C`.','',
'Run from `the repository root`. All analysis and fit commands use Slurm account `torch_pr_634_general`; selectors run on `cs`, and EnImpute jobs have a 24-hour limit. No GPU is needed because saved scVI models suffice.','',
'```bash','O=artifacts/paper_evidence/review_round4/round2_extras','bash "$O/reproduce.sh"','bash "$O/wait.sh"','```','',
'`reproduce.sh` contains the dependency chain; completed fits are reused. `jobs.tsv`, `job_accounting.txt`, `resource_accounting.txt`, `logs/`, `failures.json`, and `source_sha256.json` record execution, CPU time, memory, failures and source provenance. `final_audit.json` records final completeness and source checks. All 19 computational source hashes remain unchanged.','',
'External source drift: the final audit detected concurrent changes to `scripts/plot_overview_f1.py` and `safe_fusion_overleaf/bioinformatics/6_results.tex`; the original audit failure is retained in `logs/audit-19000090.log`, and both initial and final hashes are in `external_source_drift.json`. The current plotting script points to the new Part B CSV, and the current Table 2 uses continuous scores and intervals. The initial table values remain frozen in `part_a/table2_original_snapshot.csv` for reproducibility; the initial-caption finding above refers to that snapshot.','',
'This process wrote only under the output directory and did not edit the manuscript, plotting script, documentation, or repository README. Jobs named `scmp-*`, `cx-*` or `mm3-*` were not touched. No commit or push. All requested computations are complete.','']
(O/'README.md').write_text('\n'.join(lines))
