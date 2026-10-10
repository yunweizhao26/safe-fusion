import json
import os
from pathlib import Path
import pandas as pd

OUT = Path(os.environ.get('SEX_ZERO_OUTPUT', 'artifacts/paper_evidence/review_round4/sex_zero_comparators'))
SOURCE = Path('artifacts/paper_evidence/review_round4/sex_zero_comparators')
METHODS = {'Safe Fusion, inductive': 'safe_fusion_inductive', 'Safe Fusion, transductive': 'safe_fusion',
           'Safe Fusion, transductive + donor': 'safe_fusion_donor',
           **{m: m for m in ('SVD', 'Weighted kNN', 'MAGIC', 'scVI', 'ALRA', 'SAVER', 'DCA', 'scImpute', 'EnImpute')}}
data = {t: {n: pd.read_csv(OUT / 'evaluation' / t / f'{n}.csv') for n in ('auroc', 'fill_rates')} for t in ('pancreas', 'colon')}

def estimate(row, value, scale=1):
    return f'{row[value]*scale:.6f} [{row.ci_low*scale:.6f}, {row.ci_high*scale:.6f}]'

lines = [
    '# Sex-linked recorded-zero comparators', '',
    f'Output: `{OUT}/`. Each table cell gives **AUROC; expressing-sex fill %; other-sex fill %**, in that order. Fill rates use the 1% global budget. Every estimate includes its 95% donor-bootstrap interval. Unrounded values and the 5% and 10% fill rates are in `evaluation/{{pancreas,colon}}/fill_rates.csv` and `auroc.csv`.', '',
    '| Method | Pancreas RPS4Y1 | Pancreas XIST, fold 0 | Colon RPS4Y1 | Colon XIST |',
    '|---|---|---|---|---|',
]
for label, method in METHODS.items():
    cells = []
    for tissue, gene in [('pancreas', 'RPS4Y1'), ('pancreas', 'XIST'), ('colon', 'RPS4Y1'), ('colon', 'XIST')]:
        a, f = data[tissue]['auroc'], data[tissue]['fill_rates']
        a = a[(a.method == method) & (a.gene == gene)]
        if a.empty:
            cells.append('Unavailable')
            continue
        assert len(a) == 1
        terms = [estimate(a.iloc[0], 'auroc')]
        for kind in ('recorded zero, expressing sex', 'absent, other sex'):
            row = f[(f.method == method) & (f.gene == gene) & (f.fraction == .01) & (f.kind == kind)]
            assert len(row) == 1
            terms.append(estimate(row.iloc[0], 'fill_rate', 100))
        cells.append('<br>'.join(terms))
    lines.append('| ' + ' | '.join([label, *cells]) + ' |')
lines += [
    '',
    'Design: unchanged `scripts/disease_control_sex_zeros.py` statistical kernel; held-out recorded zeros only. Expressing-sex zeros are positives and other-sex zeros are negatives. Sex uses the existing fold-0 recorded reference: donor XIST total greater than RPS4Y1 means female, otherwise male. Pancreas RPS4Y1 covers 24 donors; XIST is present only in fold 0 (9 donors). Both colon genes cover 34 donors across three folds. The original 24/34-donor sets, sorted donor order, 2,000 within-sex resamples and seed 1729 are retained, including donors without XIST observations.', '',
    'Ranking: AUROC uses the continuous fill-order score: selector score for Safe Fusion, nonnegative imputed count for comparators. Each fold spends its global budget over all held-out recorded zeros: round(fraction × number of zeros), at 1%, 5% and 10%. A selected entry counts as filled only when its inserted count is positive. Main comparators reuse `exact_topk`, the original deployment rule; DCA, scImpute and EnImpute retain their existing deployment runner\'s random tie breaking, seed 1729. Normalized outputs use `masked_f1_units.COUNT_SCALE` and the deployment-input library sizes.', '',
    'Fits: SVD and weighted kNN reuse the label-free transductive teachers under `../transductive_references/sex_zeros/deployment_rebuilt/<unit>/`. MAGIC and scVI reuse standard all-cell fits under `../transductive_downstream/deployment/<unit>/standard/`; scVI uses the existing 200-epoch L40S fit. All deployment inputs retain masks in fitting cells and recorded counts in held-out cells. ALRA uses the unchanged paper runner: training-cell SVD, thresholds and scaling, with test-cell projection and exactly eight BLAS threads. SAVER uses its unchanged all-cell runner and count rescaling. DCA (default nb-conddisp), scImpute and EnImpute use existing runners and defaults, seed 1729, eight CPUs, with the existing Leiden rule for cluster count. No method was tuned.', '',
    'Validation: both tissues reproduce the saved inductive Safe Fusion AUROC, all three fill fractions, intervals and zero counts to 1e-12 before comparator evaluation. All three Safe Fusion rows are copied from the existing results. Reused contracts are checked for cell/gene order; MAGIC/scVI input hashes and full deployment matrices are checked against the sex-zero inputs. All 36 existing MAGIC/scVI global fill masks reproduce exactly (`verification/reused_fill_masks.json`). Verification files: `verification/{pancreas,colon}/check.json`. Each tissue retains `bootstrap_draws.npy`, `zero_scores.parquet`, `zero_counts.csv`, `donor_sex.csv`, `global_budgets.csv`, `audit.json` (fit paths, input/output hashes and settings), and `missing.json`. New fits are under `fits/<method>/<unit>/`; source snapshots and hashes are under `source_snapshot/` and `source_sha256.json`.', '',
]
missing = {t: json.loads((OUT / 'evaluation' / t / 'missing.json').read_text()) for t in data}
failures = OUT / 'failures.txt'
lines += ['Failures and limits: ' + (failures.read_text().strip() if failures.exists() else ('Some optional fits are unavailable; see missing.json and Slurm logs.' if any(missing.values()) else 'No failed fits or evaluations.')) + ' Pancreas XIST cannot represent the full cohort because folds 1 and 2 exclude that gene. Sex labels are expression-derived biological proxies.']
if any(missing.values()):
    lines += ['Incomplete fit coverage: ' + '; '.join(f'{t}: {", ".join(m)}' for t, m in missing.items() if m) + '. An endpoint is reported only when every fold containing that gene has a completed fit; partial cohorts are never substituted.']
lines += ['', 'Execution: `jobs.txt`, `job_accounting.txt` and `logs/` record Slurm jobs and failures. All fitting and evaluation run on partition `cs`, account `torch_pr_634_general`. Existing paper files, documentation and outputs remain unchanged; no commit or push.', '',
          'Reproduce from the repository root into a fresh subdirectory (submits fits, verification, evaluation and report in dependency order):', '', '```bash',
          f'bash scripts/analyses/sex_zero_comparators/reproduce.sh', '```', '',
          'Regenerate this report from the saved results:', '', '```bash',
          f'.venv/bin/python scripts/analyses/sex_zero_comparators/report.py', '```', '']
(OUT / 'README.md').write_text('\n'.join(lines))
print(OUT / 'README.md')
