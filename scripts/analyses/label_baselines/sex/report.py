import csv
import json
from pathlib import Path

OUT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/label_baselines/sex')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def value(row):
    return f"{float(row['auroc']):.3f} [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |'] + ['| ' + ' | '.join(row) + ' |' for row in rows])


names = {'safe_fusion_donor': 'Safe Fusion with donor labels', 'graph_smooth_donor': 'Label-aware weighted kNN', 'scvi_donor': 'Label-aware scVI', 'safe_fusion_donor minus graph_smooth_donor': 'SF minus label-aware kNN', 'safe_fusion_donor minus scvi_donor': 'SF minus label-aware scVI', 'safe_fusion': 'Safe Fusion, transductive', 'safe_fusion_inductive': 'Safe Fusion, inductive'}
names.update({'transductive_expected_count': 'Per-label expected count, transductive fitting', 'safe_fusion_donor minus transductive_expected_count': 'SF minus transductive expected count'})
parts = ['## Tissue sex-zero analyses', (OUT / 'design.md').read_text().strip()]
gates = [json.loads((OUT / tissue / 'gate.json').read_text()) for tissue in ['pancreas', 'colon']]
assert all(g['passed'] for g in gates)
parts.append(f"**S18 gate passed:** {sum(g['rows'] for g in gates)} rows, all three numbers per row reproduced within 1e-12 and all manuscript-rounded values exact. Frozen comparator score replay covers all 12 currently present methods, including EnImpute. Evidence: `sex/{{pancreas,colon}}/gate.json` and `s18_reproduced.csv`.")
results = {(tissue, row['gene'], row['method'], row['adjustment']): row for tissue in ['pancreas', 'colon'] for row in read(OUT / tissue / 'label_rankings_auroc.csv')}
results.update({(tissue, row['gene'], row['method'], row['adjustment']): row for tissue in ['pancreas', 'colon'] for row in read(OUT / tissue / 'transductive_expected_count_auroc.csv')})
for adjustment in ['none', 'library_quintiles']:
    parts.append('### Donor-aware continuous ranking AUROC: ' + ('unadjusted' if adjustment == 'none' else 'within library-size quintiles'))
    methods = ['safe_fusion_donor', 'graph_smooth_donor', 'scvi_donor', 'safe_fusion_donor minus graph_smooth_donor', 'safe_fusion_donor minus scvi_donor']
    methods += ['transductive_expected_count', 'safe_fusion_donor minus transductive_expected_count']
    rows = [[names[m]] + [value(results[t, g, m, adjustment]) for t in ['pancreas', 'colon'] for g in ['RPS4Y1', 'XIST']] for m in methods]
    rows.append(['Fitting-only per-label expected count'] + ['Unavailable: unseen donor'] * 4)
    parts.append(table(['Ranking / paired difference', 'Pancreas RPS4Y1', 'Pancreas XIST', 'Colon RPS4Y1', 'Colon XIST'], rows))
parts.append('### Colon sensitivity: AUROC with 95% donor-bootstrap intervals')
sensitivity = {(scenario, row['gene'], row['method']): row for scenario in ['agreed25', 'metadata34'] for row in read(OUT / 'colon' / f'{scenario}_auroc.csv')}
methods = ['safe_fusion_donor', 'safe_fusion', 'safe_fusion_inductive', 'SVD', 'Weighted kNN', 'MAGIC', 'scVI', 'ALRA', 'SAVER', 'DCA', 'scImpute', 'EnImpute']
parts.append(table(['Method', 'Agreeing 25: RPS4Y1', 'Agreeing 25: XIST', 'Metadata 34: RPS4Y1', 'Metadata 34: XIST'], [[names.get(m, m)] + [value(sensitivity[s, g, m]) for s in ['agreed25', 'metadata34'] for g in ['RPS4Y1', 'XIST']] for m in methods]))
excluded = read(OUT / 'colon' / 'excluded_donors.csv')
parts.append('Excluded donors: ' + ', '.join('`' + row['donor'] + '`' for row in excluded) + '.')
parts.append(table(['Excluded donor', 'Expression sex', 'Metadata sex'], [[r['donor'], r['sex'], r['metadata_sex']] for r in excluded]))
parts.append('### Tissue paths, reproduction and limitations')
parts.append('All paths below are relative to `artifacts/paper_evidence/review_round4/label_baselines/`. `sex/{pancreas,colon}/label_rankings_auroc.csv` contains estimates, intervals, paired differences, sample counts and finite bootstrap counts. `sex/colon/{agreed25,metadata34}_auroc.csv` contains all 24 sensitivity rows per scenario. `sex/colon/metadata_sex_audit.csv` records all 34 donors; `excluded_donors.csv` records the nine exclusions. Frozen input scores, draw arrays and teacher provenance are retained in each tissue directory.')
parts.append('```bash\nverify=$(sbatch --parsable --array=0-1 --job-name=lb-sex-verify --export=ALL,STAGE=verify artifacts/paper_evidence/review_round4/label_baselines/sex/run.sh)\nwhile squeue -h -j "$verify" | grep -q .; do sleep 600; done\n# Inspect gates and use root verify_gate.py to verify saved numerics and disclose manuscript mismatches.\nanalyze=$(sbatch --parsable --array=0-1 --job-name=lb-sex-analysis --export=ALL,STAGE=analyze artifacts/paper_evidence/review_round4/label_baselines/sex/run.sh)\nwhile squeue -h -j "$analyze" | grep -q .; do sleep 600; done\nexpected=$(sbatch --parsable --array=0-1 --job-name=lb-sex-expected --export=ALL,STAGE=transductive artifacts/paper_evidence/review_round4/label_baselines/sex/run.sh)\nwhile squeue -h -j "$expected" | grep -q .; do sleep 600; done\nsbatch -A torch_pr_634_general -p cs -c 1 --mem=2G --time=00:05:00 --job-name=lb-sex-report --output=artifacts/paper_evidence/review_round4/label_baselines/sex/report-%j.log --wrap=".venv/bin/python artifacts/paper_evidence/review_round4/label_baselines/sex/report.py"\n```')
parts.append('No tissue fit or evaluator failure occurred. The expected-count baseline using only selector-fitting donors remains undefined for held-out labels. The separately named transductive version uses the actual donor-teacher fitting population and is retained in `sex/{pancreas,colon}/transductive_expected_count_{auroc.csv,scores.parquet,audit.json}`. Expression-rule and metadata sensitivity analyses do not establish biological ground truth for disputed donors. Intervals remain conditional on frozen models, masks, fold panels and scoring procedure.')
(OUT / 'summary.md').write_text('\n\n'.join(parts) + '\n')
