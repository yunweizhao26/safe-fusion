import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_s20_rows'


def interval(row, value='estimate'):
    return f'{row[value]:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]'


def pick(table, name, metric, paired=False):
    key = 'comparator' if paired else 'ranking'
    control = metric == 'control'
    statistic = {'pooled': 'pooled_spearman', 'within': 'within_target_spearman',
                 'control': 'pooled_spearman', 'partial': 'partial_spearman'}[metric]
    rows = table.loc[(table[key] == name) & (table.scope == ('control' if control else 'all')) &
                     (table.resampling == ('cells_within_replicate' if control else 'target_clusters')) &
                     (table.statistic == statistic)]
    if metric == 'partial':
        rows = rows.loc[rows.covariates == 'target_ifng_library']
    assert len(rows) == 1, (name, metric, len(rows))
    row = rows.iloc[0]
    assert row.draws == 2000
    assert np.isfinite([row.ci_low, row.ci_high]).all()
    if not paired:
        assert row.defined_draws == 2000
    return row


def main():
    tables = {}
    audits = {}
    for block in ['small', 'full']:
        audits[block] = json.loads((OUT / block / 'audit.json').read_text())
        assert audits[block]['passed']
        tables[block] = (pd.read_csv(OUT / block / 'association.csv'),
                         pd.read_csv(OUT / block / 'paired_differences.csv'))
    lines = [
        'Supplementary Table S20: added Papalexi rankings.', '',
        'Added scVI probability-of-nonzero rankings to both blocks and MAGIC to the 655-cell block.', '',
        'The statistical evaluator `scripts/evaluate_protein_within_state.py` is unchanged. '
        'The additive wrapper reads the frozen cell frames, appends saved scores, and calls its original `run_job` function. '
        'The cells, responses, existing rankings, covariates, bootstrap units, 2,000 draws and seed 1729 are preserved.', '',
        'Entries are estimates [95% percentile intervals], rounded directly to three decimals. '
        'Pooled, within-target and partial intervals resample perturbation targets, including the non-targeting group as one cluster. '
        'Control intervals retain the original cell bootstrap within replicate. '
        'Within-target Spearman is cell-count weighted over groups with at least three cells. '
        'Partial Spearman uses `target_ifng_library`: target indicators and ranked IFN-gamma score and library size. '
        'All responses are CLR PD-L1 among recorded CD274 RNA zeros.', '',
        '| Benchmark cells / evaluated zeros | New ranking | Pooled rho | Within target | Control | Partial | SF minus ranking, pooled |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    requested = [('small', 'p_nonzero', 'scVI P(X > 0)'), ('small', 'magic', 'MAGIC'),
                 ('full', 'p_nonzero', 'scVI P(X > 0)')]
    for block, name, label in requested:
        a, p = tables[block]
        size = '3,400 / 655' if block == 'small' else '20,156 / 4,083'
        values = [interval(pick(a, name, metric)) for metric in ['pooled', 'within', 'control', 'partial']]
        values += [interval(pick(p, name, 'pooled', True), 'safe_fusion_minus_comparator')]
        lines.append('| ' + ' | '.join([size, label, *values]) + ' |')
    lines += ['', 'Paired differences use the same bootstrap draw for Safe Fusion and the comparator. Positive values favor Safe Fusion.', '',
              '| Evaluated zeros | Comparator | SF minus ranking, pooled | SF minus ranking, within target | SF minus ranking, partial |',
              '| --- | --- | --- | --- | --- |']
    for block, name, label in requested:
        values = [interval(pick(tables[block][1], name, metric, True), 'safe_fusion_minus_comparator')
                  for metric in ['pooled', 'within', 'partial']]
        lines.append('| ' + ' | '.join(['655' if block == 'small' else '4,083', label, *values]) + ' |')
    lines += ['', 'Reproduction check: recomputed Safe Fusion against each specified current `association.csv`. '
              'Every checked estimate and interval endpoint agrees at both three and four printed decimals and within absolute tolerance 1e-12. '
              'The check also covers every existing row for PD-L1/CLR/recorded zeros in the all-cell and control scopes, '
              'plus their paired differences, counts and defined-draw counts.', '',
              '| Evaluated zeros | Safe Fusion pooled | Within target | Control | Partial |',
              '| --- | --- | --- | --- | --- |']
    for block in ['small', 'full']:
        values = [interval(pick(tables[block][0], 'safe_fusion', metric))
                  for metric in ['pooled', 'within', 'control', 'partial']]
        lines.append('| ' + ' | '.join(['655' if block == 'small' else '4,083', *values]) + ' |')
    lines += ['']
    for block in ['small', 'full']:
        audit = audits[block]
        lines.append(f"- {audit['n_cells']:,} zeros: {audit['n_control']} controls, {audit['n_groups']} target groups; "
                     f"{audit['safe_fusion_rows_verified']} Safe Fusion rows, {audit['association_rows_verified']} total association rows "
                     f"and {audit['paired_rows_verified']} paired rows verified. Maximum absolute differences: "
                     f"{audit['max_association_difference']:.3g} for associations and {audit['max_paired_difference']:.3g} for paired differences.")
    lines += ['',
        'Every displayed new association has 2,000 defined draws. '
        'The original seed tuples are `[1729, 1, 0, 0, 0]` for target clusters, '
        '`[1729, 1, 0, 0, 1]` for the supplementary all-cell replicate bootstrap, '
        'and `[1729, 1, 0, 0, 2]` for controls.', '',
        'Paths below are relative to `artifacts/paper_evidence/review_round4/` unless stated otherwise.', '',
        '- Small reference: `transductive_references/protein_cs/evaluation_within_state/{association,paired_differences}.csv`. '
        'Its frozen frame is `protein_full/verification/evaluation/pdl1_cells.parquet`.',
        '- Full reference and frame: `protein_followup/full/evaluation/{association.csv,paired_differences.csv,pdl1_cells.parquet}`.',
        '- Added scores: `protein_depth/inputs/papalexi_{small,full}.parquet`, columns `p_nonzero` and `magic`. '
        'Exact cell identifiers, responses, Safe Fusion scores and covariates match the frozen frames.',
        '- Source derivation: `scripts/analyses/protein_depth/prepare.py`. The saved `p_nonzero` is '
        '`-log P(X = 0) = theta * log1p(mu / theta)`, a strictly increasing representation of `P(X > 0)`. '
        'It uses saved scVI normalized means, observed panel library size and saved gene dispersion. '
        'Scores are reused exactly, without converting to rounded probabilities.',
        '- Underlying small sources: `transductive_references/protein_cs/scvi/{mean.npy,model/model.pt}` '
        'and `transductive_references/protein_cs/magic_standard/mean.npy`. '
        'Full scVI sources: `protein_full/full/scvi/{mean.npy,model/model.pt}`.',
        '- New outputs: `protein_s20_rows/{small,full}/` contains `association.csv`, `paired_differences.csv`, '
        '`pdl1_cells.parquet`, `audit.json`, `reproduction_association.csv` and `reproduction_paired.csv`. '
        'CSVs retain full precision. Audit files include source hashes and numerical differences. '
        'Slurm receipts and logs are in `protein_s20_rows/runtime/` and `protein_s20_rows/logs/`.', '',
        'Reproduce from the repository root with the existing saved inputs. No fitting is required:', '',
        '```bash',
        'OUT=artifacts/paper_evidence/review_round4/protein_s20_rows',
        'sbatch -A torch_pr_634_general -p cs "scripts/analyses/protein_s20_rows/run.sh"',
        'while squeue -u "$USER" -h -n s20-added-rankings | grep -q .; do sleep 300; done',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "scripts/analyses/protein_s20_rows/report.py"',
        '```', '',
        'The array runs these additive evaluator commands, with BLAS threads limited to one:', '',
        '```bash',
        '.venv/bin/python "scripts/analyses/protein_s20_rows/evaluate.py" --block small --include-p-nonzero --include-magic --workers 3',
        '.venv/bin/python "scripts/analyses/protein_s20_rows/evaluate.py" --block full --include-p-nonzero --workers 3',
        '```', '',
        'Only PD-L1/CLR recorded-zero scopes are evaluated. Scores are reused without model fitting.',
    ]
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
