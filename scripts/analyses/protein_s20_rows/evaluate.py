import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_s20_rows'
ROUND = OUT.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_protein_within_state as e


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(saved, actual, key, columns):
    keys = ['protein', 'gene', 'scale', 'cells', 'scope', 'resampling',
            'statistic', 'covariates', key]
    saved = saved.fillna({'covariates': ''})
    saved = saved.loc[(saved.protein == 'PD-L1') & (saved.scale == 'clr') &
                      (saved.cells == 'recorded_rna_zero') & saved.scope.isin(['all', 'control'])]
    joined = saved.merge(actual.fillna({'covariates': ''}), on=keys,
                         suffixes=('_saved', '_new'), validate='one_to_one')
    assert len(joined) == len(saved), (len(joined), len(saved))
    checks = []
    for _, row in joined.iterrows():
        for column in columns:
            old, new = row[column + '_saved'], row[column + '_new']
            np.testing.assert_allclose(new, old, rtol=0, atol=1e-12, equal_nan=True)
            assert f'{old:.3f}' == f'{new:.3f}'
            assert f'{old:.4f}' == f'{new:.4f}'
            checks.append({**{k: row[k] for k in keys}, 'field': column,
                           'saved': old, 'recomputed': new,
                           'absolute_difference': abs(new - old),
                           'match_3dp': True, 'match_4dp': True})
    for column in ['n_cells', 'n_groups', 'draws'] + (['defined_draws'] if key == 'ranking' else []):
        np.testing.assert_array_equal(joined[column + '_saved'], joined[column + '_new'])
    return pd.DataFrame(checks)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--block', choices=['small', 'full'], required=True)
    parser.add_argument('--include-p-nonzero', action='store_true')
    parser.add_argument('--include-magic', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    small = args.block == 'small'
    frame_path = ROUND / ('protein_full/verification/evaluation/pdl1_cells.parquet' if small
                          else 'protein_followup/full/evaluation/pdl1_cells.parquet')
    source = ROUND / f'protein_depth/inputs/papalexi_{args.block}.parquet'
    reference = ROUND / ('transductive_references/protein_cs/evaluation_within_state' if small
                         else 'protein_followup/full/evaluation')
    paths = [frame_path, source, reference / 'association.csv', reference / 'paired_differences.csv',
             ROOT / 'scripts/evaluate_protein_within_state.py',
             ROOT / 'scripts/evaluate_papalexi_crossmodal.py',
             ROOT / 'scripts/evaluate_pdl1_state_baselines.py', Path(__file__)]
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    frame = pd.read_parquet(frame_path)
    extra = pd.read_parquet(source).rename(columns={'protein': 'adt_clr'})
    assert frame.cell_id.is_unique and extra.cell_id.is_unique
    assert set(frame.cell_id) == set(extra.cell_id)
    extra = extra.set_index('cell_id').loc[frame.cell_id].reset_index()

    for column in ['cell_index', 'target', 'replicate', 'adt_clr', 'safe_fusion',
                   'library_size', 'ifng_score', 'cell_zero_score', 'target_baseline_clr']:
        np.testing.assert_array_equal(frame[column], extra[column])
    assert len(frame) == (655 if small else 4083)
    assert (frame.target == e.CONTROL).sum() == (191 if small else 439)
    assert frame.target.nunique() == (25 if small else 26)
    frame['target_baseline'] = frame.target_baseline_clr
    names = list(e.RANKINGS)
    if not small:
        names += ['svd_all_cells', 'weighted_knn_all_cells', 'magic']
    additions = []
    if args.include_p_nonzero:
        additions.append('p_nonzero')
    if args.include_magic and 'magic' not in names:
        additions.append('magic')
    for name in additions:
        frame[name] = extra[name].to_numpy()
    names += additions
    assert np.isfinite(frame[names].to_numpy(dtype=float)).all()

    jobs = []
    for number, (scope, subset, resampling) in enumerate([
        ('all', frame, 'target_clusters'), ('all', frame, 'cells_within_replicate'),
        ('control', frame.loc[frame.target == e.CONTROL], 'cells_within_replicate')]):
        group_names, groups = np.unique(subset.target.astype(str), return_inverse=True)
        rankings = tuple(n for n in names if not (scope == 'control' and n == 'target_baseline'))
        jobs.append(e.Job(
            protein='PDL1', gene='CD274', scale='clr', cells='recorded_rna_zero',
            scope=scope, resampling=resampling, rankings=rankings,
            scores=subset[list(rankings)].to_numpy(dtype=float),
            response=subset.adt_clr.to_numpy(dtype=float), groups=groups,
            replicate=np.unique(subset.replicate.astype(str), return_inverse=True)[1],
            covariates={n: subset[n].to_numpy(dtype=float) for n in e.CONTINUOUS_COVARIATES},
            within=resampling == 'target_clusters',
            covariate_sets=e.CONTROL_COVARIATE_SETS if scope == 'control' else
                e.COVARIATE_SETS if resampling == 'target_clusters' else {},
            seed=(1729, list(e.PROTEIN_TO_GENE).index('PDL1'), 0, 0, number),
            draws=2000, group_names=list(group_names)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(e.run_job, jobs))
    association = pd.concat([r[0] for r in results], ignore_index=True)
    paired = pd.concat([r[1] for r in results], ignore_index=True)
    dest = OUT / args.block
    dest.mkdir(exist_ok=True)
    association.to_csv(dest / 'association.csv', index=False)
    paired.to_csv(dest / 'paired_differences.csv', index=False)
    frame.to_parquet(dest / 'pdl1_cells.parquet', index=False)
    check = compare(pd.read_csv(reference / 'association.csv'), association, 'ranking',
                    ['estimate', 'ci_low', 'ci_high'])
    pair_check = compare(pd.read_csv(reference / 'paired_differences.csv'), paired, 'comparator',
                         ['safe_fusion_minus_comparator', 'ci_low', 'ci_high'])
    check.to_csv(dest / 'reproduction_association.csv', index=False)
    pair_check.to_csv(dest / 'reproduction_paired.csv', index=False)
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report = dict(block=args.block, passed=True, additions=additions, n_cells=len(frame),
                  n_control=int((frame.target == e.CONTROL).sum()), n_groups=int(frame.target.nunique()),
                  bootstrap_draws=2000, seed=1729, job_seeds=[list(j.seed) for j in jobs],
                  association_rows_verified=len(check) // 3, paired_rows_verified=len(pair_check) // 3,
                  safe_fusion_rows_verified=int((check.ranking == 'safe_fusion').sum()) // 3,
                  max_association_difference=float(check.absolute_difference.max()),
                  max_paired_difference=float(pair_check.absolute_difference.max()),
                  matches_printed_3dp_and_4dp=True, source_sha256=before,
                  job_id=os.environ.get('SLURM_JOB_ID'))
    (dest / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
