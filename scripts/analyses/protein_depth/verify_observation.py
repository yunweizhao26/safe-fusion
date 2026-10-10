import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_depth'
BASE = ROOT / 'artifacts/paper_evidence/review_round4'
SOURCE = BASE / 'protein_multi/corrected'


def main():
    original = pd.read_csv(SOURCE / 'evaluation/per_pair.csv')
    followup = pd.read_csv(BASE / 'protein_depth/tables/association.csv')
    labels = {'safe_fusion': 'Safe Fusion', 'svd': 'SVD', 'weighted_knn': 'Weighted kNN',
              'magic': 'MAGIC', 'scvi': 'scVI', 'p_nonzero': 'scVI P(X>0)', 'library_size': 'Library size'}
    followup = followup.loc[followup.case.str.startswith('hao_') &
                            (followup.metric == 'pooled_spearman') & followup.method.isin(labels)].copy()
    followup['method'] = followup.method.map(labels)
    followup = followup.rename(columns={'estimate': 'spearman'})
    selected = pd.read_csv(SOURCE / 'selection/selected_pairs.csv')
    keys = ['pair_index', 'gene', 'protein', 'method']
    replay = original.merge(followup, on=keys, validate='one_to_one', suffixes=('_old', '_new'))
    assert len(replay) == len(original) == 41 * 7
    np.testing.assert_allclose(replay.spearman_old, replay.spearman_new, atol=1e-14, equal_nan=True)
    table = followup.pivot(index=['pair_index', 'gene', 'protein'], columns='method', values='spearman').reset_index()
    table = table.merge(selected[['gene', 'protein', 'n_train_detected']], validate='one_to_one')
    assert len(table) == 41 and table['pair_index'].nunique() == 41
    better = table[['MAGIC', 'scVI']].max(axis=1)
    delta = table['Safe Fusion'] - better
    depth = table['Library size']
    draws = np.random.default_rng(1729).integers(0, len(table), size=(2000, len(table)))
    pearson_boot = pearsonr(delta.to_numpy()[draws], depth.to_numpy()[draws], axis=1).statistic
    np.testing.assert_allclose(pearson_boot[0], pearsonr(delta.iloc[draws[0]], depth.iloc[draws[0]]).statistic)
    result = {
        'source_replay_passed': True,
        'n_pairs': len(table),
        'library_definition': 'log1p library size over the selected 2012-gene truth panel; not the full-transcriptome nCount_RNA metadata field',
        'sf_minus_better_vs_library_protein': {
            'pearson_r': float(pearsonr(delta, depth).statistic),
            'spearman_rho': float(spearmanr(delta, depth).statistic),
            'pearson_pair_bootstrap_ci_low': float(np.quantile(pearson_boot, .025)),
            'pearson_pair_bootstrap_ci_high': float(np.quantile(pearson_boot, .975)),
            'pearson_pair_bootstrap_valid_draws': int(np.isfinite(pearson_boot).sum()),
        },
        'pair_bootstrap_design': {
            'draws': 2000, 'seed': 1729, 'interval': 'percentile 95%',
            'unit': 'antibody-gene pair, within each original fixed subgroup',
            'rng': 'numpy default_rng restarted at 1729 for each subgroup; same sampled pairs for both comparators',
            'interpretation': 'Descriptive pair resampling, conditional on saved per-pair estimates; antibody pairs share genes and cell populations. Does not propagate cell-level uncertainty or estimate generalization to new donors.',
        },
        'strata': {},
        'sources_sha256': {},
    }
    for name, mask in {
        'all': np.ones(len(table), dtype=bool),
        'library_protein_above_0.1': depth > .1,
        'library_protein_below_minus_0.1': depth < -.1,
        'library_protein_between': depth.between(-.1, .1),
        'train_detected_below_200': table.n_train_detected < 200,
        'train_detected_at_least_200': table.n_train_detected >= 200,
    }.items():
        sub = table.loc[mask]
        stats = {'n_pairs': len(sub), 'sf_minus_comparator': {}}
        subgroup_draws = np.random.default_rng(1729).integers(0, len(sub), size=(2000, len(sub)))
        for method in ['MAGIC', 'scVI']:
            difference = sub['Safe Fusion'] - sub[method]
            mean_boot = difference.to_numpy()[subgroup_draws].mean(axis=1)
            stats['sf_minus_comparator'][method] = {
                'mean': float(difference.mean()), 'median': float(difference.median()),
                'wins': int((difference > 0).sum()),
                'mean_pair_bootstrap_ci_low': float(np.quantile(mean_boot, .025)),
                'mean_pair_bootstrap_ci_high': float(np.quantile(mean_boot, .975)),
                'mean_pair_bootstrap_valid_draws': int(np.isfinite(mean_boot).sum()),
            }
        result['strata'][name] = stats
    low = table.loc[table.n_train_detected < 200]
    methods = followup.method.unique().tolist()
    result['low_detection_correlations'] = {
        method: {'median': float(low[method].median()), 'minimum': float(low[method].min()),
                 'maximum': float(low[method].max()), 'finite_pairs': int(low[method].notna().sum()),
                 'n_abs_rho_above_0.1': int((low[method].abs() > .1).sum())}
        for method in methods
    }
    result['low_detection_counterexamples_abs_rho_above_0.1'] = [
        {'gene': row.gene, 'protein': row.protein, 'n_train_detected': int(row.n_train_detected),
         'method': method, 'spearman': float(row[method])}
        for _, row in low.iterrows() for method in methods if abs(row[method]) > .1
    ]
    result['per_pair'] = table.to_dict(orient='records')
    for path in [SOURCE / 'evaluation/per_pair.csv', SOURCE / 'selection/selected_pairs.csv',
                 BASE / 'protein_depth/tables/association.csv']:
        result['sources_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    result = clean(result)
    destination = OUT / 'runtime/verified_observation.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_pair'}, indent=2))


if __name__ == '__main__':
    main()
