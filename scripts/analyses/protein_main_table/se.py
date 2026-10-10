import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_main_table')
ROUND = OUT.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_protein_within_state as e

ROWS = {
    'small': {'safe_fusion': 'safe_fusion', 'svd': 'svd', 'weighted_knn': 'weighted_knn', 'scvi': 'scvi',
              'p_nonzero': 'p_nonzero', 'magic': 'magic', 'target_baseline': 'target_baseline',
              'ifng_score': 'ifng_score', 'library_size': 'library_size'},
    'full': {'safe_fusion': 'safe_fusion', 'svd': 'svd_all_cells', 'weighted_knn': 'weighted_knn_all_cells',
             'scvi': 'scvi', 'p_nonzero': 'p_nonzero', 'magic': 'magic', 'target_baseline': 'target_baseline',
             'ifng_score': 'ifng_score', 'library_size': 'library_size'},
}


def run(block):
    frame = pd.read_parquet(ROUND / f'protein_s20_rows/{block}/pdl1_cells.parquet')
    columns = list(ROWS[block].values())
    group_names, groups = np.unique(frame.target.astype(str), return_inverse=True)
    job = e.Job(protein='PDL1', gene='CD274', scale='clr', cells='recorded_rna_zero', scope='all',
                resampling='target_clusters', rankings=tuple(columns),
                scores=frame[columns].to_numpy(dtype=float), response=frame.adt_clr.to_numpy(dtype=float),
                groups=groups, replicate=np.unique(frame.replicate.astype(str), return_inverse=True)[1],
                covariates={n: frame[n].to_numpy(dtype=float) for n in e.CONTINUOUS_COVARIATES},
                within=True, covariate_sets={}, seed=(1729, list(e.PROTEIN_TO_GENE).index('PDL1'), 0, 0, 0),
                draws=2000, group_names=list(group_names))
    point = e.statistics(job, np.arange(len(frame)), job.groups)
    rng = np.random.default_rng(list(job.seed))
    clusters = [np.flatnonzero(job.groups == code) for code in np.unique(job.groups)]
    samples = []
    for _ in range(job.draws):
        chosen = rng.choice(len(clusters), size=len(clusters), replace=True)
        index = np.concatenate([clusters[c] for c in chosen])
        copies = np.concatenate([np.full(len(clusters[c]), k) for k, c in enumerate(chosen)])
        samples.append(e.statistics(job, index, copies))
    samples = pd.DataFrame(samples)
    rows = []
    for (statistic, _, column), estimate in point.items():
        if statistic == 'pooled_mean_auroc':
            continue
        draws = samples[(statistic, '', column)].to_numpy(dtype=float)
        rows.append({'block': block, 'statistic': statistic, 'ranking': column, 'estimate': estimate,
                     'se': float(np.nanstd(draws, ddof=1)), 'ci_low': float(np.nanquantile(draws, 0.025)),
                     'ci_high': float(np.nanquantile(draws, 0.975)), 'defined_draws': int(np.isfinite(draws).sum())})
    result = pd.DataFrame(rows)
    saved = pd.read_csv(ROUND / f'protein_s20_rows/{block}/association.csv').fillna({'covariates': ''})
    saved = saved[(saved.scope == 'all') & (saved.resampling == 'target_clusters') & (saved.protein == 'PD-L1')
                  & (saved.scale == 'clr') & (saved.cells == 'recorded_rna_zero') & (saved.covariates == '')]
    joined = result.merge(saved, on=['statistic', 'ranking'], suffixes=('', '_saved'), validate='one_to_one')
    assert len(joined) == len(result), (len(joined), len(result))
    for c in ['estimate', 'ci_low', 'ci_high']:
        np.testing.assert_allclose(joined[c], joined[c + '_saved'], rtol=0, atol=1e-12)
    return result


if __name__ == '__main__':
    out = pd.concat([run('small'), run('full')], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'estimates.csv', index=False)
    print(out.to_string())
