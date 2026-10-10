import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path.cwd() / 'artifacts/paper_evidence/review_round4/protein_depth'
DRAWS = 2000
REFS = ('safe_fusion', 'standard_depth', 'residual_logit')
sys.path.insert(0, str(ROOT / 'scripts'))


def load_hao():
    spec = importlib.util.spec_from_file_location('frozen_hao', ROOT / 'scripts/analyses/protein_multi/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interval(v):
    v = np.asarray(v)
    return tuple(np.nanquantile(v, [.025, .975])) if np.isfinite(v).any() else (np.nan, np.nan)


def record(values, estimate):
    lo, hi = interval(values)
    return dict(estimate=float(estimate), ci_low=float(lo), ci_high=float(hi), defined_draws=int(np.isfinite(values).sum()))


class Ranked:

    def __init__(self, scores, protein, depth):
        self.values = np.column_stack([scores, protein, depth]).astype(float)
        self.inverses = [np.unique(v, return_inverse=True)[1] for v in self.values.T]
        self.m = scores.shape[1]

    def ranks(self, weights):
        return np.asarray([(np.cumsum(c := np.bincount(inv, weights=weights)) - c / 2 + .5)[inv]
                           for inv in self.inverses])

    def calculate(self, weights):
        ranks = self.ranks(weights)
        n = weights.sum()
        if n < 3:
            return np.full((self.m, 3), np.nan), np.nan, ranks
        centered = ranks - (n + 1) / 2
        square = (centered ** 2 * weights).sum(axis=1)
        cov_y = (centered * centered[-2] * weights).sum(axis=1)
        cov_z = (centered * centered[-1] * weights).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            rho_y = cov_y[:self.m] / np.sqrt(square[:self.m] * square[-2])
            rho_z = cov_z[:self.m] / np.sqrt(square[:self.m] * square[-1])
            yz = cov_z[-2] / np.sqrt(square[-2] * square[-1])

            residual = centered - (cov_z / square[-1])[:, None] * centered[-1]
            vr = (residual ** 2 * weights).sum(axis=1)
            partial = (residual[:self.m] * residual[-2] * weights).sum(axis=1) / np.sqrt(vr[:self.m] * vr[-2])

        tol = np.maximum(square, 1) * 1e-24
        partial[(vr[:self.m] <= tol[:self.m]) | (vr[-2] <= tol[-2])] = np.nan
        return np.column_stack([rho_y, rho_z, partial]), yz, ranks


def r_squared(y, weights, group_members):
    n = weights.sum()
    mean = (y * weights).sum() / n
    total = (weights * (y - mean) ** 2).sum()
    within = 0.
    for members in group_members:
        w = weights[members]
        if w.sum():
            v = y[members]
            within += (w * (v - (w * v).sum() / w.sum()) ** 2).sum()
    return 1 - within / total if total > 0 else np.nan


def group_setup(frame, methods, level, threshold):
    result = []
    for label in sorted(frame[level].astype(str).unique()):
        members = np.flatnonzero(frame[level].astype(str).to_numpy() == label)
        result.append((label, members, len(members) >= threshold,
                       Ranked(frame.iloc[members][methods].to_numpy(), frame.protein.to_numpy()[members], frame.log_library.to_numpy()[members])))
    return result


def group_statistics(weights, groups, m):
    individual = np.full((len(groups), m), np.nan)
    total = np.zeros(m)
    denominator = np.zeros(m)
    for g, (_, members, eligible, ranked) in enumerate(groups):
        w = weights[members]
        if w.sum() >= 3:
            rho = ranked.calculate(w)[0][:, 0]
            individual[g] = rho
            valid = np.isfinite(rho) & eligible
            total[valid] += w.sum() * rho[valid]
            denominator[valid] += w.sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        average = total / denominator
    return average, individual


def metadata(case):
    path = OUT / 'inputs/metadata.json'
    if not path.exists():
        path = OUT / f'inputs/metadata_{"hao" if case.startswith("hao_") else case}.json'
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return next(v for v in raw if v.get('id', v.get('case')) == case)
    if case in raw:
        return raw[case]
    return next(v for v in raw['cases'] if v.get('id', v.get('case')) == case)


def pap_job(frame, methods, scope='all', number=0):
    import evaluate_protein_within_state as e
    names, groups = np.unique(frame.target.astype(str), return_inverse=True)
    covariates = {name: frame[name].to_numpy(float) for name in e.CONTINUOUS_COVARIATES}
    return e.Job(protein='PDL1', gene='CD274', scale='clr', cells='recorded_rna_zero', scope=scope,
                 resampling='target_clusters' if number == 0 else 'cells_within_replicate', rankings=tuple(methods),
                 scores=frame[methods].to_numpy(float), response=frame.protein.to_numpy(float), groups=groups,
                 replicate=np.unique(frame.replicate.astype(str), return_inverse=True)[1], covariates=covariates,
                 within=number == 0, covariate_sets={}, seed=(1729, list(e.PROTEIN_TO_GENE).index('PDL1'), 0, 0, number),
                 draws=DRAWS, group_names=list(names))


def evaluate(case):
    start = time.time()
    meta = metadata(case)
    frame = pd.read_parquet(OUT / f'inputs/{case}.parquet')
    methods = meta['methods']
    assert set(REFS) <= set(methods)
    assert frame.cell_id.is_unique
    assert np.isfinite(frame[methods + ['protein', 'log_library']].to_numpy(float)).all()
    n, m = len(frame), len(methods)
    hao = case.startswith('hao_')
    levels = [('cell_type_l1', 'l1', 50), ('cell_type_l2', 'l2', 50)] if hao else [('target', 'target', 3)]
    groups = {short: group_setup(frame, methods, column, threshold) for column, short, threshold in levels}
    metrics = ['pooled_spearman', 'pooled_mean_auroc', 'depth_spearman', 'partial_depth_spearman'] + [f'within_{short}_spearman' for _, short, _ in levels]
    diagnostics = ['protein_depth_spearman'] + [f'protein_variance_explained_{short}' for _, short, _ in levels]
    ranked = Ranked(frame[methods].to_numpy(), frame.protein.to_numpy(), frame.log_library.to_numpy())
    frozen = load_hao()
    inverses, bucket, thresholds = frozen.metric_setup(frame[methods].to_numpy(), frame.protein.to_numpy())
    if not hao:
        import evaluate_protein_within_state as e
        job = pap_job(frame, methods)
    def calculate(weights, index=None, sampled_groups=None):
        base, yz, _ = ranked.calculate(weights)
        result = np.full((m, len(metrics)), np.nan)
        result[:, 0] = base[:, 0]
        result[:, 2:4] = base[:, 1:3]
        if hao:

            result[:, :2] = frozen.metrics(weights, inverses, bucket)
        else:

            original = e.statistics(job, index, sampled_groups)
            for k, method in enumerate(methods):
                result[k, 0] = original[('pooled_spearman', '', method)]
                result[k, 1] = original[('pooled_mean_auroc', '', method)]
        diag = [yz]
        details = {}
        for g, (_, short, _) in enumerate(levels):
            avg, individual = group_statistics(weights, groups[short], m)
            if not hao:
                avg = np.asarray([original.get(('within_target_spearman', '', method), np.nan) for method in methods])
            result[:, 4 + g] = avg
            details[short] = individual
            diag.append(r_squared(frame.protein.to_numpy(), weights, [item[1] for item in groups[short]]))
        return result, np.asarray(diag), details
    point, point_diag, point_groups = calculate(np.ones(n), np.arange(n), None if hao else job.groups)
    boot = np.empty((DRAWS, m, len(metrics)))
    boot_diag = np.empty((DRAWS, len(diagnostics)))
    boot_groups = {short: np.empty((DRAWS, len(value), m)) for short, value in groups.items()}
    if hao:
        units = [np.flatnonzero(frame.donor.to_numpy() == d) for d in sorted(frame.donor.unique())]
        rng = np.random.default_rng(1729)
    else:
        units = [np.flatnonzero(job.groups == code) for code in np.unique(job.groups)]
        rng = np.random.default_rng(list(job.seed))
    for draw in range(DRAWS):
        if hao:
            index = np.concatenate([rng.choice(unit, len(unit), replace=True) for unit in units])
            sampled_groups = None
        else:
            chosen = rng.choice(len(units), size=len(units), replace=True)
            index = np.concatenate([units[j] for j in chosen])
            sampled_groups = np.concatenate([np.full(len(units[j]), copy) for copy, j in enumerate(chosen)])
        boot[draw], boot_diag[draw], detail = calculate(np.bincount(index, minlength=n), index, sampled_groups)
        for short in groups:
            boot_groups[short][draw] = detail[short]
        if draw and draw % 500 == 0:
            print(json.dumps(dict(case=case, draws=draw, seconds=round(time.time() - start, 1))), flush=True)
    base = dict(case=case, pair_index=int(case[4:]) if hao else -1, gene=meta['gene'], protein=meta.get('antibody', meta.get('protein')),
                n_train_detected=meta.get('n_train_detected'), n_test_zero=n)
    associations, differences = [], []
    for k, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            associations.append(dict(base, method=method, metric=metric, **record(boot[:, k, j], point[k, j])))
            for ref in REFS:
                if ref != method:
                    r = methods.index(ref)
                    differences.append(dict(base, ranking=ref, comparator=method, metric=metric,
                                            **record(boot[:, r, j] - boot[:, k, j], point[r, j] - point[k, j])))
    group_rows = []
    for short, values in groups.items():
        for g, (label, members, eligible, _) in enumerate(values):
            for k, method in enumerate(methods):
                group_rows.append(dict(base, group_level=short, group=label, n_cells=len(members), eligible=eligible, method=method,
                                       metric='spearman', **record(boot_groups[short][:, g, k], point_groups[short][g, k])))
    for subdir in ['tables', 'bootstrap', 'runtime']:
        (OUT / subdir).mkdir(exist_ok=True)
    pd.DataFrame(associations).to_csv(OUT / f'tables/{case}_association.csv', index=False)
    pd.DataFrame(differences).to_csv(OUT / f'tables/{case}_differences.csv', index=False)
    pd.DataFrame(group_rows).to_csv(OUT / f'tables/{case}_groups.csv', index=False)
    pd.DataFrame([dict(base, metric=metric, **record(boot_diag[:, j], point_diag[j])) for j, metric in enumerate(diagnostics)]).to_csv(OUT / f'tables/{case}_diagnostics.csv', index=False)
    np.savez_compressed(OUT / f'bootstrap/{case}.npz', point=point, bootstrap=boot, methods=np.asarray(methods), metrics=np.asarray(metrics),
                        point_diagnostics=point_diag, bootstrap_diagnostics=boot_diag, diagnostics=np.asarray(diagnostics), thresholds=thresholds,
                        **{f'group_{short}_bootstrap': value for short, value in boot_groups.items()})
    audit = dict(case=case, draws=DRAWS, seed=1729 if hao else list(job.seed), seconds=time.time()-start, no_refitting=True)
    if hao:
        old = np.load(ROOT / f'artifacts/paper_evidence/review_round4/protein_multi/corrected/evaluation/pair_{int(case[4:]):03}_bootstrap.npz')
        aliases = ['safe_fusion', 'svd', 'weighted_knn', 'magic', 'scvi', 'p_nonzero', 'library_size']
        take = [methods.index(x) for x in aliases]
        np.testing.assert_allclose(point[take, :2], old['point'], atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_allclose(boot[:, take, :2], old['bootstrap'], atol=1e-12, rtol=0, equal_nan=True)
        audit['original_hao_2000_draws_replayed'] = True
    else:
        audit['unchanged_S20_statistics'] = True
        source = 'protein_full/verification/evaluation' if case == 'papalexi_small' else 'protein_followup/full/evaluation'
        saved = pd.read_csv(ROOT / f'artifacts/paper_evidence/review_round4/{source}/association.csv')
        saved = saved.loc[(saved.scope == 'all') & (saved.resampling == 'target_clusters') & saved.statistic.isin(metrics)]
        actual = pd.DataFrame(associations).rename(columns={'method': 'ranking', 'metric': 'statistic'})
        merged = saved.merge(actual, on=['ranking', 'statistic'], suffixes=('_old', '_new'), validate='one_to_one')
        assert len(merged) == len(saved)
        for field in ['estimate', 'ci_low', 'ci_high']:
            np.testing.assert_allclose(merged[field + '_old'], merged[field + '_new'], atol=1e-12, rtol=0, equal_nan=True)
        audit['original_papalexi_estimates_and_intervals_replayed'] = len(merged)
    (OUT / f'runtime/{case}_evaluation.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)


def original_tables(case):

    import evaluate_protein_within_state as e
    frame = pd.read_parquet(OUT / f'inputs/{case}.parquet')
    methods = metadata(case)['methods']
    association, paired = [], []
    for number, (scope, subset) in enumerate([('all', frame), ('all', frame), ('control', frame.loc[frame.target == e.CONTROL])]):
        names = [m for m in methods if not (scope == 'control' and m == 'target_baseline')]
        job = pap_job(subset, names, scope, number)
        job.covariate_sets = e.CONTROL_COVARIATE_SETS if scope == 'control' else e.COVARIATE_SETS if number == 0 else {}

        point = e.statistics(job, np.arange(len(subset)), job.groups)
        rng = np.random.default_rng(list(job.seed))
        clusters = [np.flatnonzero(job.groups == c) for c in np.unique(job.groups)]
        reps = [np.flatnonzero(job.replicate == c) for c in np.unique(job.replicate)]
        samples = []
        for _ in range(DRAWS):
            if number == 0:
                chosen = rng.choice(len(clusters), size=len(clusters), replace=True)
                index = np.concatenate([clusters[c] for c in chosen])
                groups = np.concatenate([np.full(len(clusters[c]), copy) for copy, c in enumerate(chosen)])
            else:
                index = np.concatenate([rng.choice(unit, len(unit), replace=True) for unit in reps])
                groups = job.groups[index]
            samples.append(e.statistics(job, index, groups))
        keys = list(point)
        matrix = np.asarray([[s[key] for key in keys] for s in samples])
        base = dict(case=case, scope=scope, resampling=job.resampling, n_cells=len(subset))
        for col, key in enumerate(keys):
            metric, covariates, method = key
            association.append(dict(base, metric=metric, covariates=covariates, method=method, **record(matrix[:, col], point[key])))
            for ref in REFS:
                other = (metric, covariates, ref)
                if ref != method and other in point:
                    r = keys.index(other)
                    paired.append(dict(base, metric=metric, covariates=covariates, ranking=ref, comparator=method,
                                       **record(matrix[:, r] - matrix[:, col], point[other] - point[key])))
        np.savez_compressed(OUT / f'bootstrap/{case}_original_scope{number}.npz', point=np.asarray(list(point.values())), bootstrap=matrix, keys=np.asarray(keys))
    actual = pd.DataFrame(association)
    source = 'protein_full/verification/evaluation' if case == 'papalexi_small' else 'protein_followup/full/evaluation'
    saved = pd.read_csv(ROOT / f'artifacts/paper_evidence/review_round4/{source}/association.csv').rename(columns={'ranking': 'method', 'statistic': 'metric'})
    saved['covariates'] = saved.covariates.fillna('')
    merged = saved.merge(actual, on=['scope', 'resampling', 'metric', 'covariates', 'method'], suffixes=('_old', '_new'), validate='one_to_one')
    assert len(merged) == len(saved), (len(merged), len(saved))
    for field in ['estimate', 'ci_low', 'ci_high']:
        np.testing.assert_allclose(merged[field + '_old'], merged[field + '_new'], atol=1e-12, rtol=0, equal_nan=True)
    (OUT / f'runtime/{case}_original_replay.json').write_text(json.dumps(dict(passed=True, rows=len(merged), tolerance=1e-12)) + '\n')
    actual.to_csv(OUT / f'tables/{case}_original_association.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / f'tables/{case}_original_differences.csv', index=False)
    print(f'{case}: original S20 scopes complete', flush=True)


def selftest():
    import evaluate_protein_within_state as e
    from evaluate_papalexi_crossmodal import safe_spearman, quantile_metrics
    rng = np.random.default_rng(1729)
    scores = rng.integers(0, 7, (53, 4)).astype(float)
    scores[:, -1] = 1
    protein = rng.integers(0, 9, 53).astype(float)
    depth = rng.integers(0, 13, 53).astype(float)
    scores = np.column_stack([scores, depth])
    ranked = Ranked(scores, protein, depth)
    h = load_hao()
    inv, bucket, thresholds = h.metric_setup(scores, protein)
    error = 0.
    for repeat in range(12):
        index = np.arange(53) if repeat == 0 else np.r_[rng.choice(23, 23), rng.choice(np.arange(23, 53), 30)]
        w = np.bincount(index, minlength=53)
        got, yz, _ = ranked.calculate(w)
        rz = rankdata(depth[index])
        design = np.column_stack([np.ones(len(index)), rz])
        ry = rankdata(protein[index]); yr = ry-design @ np.linalg.lstsq(design, ry, rcond=None)[0]
        for k in range(scores.shape[1]):
            x = scores[index, k]
            want = safe_spearman(x, protein[index])
            np.testing.assert_allclose(got[k, 0], want, atol=1e-12, rtol=0, equal_nan=True)
            if k not in (3, 4):
                rx = rankdata(x); xr = rx-design @ np.linalg.lstsq(design, rx, rcond=None)[0]
                partial = np.corrcoef(xr, yr)[0, 1]
                np.testing.assert_allclose(got[k, 2], partial, atol=1e-12, rtol=0)
                error = max(error, abs(got[k, 2]-partial))
            else:
                assert np.isnan(got[k, 2])
        old = h.metrics(w, inv, bucket)
        for k in range(scores.shape[1]):
            want_auc = np.nanmean([quantile_metrics((protein[index] > t).astype(int), scores[index, k])[0] for t in thresholds]) if k != 3 else np.nan
            np.testing.assert_allclose(old[k, 1], want_auc, atol=1e-12, rtol=0, equal_nan=True)
        group = np.repeat(np.arange(4), [15, 17, 10, 11])
        frame = pd.DataFrame(scores, columns=[f'x{k}' for k in range(5)]).assign(protein=protein, log_library=depth, target=group)
        setup = group_setup(frame, [f'x{k}' for k in range(5)], 'target', 3)

        own, _ = group_statistics(w, setup, 5)
        want = e.within_group_spearman(scores[index], protein[index], group[index])
        np.testing.assert_allclose(own, want, atol=1e-12, rtol=0, equal_nan=True)
    (OUT / 'runtime').mkdir(exist_ok=True)
    report = dict(passed=True, draws_checked=12, partial_max_absolute_error=error, tolerance=1e-12,
                  checked=['original_Hao_Spearman_AUROC', 'S20_within_group', 'rank_residual_partial', 'ties', 'constant_score', 'self_adjusted_depth_undefined'])
    (OUT / 'runtime/metrics_selftest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['case', 'pair', 'selftest', 'original-tables'])
    parser.add_argument('case', nargs='?')
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='An input array is constant')
        if args.command == 'selftest':
            selftest()
        elif args.command == 'pair':
            i = int(os.environ['SLURM_ARRAY_TASK_ID'])
            evaluate(f'hao_{i:03}' if i < 41 else ['papalexi_small', 'papalexi_full'][i - 41])
        elif args.command == 'case':
            evaluate(args.case)
        else:
            original_tables(args.case)
