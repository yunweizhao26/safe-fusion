import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/paper_evidence/review_round4/label_baselines/masked'
REF = ROOT / 'artifacts/paper_evidence/review_round4/transductive_references'
sys.path.insert(0, str(ROOT / 'scripts'))
from masked_f1_units import Unit, load_unit, count_scale_values, unit_counts, UNIT_FRACTIONS, fraction_name, unit_manifest_entry
from evaluate_condition_masked_f1 import screen_unit, NormanPaths
from compute_matched_baseline_f1_curves import tie_broken_order
from paired_masked_f1_bootstrap import unit_arrays, pooled_f1

SF = 'safe_fusion_labels'
METHODS = (SF, 'weighted_knn_labels', 'scvi_labels', 'per_label_expected_count')


def compiled_ap(scores, labels, groups, weights):
    assert np.isfinite(scores).all()
    order = np.argsort(-scores, kind='stable')
    values = scores[order]
    ends = np.r_[values[1:] != values[:-1], True].astype(np.uint8)
    groups = np.ascontiguousarray(groups[order], dtype=np.int32)
    labels = np.ascontiguousarray(labels[order], dtype=np.uint8)
    weights = np.ascontiguousarray(weights, dtype=np.int32)
    result = np.empty(len(weights), dtype=np.float64)
    lib = ctypes.CDLL(str(OUT / 'weighted_ap.so'))
    lib.weighted_ap.restype = None
    lib.weighted_ap.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 5
    lib.weighted_ap(len(scores), len(weights), weights.shape[1], groups.ctypes.data,
                    labels.ctypes.data, ends.ctypes.data, weights.ctypes.data, result.ctypes.data)
    return result


def selfcheck():
    rng = np.random.default_rng(1729)
    for n in (20, 500):
        labels = rng.integers(0, 2, n).astype(bool)
        scores = rng.integers(0, 6, n).astype(float)
        groups = np.arange(n) % 5
        draws = rng.integers(0, 5, (40, 5))
        weights = np.array([np.bincount(d, minlength=5) for d in draws])
        observed = compiled_ap(scores, labels, groups, weights)
        expected = [average_precision_score(labels, scores, sample_weight=w[groups]) for w in weights]
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)

        repeated = np.repeat(np.arange(n), weights[0, groups])
        np.testing.assert_allclose(observed[0], average_precision_score(labels[repeated], scores[repeated]))
    (OUT / 'selfcheck.json').write_text(json.dumps({'weighted_ap_matches_sklearn': True, 'seed': 1729}) + '\n')


def units_for(dataset):
    if dataset in ('Pancreas', 'Colon'):
        manifest = ROOT / ('artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json' if dataset == 'Pancreas' else 'artifacts/paper_evidence/review_round3/colon_crossfit/units_manifest.json')
        result = []
        for entry in json.loads(manifest.read_text()):
            if entry['dataset'] != dataset or entry['key'] == 'colon':
                continue
            base = REF / 'sex_zeros/masked_donor' / entry['key']
            for key in ('corrupted', 'coordinates', 'splits', 'truth'):
                entry[key] = ROOT / entry[key]
            entry['selector_dir'] = base / 'selector_donor'
            entry['contracts'] = {'weighted_knn_labels': base / 'graph_smooth_donor', 'scvi_labels': base / 'scvi_donor'}
            result.append(Unit(**entry))
        return result
    benchmark = ROOT / 'artifacts/paper_evidence/review_round2/leakage_free/norman_crispra'
    views = REF / 'knockdown/evaluation_inputs/masked'
    norman = NormanPaths(benchmark, benchmark / 'methods', views / 'norman_crispra', benchmark / 'prepared.h5ad', REF / 'knockdown/masked/norman_crispra/selector_mlp_biology_range')
    unit, selectors = screen_unit(dataset, views, norman, 1729)
    unit.selector_dir = selectors['safe_fusion_condition']
    unit.contracts = {'weighted_knn_labels': unit.contracts['knn_condition'], 'scvi_labels': unit.contracts['scvi_condition']}
    return [unit]


def selector_scores(unit, data, rows, cols):
    replay = OUT / 'replay' / unit.key
    assert json.loads((replay / 'verified.json').read_text())['passed']
    scores = pd.read_parquet(replay / 'selected_gene_scores.parquet',
                            columns=['cell_index', 'gene_index', 'selector_score', 'split'], filters=[('split', '==', 'test')])
    ids = scores.cell_index.to_numpy(dtype=np.int64) * data.counts.shape[1] + scores.gene_index.to_numpy()
    order = np.argsort(ids)
    target = rows * data.counts.shape[1] + cols
    assert len(ids) == len(target) and np.array_equal(ids[order], target), 'selector candidate coordinates mismatch'
    return scores.selector_score.to_numpy()[order]


def expected_count(unit, data, rows, cols):
    fit = data.split == unit.fit_split
    labels = sorted(set(data.unit_labels[rows]))
    support = []
    fractions = {}
    for label in labels:
        members = fit & (data.unit_labels == label)
        totals = data.counts[members].sum(axis=0, dtype=np.float64)
        mass = float(totals.sum())
        support.append({'fold': unit.key, 'label': label, 'fit_split': unit.fit_split, 'fitting_cells': int(members.sum()), 'fitting_counts': mass, 'test_cells': int(((data.split == 'test') & (data.unit_labels == label)).sum()), 'available': mass > 0})
        if mass > 0:
            fractions[label] = totals / mass
    if len(fractions) != len(labels):
        return None, support
    names, group = np.unique(data.unit_labels, return_inverse=True)
    shares = np.zeros((len(names), data.counts.shape[1]), dtype=np.float64)
    for k, label in enumerate(names):
        if label in fractions:
            shares[k] = fractions[label]
    return (-np.expm1(-data.library[rows] * shares[group[rows], cols])).astype(np.float32), support


def interval(point, boot):
    return [float(point), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


def main(dataset):
    assert json.loads((OUT / 'selfcheck.json').read_text())['weighted_ap_matches_sklearn']
    dest = OUT / dataset
    dest.mkdir(exist_ok=False)
    units = units_for(dataset)
    (dest / 'units_manifest.json').write_text(json.dumps([unit_manifest_entry(u) for u in units], indent=2) + '\n')
    all_counts, supports, scores, positives, groups = [], [], {m: [] for m in METHODS}, [], []
    for unit in units:
        print('load', unit.key, flush=True)
        data = load_unit(unit)
        cells = np.flatnonzero(data.split == 'test')
        rr, cols = np.where(data.counts[cells] == 0)
        rows = cells[rr]
        labels = data.masked[rows, cols]
        biological = data.unit_labels[rows]
        local_names, local_groups = np.unique(biological, return_inverse=True)
        grouped = np.array([f'{unit.key}:{name}' for name in local_names])[local_groups]
        groups.append(grouped)
        positives.append(labels)
        count_score, support = expected_count(unit, data, rows, cols)
        supports.extend(support)
        for method in METHODS:
            print(unit.key, method, len(rows), flush=True)
            if method == 'per_label_expected_count':
                if count_score is None:
                    continue
                value = count_score
            elif method == SF:
                value = selector_scores(unit, data, rows, cols)
            else:
                value, scale = count_scale_values(unit.contracts[method], data, rows, cols)
            assert np.isfinite(value).all()
            scores[method].append(value)
            if method != SF:
                rank = np.empty(len(rows), dtype=np.int64)
                rank[tie_broken_order(value, unit.tie_seed)] = np.arange(len(rows))
            for fraction in UNIT_FRACTIONS:
                if method == SF:
                    contract = unit.selector_dir / f'safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}'
                    selected = np.load(contract / 'mean.npy', mmap_mode='r')[rows, cols] != 0
                else:
                    selected = rank < max(1, int(round(fraction * len(rows))))
                counts = unit_counts(selected, labels, biological)
                counts['unit'] = unit.key + ':' + counts.unit.astype(str)
                all_counts.append(counts.assign(dataset=dataset, fold=unit.key, method=method, fraction=fraction))
    frame = pd.concat(all_counts, ignore_index=True)
    frame.to_csv(dest / 'masked_f1_unit_counts.csv', index=False)
    pd.DataFrame(supports).to_csv(dest / 'label_support.csv', index=False)

    tissue = dataset in ('Pancreas', 'Colon')
    previous = pd.read_csv(REF / ('sex_zeros/masked_f1/masked_f1_unit_counts.csv' if tissue else 'knockdown/masked_f1/masked_f1_unit_counts.csv'))
    previous = previous[(previous.dataset == dataset) & (previous.method == ('safe_fusion_donor' if tissue else 'safe_fusion_condition'))].copy()
    if not tissue:
        previous['unit'] = dataset + ':' + previous.unit.astype(str)
    columns = ['n_selected', 'n_true_positive', 'n_masked_positives', 'n_zeros']
    pd.testing.assert_frame_equal(previous.set_index(['unit', 'fraction'])[columns].sort_index(), frame[frame.method == SF].set_index(['unit', 'fraction'])[columns].sort_index(), check_dtype=False)
    if not tissue:
        reference = pd.read_csv(REF / 'knockdown/masked_f1/masked_f1_unit_counts.csv')
        for method, original_name in [('weighted_knn_labels', 'knn_condition'), ('scvi_labels', 'scvi_condition')]:
            old = reference[(reference.dataset == dataset) & (reference.method == original_name)].copy()
            old['unit'] = dataset + ':' + old.unit.astype(str)
            pd.testing.assert_frame_equal(old.set_index(['unit', 'fraction'])[columns].sort_index(), frame[frame.method == method].set_index(['unit', 'fraction'])[columns].sort_index(), check_dtype=False)
    names = sorted(frame.unit.unique())
    group_names, group_codes = np.unique(np.concatenate(groups), return_inverse=True)
    assert list(group_names) == names
    labels = np.concatenate(positives)
    assert (np.bincount(group_codes, weights=labels, minlength=len(names)) > 0).all(), 'Every bootstrap unit must have masked positives'
    draws = np.random.default_rng(1729).integers(0, len(names), (2000, len(names)))
    weights = np.array([np.bincount(d, minlength=len(names)) for d in draws], dtype=np.int32)
    weights = np.vstack([np.ones((1, len(names)), dtype=np.int32), weights])
    boot_f1, boot_ap, point_f1 = {}, {}, {}
    for method, part in frame.groupby('method', sort=False):
        arrays = unit_arrays(part, names)
        def mean_f1(indices):
            return float(pooled_f1(arrays['n_selected'][indices], arrays['n_true_positive'][indices], arrays['n_masked_positives'][indices]).mean())
        point_f1[method] = mean_f1(slice(None))
        boot_f1[method] = np.array([mean_f1(draw) for draw in draws])
        print('AP bootstrap', dataset, method, flush=True)
        score = np.concatenate(scores[method])
        assert len(score) == len(labels)
        boot_ap[method] = compiled_ap(score, labels, group_codes, weights)
        np.testing.assert_allclose(boot_ap[method][0], average_precision_score(labels, score), rtol=1e-12)
    report = {'dataset': dataset, 'n_units': len(names), 'n_candidates': len(labels), 'n_masked_positives': int(labels.sum()), 'draws': 2000, 'seed': 1729, 'safe_fusion_counts_exact_match': True, 'mean_masked_f1': {}, 'average_precision': {}, 'safe_fusion_minus': {}, 'unavailable': {}}
    for method in METHODS:
        if method not in boot_f1:
            report['unavailable'][method] = 'No fitting cells with held-out donor labels; no all-cell or global fallback was substituted.'
            continue
        report['mean_masked_f1'][method] = interval(point_f1[method], boot_f1[method])
        report['average_precision'][method] = interval(boot_ap[method][0], boot_ap[method][1:])
        if method != SF:
            report['safe_fusion_minus'][method] = {'mean_masked_f1': interval(point_f1[SF] - point_f1[method], boot_f1[SF] - boot_f1[method]), 'average_precision': interval(boot_ap[SF][0] - boot_ap[method][0], boot_ap[SF][1:] - boot_ap[method][1:])}
    np.savez(dest / 'bootstrap_draws.npz', draws=draws, unit_names=np.array(names), **{f'f1_{k}': v for k,v in boot_f1.items()}, **{f'ap_{k}': v for k,v in boot_ap.items()})
    (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--selfcheck':
        selfcheck()
    else:
        main(sys.argv[1])
