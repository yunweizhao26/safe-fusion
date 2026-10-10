import json
import sys
import numpy as np
import pandas as pd
from evaluate import OUT, SF, units_for, compiled_ap, interval
from masked_f1_units import load_unit, unit_counts, UNIT_FRACTIONS
from compute_matched_baseline_f1_curves import tie_broken_order
from paired_masked_f1_bootstrap import unit_arrays, pooled_f1

METHOD = 'per_label_expected_count_transductive'


def main(dataset):
    assert dataset in ('Pancreas', 'Colon')
    original = json.loads((OUT / dataset / 'report.json').read_text())
    bootstrap = np.load(OUT / dataset / 'bootstrap_draws.npz')
    draws = bootstrap['draws']
    names = bootstrap['unit_names'].tolist()
    assert draws.shape == (2000, len(names))
    dest = OUT / 'supplementary' / dataset
    dest.mkdir(parents=True, exist_ok=False)
    frames, supports, all_scores, all_labels, all_units = [], [], [], [], []
    for unit in units_for(dataset):
        data = load_unit(unit)
        cells = np.flatnonzero(data.split == 'test')
        rr, cols = np.where(data.counts[cells] == 0)
        rows = cells[rr]
        labels = data.masked[rows, cols]
        donor_names, donor_codes = np.unique(data.unit_labels, return_inverse=True)
        shares = np.zeros((len(donor_names), data.counts.shape[1]), dtype=np.float64)
        for index, donor in enumerate(donor_names):
            members = donor_codes == index
            totals = data.counts[members].sum(axis=0, dtype=np.float64)
            assert totals.sum() > 0
            shares[index] = totals / totals.sum()
            supports.append({'fold': unit.key, 'donor': donor, 'fitting_scope': 'all masked cells; transductive teacher fitting population', 'fitting_cells': int(members.sum()), 'fitting_counts': float(totals.sum()), 'test_cells': int((members & (data.split == 'test')).sum())})
        np.testing.assert_allclose(shares.sum(axis=1), 1, rtol=1e-12)
        scores = (-np.expm1(-data.library[rows] * shares[donor_codes[rows], cols])).astype(np.float32)
        assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
        all_scores.append(scores)
        all_labels.append(labels)
        unique, inverse = np.unique(data.unit_labels[rows], return_inverse=True)
        all_units.append(np.array([unit.key + ':' + x for x in unique])[inverse])
        rank = np.empty(len(rows), dtype=np.int64)
        rank[tie_broken_order(scores, unit.tie_seed)] = np.arange(len(rows))
        for fraction in UNIT_FRACTIONS:
            chosen = rank < max(1, int(round(fraction * len(rows))))
            counts = unit_counts(chosen, labels, data.unit_labels[rows])
            counts['unit'] = unit.key + ':' + counts.unit.astype(str)
            frames.append(counts.assign(dataset=dataset, fold=unit.key, method=METHOD, fraction=fraction))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(dest / 'masked_f1_unit_counts.csv', index=False)
    pd.DataFrame(supports).to_csv(dest / 'label_support.csv', index=False)
    arrays = unit_arrays(frame, names)
    def f1(index):
        return float(pooled_f1(arrays['n_selected'][index], arrays['n_true_positive'][index], arrays['n_masked_positives'][index]).mean())
    point_f1 = f1(slice(None))
    boot_f1 = np.array([f1(d) for d in draws])
    group_names, groups = np.unique(np.concatenate(all_units), return_inverse=True)
    assert group_names.tolist() == names
    weights = np.array([np.bincount(d, minlength=len(names)) for d in draws], dtype=np.int32)
    weights = np.vstack([np.ones((1, len(names)), dtype=np.int32), weights])
    scores, labels = np.concatenate(all_scores), np.concatenate(all_labels)
    boot_ap = compiled_ap(scores, labels, groups, weights)
    from sklearn.metrics import average_precision_score
    np.testing.assert_allclose(boot_ap[0], average_precision_score(labels, scores), rtol=1e-12)
    report = {'dataset': dataset, 'method': METHOD, 'fitting_scope': 'all masked cells, matching actual transductive donor-teacher fitting population; supplementary to strict selector-fitting rule', 'draws': 2000, 'seed': 1729, 'n_units': len(names), 'n_candidates': len(scores), 'mean_masked_f1': interval(point_f1, boot_f1), 'average_precision': interval(boot_ap[0], boot_ap[1:]), 'safe_fusion_minus': {'mean_masked_f1': interval(original['mean_masked_f1'][SF][0] - point_f1, bootstrap['f1_' + SF] - boot_f1), 'average_precision': interval(bootstrap['ap_' + SF][0] - boot_ap[0], bootstrap['ap_' + SF][1:] - boot_ap[1:])}}
    (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(dest / 'bootstrap_draws.npz', draws=draws, unit_names=group_names, f1=boot_f1, ap=boot_ap)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
