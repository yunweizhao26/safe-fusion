
import csv
import io
import json
from pathlib import Path

import run
from collect_units import collect

np, pd = run.np, run.pd
out = run.OUT
original = out.parent / 'fill_grid' / 'fill_grid_long.csv'
gate_collected = collect('gate', [1, 10])
values = collect('dense', [p for p in run.PCTS if p not in [1, 10]])
baseline_keys = ['dataset', 'endpoint', 'unit']
pd.testing.assert_frame_equal(
    gate_collected[gate_collected.matrix == 'corrupted_raw'].set_index(baseline_keys).sort_index(),
    values[values.matrix == 'corrupted_raw'].set_index(baseline_keys).sort_index(),
    check_exact=True)

gate_values = pd.read_csv(out / 'gate_unit_values.csv', dtype={'unit': str})
values = pd.concat([gate_values, values[values.matrix != 'corrupted_raw']], ignore_index=True)
values.to_csv(out / 'unit_values.csv', index=False)
result = run.summarize(values, run.PCTS)
for column in ['error_reduction', 'ci_low', 'ci_high']:
    result[column + '_x100'] = 100 * result[column]
result['fill_fraction'] = result.fill_pct / 100
result['method_label'] = result.method.map(dict(zip(run.METHODS, run.LABELS)))
keys = ['dataset', 'endpoint', 'method', 'fill_pct']
assert len(result) == 728 and not result.duplicated(keys).any()
assert list(result.columns) == next(csv.reader(original.open()))

gap_rows = []
draws = {}
diagnostics = []
for number, ((dataset, endpoint), frame) in enumerate(values.groupby(['dataset', 'endpoint'], sort=False)):
    pivot = frame.pivot(index='unit', columns='matrix', values='value')
    units = pivot.index
    reference = frame.drop_duplicates('unit').set_index('unit')['unthinned'].reindex(units).to_numpy()
    strata = frame.drop_duplicates('unit').set_index('unit')['stratum'].reindex(units).to_numpy()
    index = run.thin.unit_draws(strata, 2000, 1729)
    raw_error = np.abs(pivot['corrupted_raw'].to_numpy() - reference)
    denominator = np.nanmean(raw_error[index], axis=1)
    assert np.isfinite(denominator).all() and (denominator > 0).all()
    prefix = f'endpoint_{number}'
    draws[prefix + '_indices'] = index
    draws[prefix + '_units'] = units.to_numpy(dtype=str)
    draws[prefix + '_denominator'] = denominator
    names, numerators, ratios = [], [], []
    for method in run.METHODS:
        for pct in run.PCTS:
            name = run.name(method, pct)
            reduction = raw_error - np.abs(pivot[name].to_numpy() - reference)
            numerator = np.nanmean(reduction[index], axis=1)
            boot_ratio = (numerator / denominator) * 100
            low, high = np.nanquantile(boot_ratio, [0.025, 0.975])
            row = result[(result.dataset == dataset) & (result.endpoint == endpoint)
                         & (result.method == method) & (result.fill_pct == pct)].iloc[0]

            assert row.error_reduction == float(np.nanmean(reduction))
            assert row.error_unfilled == float(np.nanmean(raw_error))
            assert np.array_equal(np.nanquantile(numerator, [0.025, 0.975]), [row.ci_low, row.ci_high])
            gap_rows.append(dict(dataset=dataset, endpoint=endpoint, method=method, fill_pct=pct,
                gap_closed_pct=(row.error_reduction / row.error_unfilled) * 100,
                gap_ci_low_pct=float(low), gap_ci_high_pct=float(high)))
            names.append(name)
            numerators.append(numerator)
            ratios.append(boot_ratio)
    draws[prefix + '_methods'] = np.asarray(names)
    draws[prefix + '_numerator'] = np.asarray(numerators)
    draws[prefix + '_gap_pct'] = np.asarray(ratios)
    diagnostics.append(dict(array_prefix=prefix, dataset=dataset, endpoint=endpoint,
        n_units=len(units), draws=2000, seed=1729, min_denominator=float(denominator.min()),
        max_denominator=float(denominator.max()), units=units.astype(str).tolist()))
result = result.merge(pd.DataFrame(gap_rows), on=keys, how='left', validate='one_to_one', sort=False)
numeric = result.select_dtypes('number')
assert np.isfinite(numeric).all().all()
assert (result.gap_ci_low_pct <= result.gap_ci_high_pct).all()
result.to_csv(out / 'fill_grid_dense_long.csv', index=False)
np.savez_compressed(out / 'bootstrap_draws.npz', **draws)
run.dump(out / 'bootstrap_diagnostics.json', diagnostics)

def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def key(row):
    return tuple(row[k] for k in keys[:-1]) + (float(row['fill_pct']),)

old = read_rows(original)
new = {key(row): row for row in read_rows(out / 'fill_grid_dense_long.csv')}
comparisons = []
for row in old:
    other = new[key(row)]
    for column in ['error_reduction', 'ci_low', 'ci_high']:
        comparisons.append({**{k: row[k] for k in keys}, 'column': column,
            'original': row[column], 'regenerated': other[column],
            'exact_printed_match': row[column] == other[column],
            'absolute_difference': abs(float(row[column]) - float(other[column]))})
pd.DataFrame(comparisons).to_csv(out / 'reproduction_check.csv', index=False)
receipt = dict(rows=728, overlapping_rows=len(old), compared_values=len(comparisons),
    exact_printed_matches=sum(r['exact_printed_match'] for r in comparisons),
    max_absolute_difference=max(r['absolute_difference'] for r in comparisons),
    fractions=run.PCTS, bootstrap_draws=2000, seed=1729,
    same_draw_reduction_interval_checks=728,
    all_gap_denominators_positive=True)
receipt['passed'] = len(comparisons) == 1176 and all(
    r['exact_printed_match'] or (r['dataset'] == 'zebrafish' and r['endpoint'] == 'Stage error'
    and r['absolute_difference'] <= 5e-8
    and f"{100 * float(r['original']):.2f}" == f"{100 * float(r['regenerated']):.2f}")
    for r in comparisons)
receipt['mismatches_by_endpoint'] = pd.DataFrame(comparisons).query('not exact_printed_match').groupby(['dataset', 'endpoint']).size().to_dict()
receipt['mismatches_by_endpoint'] = {str(k): int(v) for k, v in receipt['mismatches_by_endpoint'].items()}
run.dump(out / 'reproduction.json', receipt)
print(json.dumps(receipt, indent=2), flush=True)
assert receipt['passed'], 'Existing percentages exceed the documented tolerance; inspect reproduction_check.csv'
