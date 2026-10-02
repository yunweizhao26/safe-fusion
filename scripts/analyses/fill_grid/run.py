import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path[:0] = ['scripts', 'src']
import anndata as ad
import numpy as np
import pandas as pd
import r3_downstream_thinning as thin
from selector_attribution import exact_topk
from safefusion_benchmark.contracts import write_output_contract

OUT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/fill_grid')
BASE = Path('artifacts/paper_evidence/review_round4/transductive_downstream')
EXTRA = Path('artifacts/paper_evidence/review_round4/reviewer_extras/B_thinning')
D = [d for d in json.loads((BASE/'manifest.json').read_text()) if d['design'] == 'thinning']
METHODS = ['detection', 'transductive', 'svd', 'magic', 'scvi', 'alra', 'selector_scvi', 'selector_magic']
PCTS = [0.25, 0.5, 1, 2, 3, 5, 10]
SPECS = {
    'colon': [('Marker AUPRC', 'canonical_marker_pr_auc'), ('Reference mapping F1', 'cell_identity_macro_f1')],
    'pancreas': [('Marker AUPRC', 'canonical_marker_pr_auc'), ('Reference mapping F1', 'cell_identity_macro_f1')],
    'zebrafish': [('Dynamics', 'dynamic_gene_spearman'), ('Stage error', 'stage_rank_mae')],
    'norman_crispra': [('Edge AUPRC', 'edge_pr_auc_q10')],
}
LABELS = ['Safe Fusion, detection-weighted', 'Safe Fusion, fused', 'SVD', 'MAGIC', 'scVI', 'ALRA', 'Selector on scVI', 'Selector on MAGIC']
TITLES = ['Colon marker AUPRC', 'Colon mapping F1', 'Pancreas marker AUPRC', 'Pancreas mapping F1', 'Zebrafish dynamics', 'Zebrafish stage error', 'Norman edge AUPRC']

def name(method, pct):
    return f'{method}_{pct:g}pct'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def summarize(values, pcts):
    thin.METHODS = dict(zip(METHODS, METHODS))
    thin.PCTS = pcts

    thin.method_names = lambda: [name(m, p) for m in METHODS for p in pcts]
    thin.split_name = lambda n: (n.rsplit('_', 1)[0], float(n.rsplit('_', 1)[1].removesuffix('pct')))
    return thin.summarize_units(values, 2000, 1729)

def unit(stage, index):
    if stage == 'grid':
        assert json.loads((OUT/'gate.json').read_text())['passed']
    pcts = [1, 10] if stage == 'gate' else [0.25, 0.5, 2, 3, 5]
    d = D[index]; p = Path(d['dir']); key = d['key']
    dest = OUT/stage/key
    dest.mkdir(parents=True, exist_ok=False)
    a = ad.read_h5ad(p/'recorded.h5ad')
    recorded = thin.dense(a.layers['corrupted_counts']).astype(np.float32)
    split = pd.read_parquet(p/'splits.parquet').set_index('cell_id').loc[a.obs_names, 'split'].to_numpy()
    test = split == 'test'
    hybrid = ad.read_h5ad(p/'hybrid.h5ad')
    assert np.array_equal(recorded[test], thin.dense(hybrid.layers['corrupted_counts'])[test])
    del hybrid
    rows, cols = np.where((recorded == 0) & test[:, None])
    saved = np.load(p/'transductive/selector/detection.npz')
    assert np.array_equal(saved['rows'], rows) and np.array_equal(saved['cols'], cols)
    checks = []; sources = {p/'recorded.h5ad', p/'hybrid.h5ad', p/'truth.h5ad', p/'splits.parquet', p/'transductive/selector/detection.npz'}
    for method in METHODS:
        oldroot = EXTRA/key if method in METHODS[5:] else p
        template = oldroot/name(method, 1)/'metadata.json'
        meta = json.loads(template.read_text())
        sources.add(template)
        if method in ['detection', 'transductive']:
            source = p/'transductive/safe_fusion'
            score = saved['scores']
        elif method == 'svd':
            source = p/'inductive/svd_impute'
        elif method == 'alra':
            source = EXTRA/key/'alra'
        else:
            source = p/'transductive'/f'{method.removeprefix("selector_")}_inductive'
        source_meta = json.loads((source/'metadata.json').read_text())
        assert source_meta['cell_ids'] == a.obs_names.astype(str).tolist()
        assert source_meta['gene_ids'] == a.var_names.astype(str).tolist()
        assert source_meta['scale'] == 'counts'
        value = np.load(source/'mean.npy', mmap_mode='r')[rows, cols]
        if method not in ['detection', 'transductive']:
            value = np.maximum(value, 0)
            score = value
        if method.startswith('selector_'):
            scores_path = EXTRA/key/method/'test_scores.npz'
            scores = np.load(scores_path)
            assert np.array_equal(scores['rows'], rows) and np.array_equal(scores['cols'], cols)
            score = scores['score']; sources.add(scores_path)
        assert np.isfinite(score).all() and np.isfinite(value).all()
        if method == 'detection':
            value = value * saved['detection']
        sources.update([source/'mean.npy', source/'metadata.json'])
        for pct in pcts:
            selected = exact_topk(score, max(1, round(pct/100 * len(rows))))
            result = recorded.copy()
            result[rows[selected], cols[selected]] = value[selected]
            assert np.array_equal(result[~test], recorded[~test])
            assert np.array_equal(result[recorded > 0], recorded[recorded > 0])
            assert selected.sum() == max(1, round(pct/100 * len(rows)))
            equal = None
            if pct in [1, 5, 10]:
                oldpath = oldroot/name(method, pct)/'mean.npy'
                equal = np.array_equal(result, np.load(oldpath, mmap_mode='r'))
                assert equal, (key, method, pct, 'fill differs from saved contract')
                sources.add(oldpath)
            m = copy.deepcopy(meta); m['method'] = name(method, pct)
            m['parameters']['fill_fraction'] = pct/100
            m['parameters']['fill_grid'] = dict(source_contract=str(source), saved_scores=True, refitted=False)
            decision = m['parameters'].get('decision_layer')
            if decision:
                decision.update(fill_budget=pct/100, threshold=float(np.min(score[selected])))
                decision.pop('calibration_threshold', None)
            checks.append(dict(method=method, fill_pct=pct, candidates=len(rows), selected=int(selected.sum()), changed=int(np.sum(value[selected] != 0)), saved_fill_bitwise_equal=equal))
            write_output_contract(dest/name(method, pct), result, m)
        print(key, method, 'fills complete', flush=True)
    pd.DataFrame(checks).to_csv(dest/'fill_checks.csv', index=False)
    dump(dest/'source_sha256.json', {str(path): sha(path) for path in sorted(sources)})
    methods = sum((['--method', f'{name(m, pct)}={dest/name(m, pct)}'] for m in METHODS for pct in pcts), [])
    args = ['--truth', p/'truth.h5ad', '--corrupted', p/'recorded.h5ad', '--splits', p/'splits.parquet', *methods, '--output-dir', dest/'evaluation', '--bootstrap', 2000, '--seed', 1729, '--allow-transductive']
    if key.startswith('pancreas'):
        script = BASE/'code/pancreas_fold.py'; args += ['--coordinates', p/'empty_coordinates.parquet', '--fold', key[-1]]
    elif key.startswith('colon'):
        script = Path('scripts/evaluate_colon_donor_biology.py'); args += ['--coordinates', p/'empty_coordinates.parquet', '--marker-panel', 'source']
    elif key == 'zebrafish':
        script = Path('scripts/evaluate_trajectory_preservation.py')
    else:
        script = Path('scripts/evaluate_interventional_grn.py'); args += ['--dataset', 'norman_crispra', '--intervention', 'gain_of_function', '--publication-doi', '10.1126/science.aax4438']
    cmd = ['.venv/bin/python', str(script), *map(str, args)]
    dump(dest/'command.json', cmd)
    subprocess.run(cmd, check=True)
    (dest/'DONE').write_text('completed\n')

def collect(stage, pcts):
    old = pd.read_csv(BASE/'summary/thinning_unit_values.csv', dtype={'unit': str})
    frames = []; checks = []
    for dataset, endpoints in SPECS.items():
        donors = dataset in ['colon', 'pancreas']
        keys = [f'{dataset}_{i}' for i in range(3)] if donors else [dataset]
        unitcol = 'donor' if donors else 'unit'
        filename = 'donor_metrics.parquet' if donors else 'unit_metrics.parquet'
        for k in keys:
            assert (OUT/stage/k/'DONE').exists()
        frame = pd.concat([pd.read_parquet(OUT/stage/k/'evaluation'/filename) for k in keys])
        for endpoint, metric in endpoints:
            f = frame[frame.metric == metric].copy(); f[unitcol] = f[unitcol].astype(str)
            pivot = f.pivot(index=unitcol, columns='method', values='value')
            ref = old[(old.dataset == dataset) & (old.endpoint == endpoint) & (old.matrix == 'corrupted_raw')].set_index('unit').reindex(pivot.index)
            assert ref.unthinned.notna().all()
            for column, savedcol in [('corrupted_raw', 'value'), ('reference_truth', 'thinned_truth_pipeline')]:
                delta = float(np.nanmax(abs(pivot[column].to_numpy() - ref[savedcol].to_numpy())))
                assert delta < 1e-12, (dataset, endpoint, column, delta)
                checks.append(dict(dataset=dataset, endpoint=endpoint, reference=column, max_difference=delta))
            for n in ['corrupted_raw'] + [name(m, pct) for m in METHODS for pct in pcts]:
                frames.append(pd.DataFrame(dict(dataset=dataset, endpoint=endpoint, unit=pivot.index, stratum=ref.stratum.to_numpy(), matrix=n, value=pivot[n].to_numpy(), unthinned=ref.unthinned.to_numpy(), thinned_truth_pipeline=ref.thinned_truth_pipeline.to_numpy())))
    values = pd.concat(frames, ignore_index=True)
    values.to_csv(OUT/f'{stage}_unit_values.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/f'{stage}_reference_parity.csv', index=False)
    return values

def gate():
    result = summarize(collect('gate', [1, 10]), [1, 10])
    result.to_csv(OUT/'gate_reproduced.csv', index=False)
    old = pd.concat([pd.read_csv(BASE/'summary/thinning_reference.csv'), pd.read_csv(EXTRA/'new_rows.csv')])
    old = old[old.method.isin(METHODS) & old.fill_pct.isin([1, 10])]
    keys = ['dataset', 'endpoint', 'method', 'fill_pct']
    a = old.set_index(keys).sort_index(); b = result.set_index(keys).sort_index()
    assert list(a.index) == list(b.index) and len(a) == 112
    cols = list(a.select_dtypes('number').columns)
    delta = np.abs(a[cols].to_numpy() - b[cols].to_numpy())
    pd.DataFrame(delta, index=a.index, columns=cols).to_csv(OUT/'gate_numeric_differences.csv')
    assert np.allclose(a[cols], b[cols], rtol=0, atol=2e-14, equal_nan=True), float(np.nanmax(delta))

    tex = (Path(__import__('os').environ['SAFE_FUSION_MANUSCRIPT_DIR']) / '6_results.tex').read_text()
    lines = tex.split('\\label{tab:deployment}', 1)[1].split('\\end{table*}', 1)[0].splitlines()
    cells = []
    for title, (dataset, endpoint) in zip(TITLES, [(d, e) for d, pairs in SPECS.items() for e, _ in pairs]):
        line = next(line for line in lines if line.startswith(title + ' &'))
        entries = line.split('&')[1:]
        for method, entry in zip(METHODS, entries):
            clean = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', entry).replace('$-$', '-').replace('\\\\', '').strip()
            expected = []
            for pct in [1, 10]:
                row = b.loc[(dataset, endpoint, method, pct)]
                star = '*' if row.ci_low > 0 or row.ci_high < 0 else ''
                expected.append(f'{100*row.error_reduction:.2f}{star}')
            actual = ' / '.join(expected)
            cells.append(dict(endpoint=title, method=method, printed=clean, reproduced=actual, exact_match=clean == actual))
    pd.DataFrame(cells).to_csv(OUT/'table3_printed_parity.csv', index=False)
    assert len(cells) == 56 and all(c['exact_match'] for c in cells)
    dump(OUT/'gate.json', dict(passed=True, rows=112, printed_cells=56, printed_values_and_stars_exact=True, fills_bitwise_equal=True, numeric_tolerance=2e-14, maximum_numeric_difference=float(np.nanmax(delta))))
    print((OUT/'gate.json').read_text(), flush=True)

def finish():
    assert json.loads((OUT/'gate.json').read_text())['passed']
    values = collect('grid', [0.25, 0.5, 2, 3, 5])
    gate_values = pd.read_csv(OUT/'gate_unit_values.csv', dtype={'unit': str})
    values = pd.concat([gate_values, values[values.matrix != 'corrupted_raw']], ignore_index=True)
    values.to_csv(OUT/'unit_values.csv', index=False)
    result = summarize(values, PCTS)
    assert len(result) == 392 and not result.duplicated(['dataset', 'endpoint', 'method', 'fill_pct']).any()
    assert np.isfinite(result[['error_reduction', 'ci_low', 'ci_high']]).all().all()
    for col in ['error_reduction', 'ci_low', 'ci_high']:
        result[col + '_x100'] = 100*result[col]
    result['fill_fraction'] = result.fill_pct/100
    result['method_label'] = result.method.map(dict(zip(METHODS, LABELS)))
    result.to_csv(OUT/'fill_grid_long.csv', index=False)

    tree = ast.parse(Path('scripts/plot_overview_f1.py').read_text())
    styles = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STYLES' for t in n.targets)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = [styles[n]['color'] for n in ['Safe Fusion (transductive)', 'Safe Fusion (transductive)', 'SVD', 'MAGIC', 'scVI', 'ALRA', 'scVI', 'MAGIC']]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.1))
    summaries = []
    for ax, title, (dataset, endpoint) in zip(axes.flat, TITLES, [(d, e) for d, pairs in SPECS.items() for e, _ in pairs]):
        subset = result[(result.dataset == dataset) & (result.endpoint == endpoint)]
        ax.axhline(0, color='#777777', linewidth=0.7)
        for i, method in enumerate(METHODS):
            data = subset[subset.method == method].sort_values('fill_pct')
            style = '--' if i in [1, 6, 7] else '-'
            ax.plot(data.fill_pct, data.error_reduction_x100, color=colors[i], linestyle=style, linewidth=1.5 if i < 2 else 1, label=LABELS[i], zorder=10-i)
            if i < 2:
                ax.fill_between(data.fill_pct, data.ci_low_x100, data.ci_high_x100, color=colors[i], alpha=0.12 if i == 0 else 0.07, linewidth=0)
        ax.set_xscale('log'); ax.set_xticks(PCTS); ax.set_xticklabels([f'{p:g}' for p in PCTS])
        ax.set_title(title, fontsize=10); ax.set_xlabel('Fill fraction (%)', fontsize=9)
        ax.set_ylabel('Error reduction ×100', fontsize=9); ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
        sf = subset[subset.method == 'detection'].sort_values('fill_pct')
        best = sf.loc[sf.error_reduction.idxmax()]
        pos = sf[sf.error_reduction > 0]; detectable = sf[sf.ci_low > 0]
        summaries.append(dict(endpoint=title, best_fill_pct=best.fill_pct, best_error_reduction_x100=best.error_reduction_x100, ci_low_x100=best.ci_low_x100, ci_high_x100=best.ci_high_x100, largest_positive_fill_pct=pos.fill_pct.max() if len(pos) else None, largest_ci_above_zero_fill_pct=detectable.fill_pct.max() if len(detectable) else None))
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'fill_grid.png', dpi=240)
    fig.savefig(OUT/'fill_grid.pdf', dpi=240)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(OUT/'safe_fusion_summary.csv', index=False)
    dump(OUT/'completion.json', dict(rows=392, endpoints=7, methods=8, fractions=PCTS, bootstrap=2000, seed=1729, slurm_job=os.environ['SLURM_JOB_ID']))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['gate-unit', 'gate', 'grid-unit', 'summary'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    if args.stage.endswith('-unit'):
        unit(args.stage.split('-')[0], args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif args.stage == 'gate':
        gate()
    else:
        finish()
