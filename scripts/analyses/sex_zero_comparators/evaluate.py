import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
import disease_control_sex_zeros as sex
from disease_control_common import dense, stratified_draws
from masked_f1_units import COUNT_SCALE, fraction_name
from selector_attribution import exact_topk
from compute_matched_baseline_f1_curves import tie_broken_order

OUT = Path(os.environ.get('SEX_ZERO_OUTPUT', str(ROOT / 'artifacts/paper_evidence/review_round4/sex_zero_comparators'))).resolve()
REF = ROOT / 'artifacts/paper_evidence/review_round4/transductive_references/sex_zeros'
DOWN = ROOT / 'artifacts/paper_evidence/review_round4/transductive_downstream/deployment'
METHODS = {'SVD': 'svd_impute', 'Weighted kNN': 'graph_smooth', 'MAGIC': 'magic', 'scVI': 'scvi',
           'ALRA': 'alra', 'SAVER': 'saver', 'DCA': 'dca', 'scImpute': 'scimpute', 'EnImpute': 'enimpute'}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def contract(method, unit):
    name = METHODS[method]
    if method in ('SVD', 'Weighted kNN'):
        return REF / 'deployment_rebuilt' / unit / name
    if method in ('MAGIC', 'scVI'):
        return DOWN / unit / 'standard' / name
    return OUT / 'fits' / name / unit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', choices=('pancreas', 'colon'), required=True)
    parser.add_argument('--stage', choices=('verify', 'comparators'), required=True)
    args = parser.parse_args()
    units = [f'{args.tissue}_{i}' for i in range(3)]
    saved = REF / 'evaluation' / args.tissue
    donors_frame = pd.read_csv(saved / 'donor_sex.csv', dtype={'donor': str})
    assert not donors_frame.donor.duplicated().any()
    donors = sorted(donors_frame.donor.unique())
    donor_sex = donors_frame.set_index('donor').sex
    assert len(donors) == (24 if args.tissue == 'pancreas' else 34)
    draws = stratified_draws(donor_sex.loc[donors].to_numpy(), 2000, 1729)
    out = OUT / ('verification' if args.stage == 'verify' else 'evaluation') / args.tissue
    out.mkdir(parents=True, exist_ok=True)
    missing = {m: [str(contract(m, u)) for u in units if not (contract(m, u) / 'mean.npy').exists()] for m in METHODS}
    methods = ['safe_fusion_inductive'] if args.stage == 'verify' else [m for m in METHODS if len(missing[m]) < len(units)]
    if args.stage == 'comparators':
        assert json.loads((OUT / 'verification' / args.tissue / 'check.json').read_text())['passed']
    tables, audit, budgets = [], [], []
    reference = ad.read_h5ad(REF / 'deployment_rebuilt' / units[0] / 'recorded.h5ad')
    ref_counts = dense(reference.layers['corrupted_counts'])
    symbols = list(reference.var.feature_name.astype(str))
    for donor, expected in donor_sex.items():
        members = reference.obs.donor.astype(str).to_numpy() == donor
        x, y = (ref_counts[members, symbols.index(g)].sum() for g in ('XIST', 'RPS4Y1'))
        assert expected == ('female' if x > y else 'male')
    for unit in units:
        base = REF / 'deployment_rebuilt' / unit
        a = ad.read_h5ad(base / 'recorded.h5ad')
        hybrid = ad.read_h5ad(base / 'hybrid.h5ad')
        assert list(a.obs_names) == list(reference.obs_names) == list(hybrid.obs_names)
        assert list(a.var_names) == list(hybrid.var_names)
        counts = dense(a.layers['corrupted_counts']).astype(np.float32)
        observed = dense(hybrid.layers['corrupted_counts']).astype(np.float32)
        split = pd.read_parquet(base / 'splits.parquet').set_index('cell_id').loc[a.obs_names, 'split'].to_numpy()
        test = np.flatnonzero(split == 'test')
        np.testing.assert_array_equal(counts[test], observed[test])
        obs = a.obs.copy().reset_index(drop=True)
        obs['donor'] = obs.donor.astype(str)
        obs['sex'] = obs.donor.map(donor_sex)
        assert not obs.iloc[test].sex.isna().any()
        symbols = list(a.var.feature_name.astype(str))
        genes = {g: symbols.index(g) for g in sex.SEX_GENES if g in symbols}
        original = sex.SEX_GENES
        try:
            sex.SEX_GENES = {g: v for g, v in original.items() if g in genes}
            table = sex.zero_table(obs.iloc[test].reset_index(drop=True), counts[test], genes, None)
        finally:
            sex.SEX_GENES = original
        rows, cols = test[table.row.to_numpy()], table.col.to_numpy()
        zero_rows, zero_cols = np.where((counts == 0) & (split == 'test')[:, None])

        flat = zero_rows * counts.shape[1] + zero_cols
        positions = np.searchsorted(flat, rows * counts.shape[1] + cols)
        np.testing.assert_array_equal(flat[positions], rows * counts.shape[1] + cols)
        library = observed.sum(axis=1, dtype=np.float64)
        for method in methods:
            if method == 'safe_fusion_inductive':
                path = REF / 'inductive_rebuilt' / unit / 'selector'
                scores = pd.read_parquet(path / 'selected_gene_scores.parquet')
                scores = scores[scores.split == 'test'].set_index(['cell_index', 'gene_index'])
                table[f'score:{method}'] = scores.loc[list(zip(rows, cols)), 'selector_score'].to_numpy()
                for fraction in sex.FRACTIONS:
                    values = np.load(path / f'safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}' / 'mean.npy', mmap_mode='r')
                    table[f'filled:{method}:{fraction}'] = values[rows, cols] != 0
                continue
            path = contract(method, unit)
            if not (path / 'mean.npy').exists():
                table[f'score:{method}'] = np.nan
                for fraction in sex.FRACTIONS:
                    table[f'filled:{method}:{fraction}'] = False
                continue
            metadata = json.loads((path / 'metadata.json').read_text())
            assert metadata['cell_ids'] == list(a.obs_names)
            assert metadata['gene_ids'] == list(a.var_names)
            input_path = DOWN / unit / 'hybrid.h5ad' if method in ('MAGIC', 'scVI') else base / 'hybrid.h5ad'
            input_sha = sha(input_path)
            if metadata.get('input_sha256'):
                assert metadata['input_sha256'] == input_sha, (method, unit, 'input hash mismatch')
            if method in ('MAGIC', 'scVI'):
                source = ad.read_h5ad(input_path)
                assert list(source.obs_names) == list(hybrid.obs_names)
                assert list(source.var_names) == list(hybrid.var_names)
                np.testing.assert_array_equal(dense(source.layers['corrupted_counts']), observed)
            mean = np.load(path / 'mean.npy', mmap_mode='r')
            assert mean.shape == counts.shape
            values = np.maximum(COUNT_SCALE[metadata['scale']](np.asarray(mean[zero_rows, zero_cols], dtype=np.float64), library[zero_rows]), 0)
            optional = method in ('DCA', 'scImpute', 'EnImpute')
            if not optional:

                values = values.astype(np.float32)
            assert np.isfinite(values).all()
            table[f'score:{method}'] = values[positions]

            order = tie_broken_order(values, 1729) if optional else None
            for fraction in sex.FRACTIONS:
                k = max(1, int(round(fraction * len(values))))
                selected = exact_topk(values, k)
                if optional:
                    selected[:] = False
                    selected[order[:k]] = True
                assert selected.sum() == k
                filled = selected & (values.astype(np.float32) > 0)
                if method in ('MAGIC', 'scVI'):
                    saved_fill = np.load(DOWN / unit / f'{METHODS[method]}_{int(round(100*fraction))}pct' / 'mean.npy', mmap_mode='r')
                    np.testing.assert_array_equal(filled, saved_fill[zero_rows, zero_cols] != 0)
                table[f'filled:{method}:{fraction}'] = filled[positions]
                budgets.append({'unit': unit, 'method': method, 'fraction': fraction, 'n_candidates': len(values), 'n_selected': k, 'n_filled': int(filled.sum())})
            audit.append({'unit': unit, 'method': method, 'contract': str(path.relative_to(ROOT)), 'input': str(input_path.relative_to(ROOT)), 'input_sha256': input_sha, 'scale': metadata['scale'], 'mean_sha256': sha(path / 'mean.npy'), 'parameters': metadata.get('parameters', {})})
        tables.append(table.assign(unit=unit, cell_index=rows, gene_index=cols))
    table = pd.concat(tables, ignore_index=True)
    counts = pd.DataFrame(sex.zero_counts(table, 'recorded counts'))
    pd.testing.assert_frame_equal(counts, pd.read_csv(saved / 'zero_counts.csv'), check_dtype=False, atol=1e-12, rtol=1e-12)
    fill, auc = [], []
    for method in methods:

        complete = [g for g, block in table.groupby('gene') if block[f'score:{method}'].notna().all()]
        available = table[table.gene.isin(complete)].reset_index(drop=True)
        if available.empty:
            continue
        f, a = sex.summarize(available, [method], ['recorded zero, expressing sex'], draws, donors, 'recorded counts')
        fill.extend(f)
        auc.extend(a)
    results = {'auroc.csv': pd.DataFrame(auc), 'fill_rates.csv': pd.DataFrame(fill)}
    if args.stage == 'verify':
        for name, actual in results.items():
            expected = pd.read_csv(saved / name)
            expected = expected[expected.method == 'safe_fusion_inductive'].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)
        (out / 'check.json').write_text(json.dumps({'passed': True, 'donors': len(donors), 'draws': 2000, 'seed': 1729, 'tolerance': 1e-12, 'tables': list(results)}, indent=2))
    else:
        for name in results:
            results[name] = pd.concat([pd.read_csv(saved / name), results[name]], ignore_index=True)
        pd.DataFrame(budgets).to_csv(out / 'global_budgets.csv', index=False)
        (out / 'audit.json').write_text(json.dumps(audit, indent=2))
        (out / 'missing.json').write_text(json.dumps({m: p for m, p in missing.items() if p}, indent=2))
    for name, frame in results.items():
        frame.to_csv(out / name, index=False)
    counts.to_csv(out / 'zero_counts.csv', index=False)
    donors_frame.to_csv(out / 'donor_sex.csv', index=False)
    np.save(out / 'bootstrap_draws.npy', draws)
    table.to_parquet(out / 'zero_scores.parquet', index=False)
    print(json.dumps({'tissue': args.tissue, 'stage': args.stage, 'methods': methods, 'zero_rows': len(table), 'passed': True}), flush=True)

if __name__ == '__main__':
    main()
