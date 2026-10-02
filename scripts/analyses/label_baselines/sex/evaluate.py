import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
import disease_control_sex_zeros as sex
import evaluate_knockdown_zero_analyses as kd
from disease_control_common import donor_sex, stratified_draws, interval, dense

OUT = ROOT / 'artifacts/paper_evidence/review_round4/label_baselines/sex'
REF = ROOT / 'artifacts/paper_evidence/review_round4/transductive_references/sex_zeros'
SAVED = ROOT / 'artifacts/paper_evidence/review_round4/sex_zero_comparators/reproduction_run/evaluation'
SF = ['safe_fusion', 'safe_fusion_donor', 'safe_fusion_inductive']
NAMES = {'safe_fusion_donor': 'Safe Fusion, donor labels', 'safe_fusion': 'Safe Fusion', 'safe_fusion_inductive': 'Safe Fusion, inductive'}


def original_summary(table, methods, donors_frame):
    donors = sorted(donors_frame.donor)
    draws = stratified_draws(donors_frame.set_index('donor').loc[donors, 'sex'].to_numpy(), 2000, 1729)
    _, rows = sex.summarize(table.reset_index(drop=True), methods, ['recorded zero, expressing sex'], draws, donors, 'recorded counts')
    return pd.DataFrame(rows), draws


def verify(tissue, out):
    table = pd.read_parquet(SAVED / tissue / 'zero_scores.parquet')
    expected = pd.read_csv(SAVED / tissue / 'auroc.csv')
    donors = pd.read_csv(SAVED / tissue / 'donor_sex.csv', dtype={'donor': str})
    table['donor'] = table.donor.astype(str)
    for method in SF:
        table[f'score:{method}'] = np.nan

        for fraction in sex.FRACTIONS:
            table[f'filled:{method}:{fraction}'] = False
        for unit, ix in table.groupby('unit').groups.items():
            path = REF / ('inductive_rebuilt' if method == 'safe_fusion_inductive' else 'deployment_rebuilt') / unit / ('selector_donor' if method == 'safe_fusion_donor' else 'selector')
            s = pd.read_parquet(path / 'selected_gene_scores.parquet')
            s = s[s.split == 'test'].set_index(['cell_index', 'gene_index']).selector_score
            table.loc[ix, f'score:{method}'] = s.loc[list(zip(table.loc[ix, 'cell_index'], table.loc[ix, 'gene_index']))].to_numpy()
    methods = list(expected.method.unique())
    assert all(table[f'score:{m}'].notna().all() for m in methods)
    actual, draws = original_summary(table, methods, donors)
    keys = ['gene', 'method']
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=0)
    paper = (Path(__import__('os').environ['SAFE_FUSION_MANUSCRIPT_DIR']) / 'supplementary_results.tex').read_text().split('\\label{tab:s_sex}', 1)[1].split('\\bottomrule', 1)[0]
    checks = []
    for (gene, method), row in actual.iterrows():
        name = NAMES.get(method, method)
        line = next(line for line in paper.splitlines() if line.startswith(name + ' &'))
        cells = line.split(' & ')[1:]
        col = (0 if tissue == 'pancreas' else 2) + (gene == 'XIST')
        printed = re.findall(r'\d+\.\d+', cells[col])
        calculated = [f'{row[k]:.2f}' for k in ['auroc', 'ci_low', 'ci_high']]
        assert printed == calculated, (tissue, gene, method, printed, calculated)
        checks.append({'tissue': tissue, 'gene': gene, 'method': method, 'paper': printed, 'reproduced': calculated})
    actual.reset_index().to_csv(out / 's18_reproduced.csv', index=False)
    table.to_parquet(out / 'zero_scores.parquet', index=False)
    donors.to_csv(out / 'donor_sex.csv', index=False)
    np.save(out / 'bootstrap_draws.npy', draws)
    (out / 'gate.json').write_text(json.dumps({'passed': True, 'rows': len(actual), 'numeric_atol': 1e-12, 'numeric_rtol': 0, 'draws': 2000, 'seed': 1729, 'paper_checks': checks}, indent=2))
    print(f'{tissue}: S18 verified {len(actual)} rows', flush=True)


def bootstrap_auc(labels, scores, codes, weights, bins):

    numer = np.zeros(len(weights)); denom = np.zeros(len(weights))
    for level in np.unique(bins):
        members = np.flatnonzero(bins == level)
        order = members[np.argsort(scores[members], kind='stable')]
        _, starts = np.unique(scores[order], return_index=True)
        w = weights[:, codes[order]]
        p = np.add.reduceat(w * labels[order], starts, axis=1)
        n = np.add.reduceat(w * (1 - labels[order]), starts, axis=1)
        numer += np.sum(p * (np.cumsum(n, axis=1) - 0.5 * n), axis=1)
        denom += p.sum(axis=1) * n.sum(axis=1)
    return np.divide(numer, denom, out=np.full(len(weights), np.nan), where=denom > 0)


def ranking_summary(table, methods, donors_frame, scenario):
    donors = sorted(donors_frame.donor)
    draws = stratified_draws(donors_frame.set_index('donor').loc[donors, 'sex'].to_numpy(), 2000, 1729)
    weights = np.vstack([np.ones(len(donors)), [np.bincount(d, minlength=len(donors)) for d in draws]])
    rows = []
    for gene, frame in table.groupby('gene'):
        labels = (frame.kind == 'recorded zero, expressing sex').to_numpy().astype(int)
        codes = pd.Categorical(frame.donor, categories=donors).codes
        assert (codes >= 0).all() and len(np.unique(labels)) == 2
        library = frame.library_size.to_numpy()
        assert (library > 0).all()
        log_library = np.log(library)
        bins = np.searchsorted(np.quantile(log_library, np.linspace(0, 1, 6)[1:-1]), log_library, side='right')
        for adjustment, strata in [('none', np.zeros(len(frame), dtype=int)), ('library_quintiles', bins)]:
            samples = {}
            for method in methods:
                score = frame[f'score:{method}'].to_numpy()
                assert np.isfinite(score).all()
                values = bootstrap_auc(labels, score, codes, weights, strata)
                point = roc_auc_score(labels, score) if adjustment == 'none' else kd.stratified_auroc(labels, score, log_library, 5)[0]
                np.testing.assert_allclose(values[0], point, atol=1e-12, rtol=0)
                if adjustment == 'none':
                    for j in range(1, 4):
                        w = weights[j, codes]; keep = w > 0
                        if len(np.unique(labels[keep])) == 2:
                            np.testing.assert_allclose(values[j], roc_auc_score(labels[keep], score[keep], sample_weight=w[keep]), atol=1e-12, rtol=0)
                samples[method] = values
                low, high = interval(values[1:])
                rows.append(dict(scenario=scenario, gene=gene, method=method, adjustment=adjustment, auroc=point, ci_low=low, ci_high=high, n_positive=int(labels.sum()), n_negative=int((1-labels).sum()), n_donors=len(donors), finite_draws=int(np.isfinite(values[1:]).sum())))
            if 'safe_fusion_donor' in methods:
                for method in methods:
                    if method == 'safe_fusion_donor': continue
                    values = samples['safe_fusion_donor'] - samples[method]
                    low, high = interval(values[1:])
                    rows.append(dict(scenario=scenario, gene=gene, method='safe_fusion_donor minus ' + method, adjustment=adjustment, auroc=values[0], ci_low=low, ci_high=high, n_positive=int(labels.sum()), n_negative=int((1-labels).sum()), n_donors=len(donors), finite_draws=int(np.isfinite(values[1:]).sum())))
    return pd.DataFrame(rows), draws


def analyze(tissue, out):
    assert json.loads((OUT.parent / 'verification_passed.json').read_text())['passed']
    assert json.loads((out / 'gate.json').read_text())['passed']
    table = pd.read_parquet(out / 'zero_scores.parquet')
    donors = pd.read_csv(out / 'donor_sex.csv', dtype={'donor': str})
    audit = []
    for unit, ix in table.groupby('unit').groups.items():
        base = REF / 'deployment_rebuilt' / unit
        a = ad.read_h5ad(base / 'hybrid.h5ad')
        split = pd.read_parquet(base / 'splits.parquet').set_index('cell_id').loc[a.obs_names, 'split'].to_numpy()

        fit_donors = set(a.obs.donor.astype(str).to_numpy()[split != 'test'])
        test_donors = set(a.obs.donor.astype(str).to_numpy()[split == 'test'])
        assert not (fit_donors & test_donors)
        for method in ['graph_smooth_donor', 'scvi_donor']:
            path = base / method
            metadata = json.loads((path / 'metadata.json').read_text())
            assert metadata['cell_ids'] == list(a.obs_names)
            assert metadata['gene_ids'] == list(a.var_names)
            assert metadata['scale'] == 'counts' and metadata['seed'] == 1729
            values = np.load(path / 'mean.npy', mmap_mode='r')
            table.loc[ix, f'score:{method}'] = np.maximum(values[table.loc[ix, 'cell_index'], table.loc[ix, 'gene_index']], 0)
            audit.append({'unit': unit, 'method': method, 'path': str(path.relative_to(ROOT)), 'scale': metadata['scale'], 'parameters': metadata['parameters']})
        audit.append({'unit': unit, 'method': 'per_label_expected_count', 'status': 'unavailable', 'reason': 'Every test donor is absent from fitting cells; a_g,l is undefined. No all-cell estimate or label-agnostic fallback substituted.', 'test_donors': sorted(test_donors), 'fitting_donors': sorted(fit_donors)})
    results, _ = ranking_summary(table, ['safe_fusion_donor', 'graph_smooth_donor', 'scvi_donor'], donors, 'expression_sex')
    results.to_csv(out / 'label_rankings_auroc.csv', index=False)
    table.to_parquet(out / 'label_zero_scores.parquet', index=False)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    if tissue == 'colon':
        metadata = donor_sex(tissue)
        donors['metadata_sex'] = donors.donor.map(metadata)
        assert donors.metadata_sex.isin(['female', 'male']).all()
        donors['agrees'] = donors.sex == donors.metadata_sex
        assert donors.agrees.sum() == 25 and len(donors) == 34
        donors.to_csv(out / 'metadata_sex_audit.csv', index=False)
        donors[~donors.agrees].to_csv(out / 'excluded_donors.csv', index=False)
        methods = list(pd.read_csv(SAVED / tissue / 'auroc.csv').method.unique())
        for scenario in ['agreed25', 'metadata34']:
            d = donors[donors.agrees].copy() if scenario == 'agreed25' else donors.copy()
            if scenario == 'metadata34': d['sex'] = d.metadata_sex
            t = table[table.donor.isin(d.donor)].copy().reset_index(drop=True)
            t['sex'] = t.donor.map(d.set_index('donor').sex)
            t['kind'] = np.where(t.sex == t.gene.map(sex.SEX_GENES), 'recorded zero, expressing sex', 'absent, other sex')
            result, draws = original_summary(t, methods, d)
            result.to_csv(out / f'{scenario}_auroc.csv', index=False)
            np.save(out / f'{scenario}_bootstrap_draws.npy', draws)
            print(f'{tissue}: {scenario} {len(result)} rows', flush=True)
    print(f'{tissue}: analysis complete', flush=True)


def transductive(tissue, out):

    assert json.loads((OUT.parent / 'verification_passed.json').read_text())['passed']
    table = pd.read_parquet(out / 'label_zero_scores.parquet')
    donors = pd.read_csv(out / 'donor_sex.csv', dtype={'donor': str})
    audit = []
    for unit, ix in table.groupby('unit').groups.items():
        base = REF / 'deployment_rebuilt' / unit
        a = ad.read_h5ad(base / 'hybrid.h5ad')
        counts = dense(a.layers['corrupted_counts']).astype(np.float64)
        library = counts.sum(axis=1)
        labels = a.obs.donor.astype(str).to_numpy()
        rows = table.loc[ix, 'cell_index'].to_numpy()
        cols = table.loc[ix, 'gene_index'].to_numpy()
        np.testing.assert_array_equal(library[rows], table.loc[ix, 'library_size'].to_numpy())
        assert (counts[rows, cols] == 0).all()
        score = np.full(len(ix), np.nan)
        for donor in np.unique(labels[rows]):
            fitting = labels == donor
            totals = counts[fitting].sum(axis=0)
            assert totals.sum() > 0
            share = totals / totals.sum()
            selected = labels[rows] == donor
            score[selected] = -np.expm1(-library[rows[selected]] * share[cols[selected]])
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
        table.loc[ix, 'score:transductive_expected_count'] = score.astype(np.float32)
        for method in ['graph_smooth_donor', 'scvi_donor']:
            m = json.loads((base / method / 'metadata.json').read_text())
            assert m['parameters']['test_used_for_fit'] and m['parameters']['fit_cells'] == len(a)
        audit.append({'unit': unit, 'input': str((base / 'hybrid.h5ad').relative_to(ROOT)), 'fitting_population': 'all observed hybrid cells, matching frozen donor teachers', 'fit_cells': len(a), 'fit_donors': len(np.unique(labels)), 'formula': '-expm1(-library_size * donor_gene_counts / donor_total_counts)', 'precision': 'float64 accumulation and probability, cast to float32 for ranking consistently with other expected-count endpoints', 'strict_selector_fitting_baseline': 'remains unavailable for held-out donors'})
    result, _ = ranking_summary(table, ['safe_fusion_donor', 'transductive_expected_count'], donors, 'expression_sex_transductive_fitting')
    result.to_csv(out / 'transductive_expected_count_auroc.csv', index=False)
    table.to_parquet(out / 'transductive_expected_count_scores.parquet', index=False)
    (out / 'transductive_expected_count_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'{tissue}: supplementary transductive expected-count analysis complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', required=True, choices=['colon', 'pancreas'])
    parser.add_argument('--stage', required=True, choices=['verify', 'analyze', 'transductive'])
    args = parser.parse_args()
    out = OUT / args.tissue
    out.mkdir(parents=True, exist_ok=True)
    {'verify': verify, 'analyze': analyze, 'transductive': transductive}[args.stage](args.tissue, out)
