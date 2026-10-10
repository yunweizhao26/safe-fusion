
import ast
import os
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/paper_evidence/review_round4/main_tables_v3'
R4 = OUT.parent
TEX = Path(os.environ['SAFE_FUSION_MANUSCRIPT_DIR']) / 'supplementary_results.tex'
SOURCES = {}
ROWS, CHECKS, BOOT = [], [], {}
DRAWS, SEED = 2000, 1729
FRACTIONS = np.arange(1, 11) / 100


def source(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if key not in SOURCES:
        SOURCES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    path = source(path)
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def load(path):
    return np.load(source(path), allow_pickle=False)


def js(path):
    return json.loads(source(path).read_text())


def original_function(path, name):

    tree = ast.parse(source(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'np': np, 'DRAWS': DRAWS, 'SEED': SEED}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def check(table, method, column, observed, expected, decimals=2, kind='point'):
    actual = f'{observed:.{decimals}f}'
    wanted = f'{float(expected):.{decimals}f}'
    CHECKS.append(dict(table=table, method=method, column=column, kind=kind,
                       observed=float(observed), expected=float(expected), decimals=decimals,
                       observed_printed=actual, expected_printed=wanted, passed=actual == wanted))
    assert actual == wanted, CHECKS[-1]


def result(table, method, column, point, samples, n_units, provenance, saved=False):
    samples = np.asarray(samples, dtype=float)
    assert samples.shape == (DRAWS,) and np.isfinite(samples).all()
    low, high = np.quantile(samples, [.025, .975])
    record = dict(table=table, method=method, column=column, mean=float(point),
                  bootstrap_se=float(np.std(samples, ddof=1)), ci_low=float(low), ci_high=float(high),
                  excludes_zero=bool(low > 0 or high < 0), n_units=int(n_units), draws=DRAWS, seed=SEED,
                  saved_statistic_draws=saved, source=provenance)
    ROWS.append(record)
    BOOT[f'{table}|{method}|{column}'] = samples
    return record


def f1(frame, method, draws=None):
    part = frame[(frame.method == method) & frame.fraction.isin(FRACTIONS)]
    units = sorted(part.unit.unique())
    arrays = [part.pivot(index='unit', columns='fraction', values=c).loc[units, FRACTIONS].to_numpy()
              for c in ['n_selected', 'n_true_positive', 'n_masked_positives']]
    assert all(np.isfinite(x).all() for x in arrays)
    selected, hits, positive = arrays
    if draws is None:
        draws = np.random.default_rng(SEED).integers(0, len(units), size=(DRAWS, len(units)))
    observed = np.mean(200 * hits.sum(0) / (selected.sum(0) + positive.sum(0)))
    samples = np.mean(200 * hits[draws].sum(1) / (selected[draws].sum(1) + positive[draws].sum(1)), axis=1)
    return observed, samples, units


paper = source(TEX).read_text()
def paper_cells(label, row):
    block = paper.split('\\label{' + label + '}', 1)[1].split('\\bottomrule', 1)[0]
    line = next(x for x in block.splitlines() if x.startswith(row + ' &'))
    return line.replace('$-$', '-').split(' & ')[1:]


def point(cell):
    return float(re.search(r'-?\d+(?:\.\d+)?', cell).group())



V2 = R4/'main_tables_v2'
previous = read(V2/'estimates.csv')
previous_draws = load(V2/'bootstrap_quantities.npz')
for row in previous[previous.table == 'Table 1'].to_dict('records'):
    key = '|'.join([row['table'], row['method'], row['column']])
    result(row['table'], row['method'], row['column'], row['mean'], previous_draws[key],
           row['n_units'], 'main_tables_v2/estimates.csv + bootstrap_quantities.npz', True)
added = read(OUT/'added_unit_counts.parquet')
SOURCES.update(js(OUT/'added_input_sha256.json'))
for dataset in ['Pancreas', 'Colon', 'CRISPRa']:
    base = R4/'transductive_comparators/evaluate'/dataset
    counts = read(base/'unit_counts.parquet')
    main = read(R4/'transductive_main/masked'/dataset/'unit_counts.parquet')
    ref_point, ref_samples, ref_units = f1(counts, 'Safe Fusion (transductive)')
    np.testing.assert_allclose(np.r_[ref_point, ref_samples], np.r_[
        previous[(previous.table=='Table 1') & (previous.method=='Safe Fusion') & (previous.column==dataset)].iloc[0]['mean'],
        previous_draws['Table 1|Safe Fusion|'+dataset]], atol=1e-12, rtol=0)
    for method in ['SVD', 'Weighted kNN', 'SAVER', 'DCA', 'scImpute', 'scRecover', 'scGPT']:
        if method in ['SVD', 'Weighted kNN']:
            data, key = counts, method+' (transductive)'
            provenance = str((base/'unit_counts.parquet').relative_to(ROOT))
            expected_file = base/'absolute.csv'
        elif method in ['SAVER', 'scGPT']:
            data, key = main, method
            provenance = f'transductive_main/masked/{dataset}/unit_counts.parquet'
            expected_file = R4/'transductive_main/masked'/dataset/'absolute.csv'
        else:
            data, key = added[added.dataset == dataset], method
            provenance = 'main_tables_v3/added_unit_counts.parquet (frozen round3 comparison contracts)'
            expected_file = R4/'transductive_main/masked'/dataset/'new_comparators_absolute.csv'
        observed, samples, units = f1(data, key)
        assert units == ref_units, (dataset, method)
        if dataset == 'CRISPRa':
            assert len(units) == 66 and 'norman_crispra:none' in units
        expected = read(expected_file)
        saved = expected[(expected.method == key) & (expected.statistic == 'mean_f1_1_to_10')].iloc[0]
        np.testing.assert_allclose([observed, *np.quantile(samples, [.025,.975])],
                                  saved[['estimate_percent','lower_percent','upper_percent']].to_numpy(float),
                                  atol=5.1e-5, rtol=0)
        result('Table 1', method, dataset, observed, samples, len(units), provenance)
    sf = next(r['mean'] for r in ROWS if r['table']=='Table 1' and r['method']=='Safe Fusion' and r['column']==dataset)
    for row in [r for r in ROWS if r['table']=='Table 1' and r['column']==dataset and r['method']!='Safe Fusion']:
        method = row['method']
        check('Table 1', method, dataset, sf-row['mean'],
              point(paper_cells('tab:s_masked_full', method)[['Pancreas','Colon','CRISPRa'].index(dataset)]),
              decimals=1 if method in ['scRecover','scGPT'] else 2, kind='S4 Safe Fusion minus method')
print('Table 1 complete', flush=True)


A = R4 / 'round2_extras/part_a'
methods = ['Safe Fusion, labels','Safe Fusion','scVI','MAGIC','EnImpute','SVD','Weighted kNN','ALRA','SAVER','DCA','scImpute']
auc = {}
for dataset, column, paper_column in [('adamson_crispri','CRISPRi',0),('norman_crispra','CRISPRa',2)]:
    frame = read(A / f'{dataset}_per_target.csv')
    frame = frame[(frame.adjustment == 'none') & (frame.score_type == 'continuous')]
    genes = frame.gene.drop_duplicates().tolist()
    draws = load(A / f'{dataset}_draws.npy')
    assert draws.shape == (DRAWS,len(genes))
    scopes = original_scopes = ast.parse(source('scripts/evaluate_knockdown_zero_analyses.py').read_text())
    scope_node = next(n for n in scopes.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCOPES' for t in n.targets))
    screen_tree = ast.parse(source('scripts/evaluate_perturbation_zeros.py').read_text())
    screen_node = next(n for n in screen_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCREENS' for t in n.targets))
    scopes = (*ast.literal_eval(screen_node.value), 'knockdown_pooled')
    np.testing.assert_array_equal(draws,np.random.default_rng([SEED,scopes.index(dataset)]).integers(0,len(genes),size=draws.shape))
    summaries = read(A / f'{dataset}_summary.csv')
    for method in methods:
        values = frame[frame.method==method].set_index('gene').loc[genes].auroc.to_numpy()
        samples = values[draws].mean(1)
        old = summaries[(summaries.method==method)&(summaries.adjustment=='none')&(summaries.score_type=='continuous')].iloc[0]
        np.testing.assert_allclose([values.mean(),*np.quantile(samples,[.025,.975])],[old.estimate,old.lower,old.upper],atol=1e-12,rtol=0)
        auc[method,column] = (values.mean(),samples,len(genes),str((A/f'{dataset}_per_target.csv').relative_to(ROOT)))
        check('Table 2',method,column,values.mean(),point(paper_cells('tab:s_knockdown_full',method)[paper_column]))

bootstrap_auc = original_function(ROOT/'scripts/analyses/label_baselines/sex/evaluate.py','bootstrap_auc')
stratified_draws = original_function('scripts/disease_control_common.py','stratified_draws')
for tissue, column, paper_column in [('pancreas','Pancreas sex',3),('colon','Colon sex',4)]:
    frame = read(A / f'{tissue}_zero_scores.parquet')
    assert set(frame.gene) == {'RPS4Y1'}
    frame.donor = frame.donor.astype(str)
    donor_frame = read(R4/'label_baselines/sex'/tissue/'donor_sex.csv')
    donor_frame.donor = donor_frame.donor.astype(str)
    donors = sorted(donor_frame.donor)
    draws = load(A / f'{tissue}_draws.npy')
    assert draws.shape == (DRAWS,len(donors))
    np.testing.assert_array_equal(draws, stratified_draws(donor_frame.set_index('donor').loc[donors,'sex'].to_numpy(), DRAWS, SEED))
    weights = np.vstack([np.ones(len(donors)),[np.bincount(x,minlength=len(donors)) for x in draws]])
    codes = pd.Categorical(frame.donor,categories=donors).codes
    labels = (frame.kind == 'recorded zero, expressing sex').to_numpy().astype(int)
    assert (codes>=0).all()
    mapping = {'Safe Fusion, labels':'safe_fusion_donor','Safe Fusion':'safe_fusion','Weighted kNN, labels':'graph_smooth_donor','Per-label expected count':'transductive_expected_count'}
    summaries = read(A / f'{tissue}_summary.csv')
    for method in methods:
        scores = frame['score:'+mapping.get(method,method)].to_numpy()
        values = np.concatenate([bootstrap_auc(labels,scores,codes,weights[start:start+100],np.zeros(len(frame),int)) for start in range(0,len(weights),100)])
        old = summaries[(summaries.method==method)&(summaries.score_type=='continuous')].iloc[0]
        np.testing.assert_allclose([values[0],*np.quantile(values[1:],[.025,.975])],[old.estimate,old.lower,old.upper],atol=1e-12,rtol=0)
        auc[method,column] = (values[0],values[1:],len(donors),str((A/f'{tissue}_zero_scores.parquet').relative_to(ROOT)))
        check('Table 2',method,column,values[0],point(paper_cells('tab:s_knockdown_full',method)[paper_column]))
for method in methods:
    for column in ['CRISPRi','CRISPRa','Pancreas sex','Colon sex']:
        result('Table 2',method,column,*auc[method,column])
print('Table 2 complete', flush=True)


retained = previous[previous.table.isin(['Table 1','Table 2'])]
for old in retained.to_dict('records'):
    key = '|'.join([old['table'],old['method'],old['column']])
    row = next(r for r in ROWS if (r['table'],r['method'],r['column']) == (old['table'],old['method'],old['column']))
    np.testing.assert_allclose([row['mean'], row['bootstrap_se']], [old['mean'], old['bootstrap_se']], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(BOOT[key], previous_draws[key])
for dataset, mean, se in [('Pancreas',59.18,.84),('Colon',51.99,1.29),('CRISPRa',26.39,.16)]:
    old = previous[(previous.table == 'Supplementary') & (previous.method == 'Safe Fusion, labels') & (previous.column == dataset + ' F1')].iloc[0]
    samples = previous_draws['Supplementary|Safe Fusion, labels|' + dataset + ' F1']
    result('Table 1', 'Safe Fusion, labels', dataset, old['mean'], samples, old.n_units, old.source)
    check('Table 1', 'Safe Fusion, labels', dataset, old['mean'], mean)
    check('Table 1 SE', 'Safe Fusion, labels', dataset, old.bootstrap_se, se)
assert len(ROWS) == 83 and len(CHECKS) == 83
assert len(BOOT) == len(ROWS)
for p in [ROOT/'scripts/analyses/main_tables_v2/build_tables.py', V2/'README.md', ROOT/'scripts/analyses/main_tables_v3/build_tables.py', ROOT/'scripts/analyses/main_tables_v3/prepare_added_counts.py',
          ROOT/'scripts/analyses/round2_extras/part_a/analyze.py']:
    source(p)
pd.DataFrame(ROWS).to_csv(OUT/'estimates.csv', index=False)
pd.DataFrame(CHECKS).to_csv(OUT/'point_checks.csv', index=False)
np.savez_compressed(OUT/'bootstrap_quantities.npz', **BOOT)
(OUT/'source_sha256.json').write_text(json.dumps(SOURCES, indent=2)+'\n')
(OUT/'verification.json').write_text(json.dumps(dict(passed=True, point_checks=len(CHECKS), estimates=len(ROWS),
    retained_v2_estimates=len(retained), retained_v2_draws_exact=True, draws=DRAWS, seed=SEED, se_ddof=1, no_refits=True), indent=2)+'\n')
