
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
OUT = ROOT / 'artifacts/paper_evidence/review_round4/main_tables_v2'
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
                       observed=float(observed), expected=float(expected),
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



subprocess.run([sys.executable,str(ROOT/'scripts/analyses/main_tables_v2/prepare_enimpute_counts.py')],check=True)
replicates = read(OUT/'enimpute_unit_counts.parquet')
SOURCES.update(js(OUT/'enimpute_input_sha256.json'))
for dataset in ['Pancreas', 'Colon', 'CRISPRa']:
    base = R4 / 'transductive_comparators/evaluate' / dataset
    saved = load(base / 'bootstrap_statistics.npz')
    counts = read(base / 'unit_counts.parquet')
    main = read(R4 / 'transductive_main/masked' / dataset / 'unit_counts.parquet')
    n_units = len(counts.unit.unique())
    for method, key in [('Safe Fusion','Safe Fusion (transductive)'), ('ALRA','ALRA (transductive)')]:
        values = 100 * saved[key + '|mean_f1_1_to_10']
        observed, samples, units = f1(counts, key)
        np.testing.assert_allclose(values, np.r_[observed, samples], atol=1e-12, rtol=0)
        result('Table 1', method, dataset, values[0], values[1:], n_units,
               str((base / 'bootstrap_statistics.npz').relative_to(ROOT)), True)
    for method in ['EnImpute','scVI','MAGIC']:
        data = replicates[replicates.dataset == dataset] if method == 'EnImpute' else main
        observed, samples, units = f1(data, method)
        assert len(units) == n_units
        if dataset == 'CRISPRa':
            assert len(units) == 66 and 'norman_crispra:none' in units
        result('Table 1', method, dataset, observed, samples, n_units,
               'main_tables_v2/enimpute_unit_counts.parquet (original saved EnImpute contracts)' if method == 'EnImpute'
               else f'transductive_main/masked/{dataset}/unit_counts.parquet')
        estimates = read(R4 / 'transductive_main/masked' / dataset / ('new_comparators_absolute.csv' if method == 'EnImpute' else 'absolute.csv'))
        old = estimates[(estimates.method == method) & (estimates.statistic == 'mean_f1_1_to_10')].iloc[0]
        np.testing.assert_allclose([observed,*np.quantile(samples,[.025,.975])], old[['estimate_percent','lower_percent','upper_percent']].to_numpy(float), atol=5.1e-5, rtol=0)
    sf = next(x['mean'] for x in ROWS if x['table']=='Table 1' and x['column']==dataset and x['method']=='Safe Fusion')
    for method in ['EnImpute','scVI','MAGIC','ALRA']:
        value = next(x['mean'] for x in ROWS if x['table']=='Table 1' and x['column']==dataset and x['method']==method)
        check('Table 1', method, dataset, sf-value, point(paper_cells('tab:s_masked_full',method)[['Pancreas','Colon','CRISPRa'].index(dataset)]), kind='Safe Fusion minus method')
print('Table 1 complete', flush=True)


A = R4 / 'round2_extras/part_a'
methods = ['Safe Fusion, labels','Safe Fusion','scVI','MAGIC','EnImpute','Weighted kNN, labels','Per-label expected count']
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
        check('Table 2 / supplementary',method,column,values.mean(),point(paper_cells('tab:s_knockdown_full',method)[paper_column]))

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
        check('Table 2 / supplementary',method,column,values[0],point(paper_cells('tab:s_knockdown_full',method)[paper_column]))
for method in methods[:5]:
    for column in ['CRISPRi','CRISPRa','Pancreas sex','Colon sex']:
        result('Table 2',method,column,*auc[method,column])
for method in [methods[0],*methods[5:]]:
    for column in ['CRISPRi','CRISPRa','Pancreas sex','Colon sex']:
        result('Supplementary',method,column,*auc[method,column])
print('AUROC complete', flush=True)


for dataset,key in [('Pancreas','Pancreas'),('Colon','Colon'),('CRISPRa','norman_crispra')]:
    base = R4/'label_baselines/masked'/key
    report = js(base/'report.json'); saved = load(base/'bootstrap_draws.npz')
    assert report['draws'] == DRAWS and report['seed'] == SEED
    n = report['n_units']
    np.testing.assert_array_equal(saved['draws'], np.random.default_rng(SEED).integers(0,n,size=(DRAWS,n)))
    for method,slug in [('Safe Fusion, labels','safe_fusion_labels'),('Weighted kNN, labels','weighted_knn_labels'),('Per-label expected count','label_detection')]:
        if method=='Per-label expected count' and dataset!='CRISPRa':
            path=R4/'label_baselines/masked/supplementary'/dataset
            sup = js(path/'report.json'); arr=load(path/'bootstrap_draws.npz')
            observed=sup['mean_masked_f1'][0]; samples=arr['f1']; expected=sup['mean_masked_f1']
            np.testing.assert_array_equal(arr['draws'],saved['draws'])
        else:
            path=base

            if method=='Per-label expected count':
                slug=next(k for k in report['mean_masked_f1'] if k not in ['safe_fusion_labels','weighted_knn_labels','scvi_labels'])
            expected=report['mean_masked_f1'][slug]; observed=expected[0]; samples=saved['f1_'+slug]
        np.testing.assert_allclose(np.quantile(samples,[.025,.975]),expected[1:],atol=1e-12,rtol=0)
        result('Supplementary',method,dataset+' F1',100*observed,100*samples,report['n_units'],str((path/'bootstrap_draws.npz').relative_to(ROOT)),True)
    sf=next(r['mean'] for r in ROWS if r['table']=='Supplementary' and r['method']=='Safe Fusion, labels' and r['column']==dataset+' F1')
    for method,expected in [('Weighted kNN, labels',{'Pancreas':11.96,'Colon':7.71,'CRISPRa':1.01}[dataset]),('Per-label expected count',{'Pancreas':19.55,'Colon':11.97,'CRISPRa':.72}[dataset])]:
        other=next(r['mean'] for r in ROWS if r['table']=='Supplementary' and r['method']==method and r['column']==dataset+' F1')
        check('S6',method,dataset,sf-other,expected,kind='Safe Fusion with labels minus method')
print('Supplementary F1 complete',flush=True)


unit_draws=original_function('scripts/r3_downstream_thinning.py','unit_draws')
old=read(R4/'transductive_downstream/summary/thinning_unit_values.csv')
extra=read(R4/'reviewer_extras/B_thinning/unit_values.csv')
values=pd.concat([old,extra[extra.matrix=='alra_1pct']],ignore_index=True)
expected=read(R4/'reviewer_extras/B_thinning/all_rows.csv')
endpoints=[('colon','Marker AUPRC','Colon marker AUPRC'),('colon','Reference mapping F1','Colon mapping F1'),('pancreas','Marker AUPRC','Pancreas marker AUPRC'),('pancreas','Reference mapping F1','Pancreas mapping F1'),('zebrafish','Dynamics','Zebrafish dynamics'),('zebrafish','Stage error','Zebrafish stage error'),('norman_crispra','Edge AUPRC','Norman edge AUPRC')]
for dataset,endpoint,column in endpoints:
    frame=values[(values.dataset==dataset)&(values.endpoint==endpoint)].copy()
    frame.unit=frame.unit.astype(str)
    pivot=frame.pivot(index='unit',columns='matrix',values='value')
    units=pivot.index
    metadata=frame.drop_duplicates('unit').set_index('unit').loc[units]
    reference=metadata.unthinned.to_numpy(); raw=pivot.corrupted_raw.to_numpy()
    draws=unit_draws(metadata.stratum.to_numpy(),DRAWS,SEED)
    for method,slug,paper_column in [('Safe Fusion, weighted','detection',0),('SVD','svd',2),('MAGIC','magic',3),('scVI','scvi',4),('ALRA','alra',5)]:
        reduction=np.abs(raw-reference)-np.abs(pivot[slug+'_1pct'].to_numpy()-reference)
        samples=100*np.nanmean(reduction[draws],axis=1); observed=100*np.nanmean(reduction)
        saved=expected[(expected.dataset==dataset)&(expected.endpoint==endpoint)&(expected.method==slug)&(expected.fill_pct==1)].iloc[0]
        np.testing.assert_allclose([observed,*np.quantile(samples,[.025,.975])],100*saved[['error_reduction','ci_low','ci_high']].to_numpy(float),atol=1e-12,rtol=0)
        row=result('Supplementary Table S25',method,column,observed,samples,np.isfinite(reduction).sum(),
                   'reviewer_extras/B_thinning/unit_values.csv' if slug=='alra' else 'transductive_downstream/summary/thinning_unit_values.csv')
        cell=f'{100*saved.error_reduction:.2f}' + ('*' if saved.ci_low>0 or saved.ci_high<0 else '')
        check('Supplementary Table S25',method,column,observed,point(cell),kind='prerequisite error reduction, before gap normalization')
        assert row['excludes_zero']==('*' in cell),(method,column,cell)
print('Supplementary Table S25 complete',flush=True)

for p in ['docs/PAPER_EXPERIMENTS.md','docs/REPRODUCTION.md','scripts/paired_masked_f1_bootstrap.py','scripts/fusion_value_bootstrap.py',
          'scripts/masked_f1_units.py','scripts/compute_matched_baseline_f1_curves.py',str(R4/'round2_extras/part_c/units.json'),
          'scripts/standard_imputers/evaluate_r3_masked.py','scripts/disease_control_common.py',
          str(TEX),
          str(ROOT/'scripts/analyses/transductive_comparators/workflow.py'),str(ROOT/'scripts/analyses/round2_extras/part_c/evaluate.py'),
          str(ROOT/'scripts/analyses/round2_extras/part_a/analyze.py'),str(ROOT/'scripts/analyses/transductive_downstream/summarize.py'),
          str(ROOT/'scripts/analyses/reviewer_extras/B_thinning/run.py')]:
    source(p)
frame=pd.DataFrame(ROWS); frame.to_csv(OUT/'estimates.csv',index=False)
pd.DataFrame(CHECKS).to_csv(OUT/'point_checks.csv',index=False)
np.savez_compressed(OUT/'bootstrap_quantities.npz',**BOOT)
(OUT/'source_sha256.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
(OUT/'verification.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in CHECKS),point_checks=len(CHECKS),table3_interval_flags_checked=35,estimates=len(ROWS),draws=DRAWS,seed=SEED,se_ddof=1,no_refits=True),indent=2)+'\n')
print('All checks passed',len(CHECKS),'estimates',len(ROWS),flush=True)
