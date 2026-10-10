from common import *
import json
import os
import subprocess
from dataclasses import replace
from masked_f1_units import unit_manifest_entry

mode, stage = sys.argv[1:3]
index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
uu = units()
if mode == 'replicate':
    seed, u = 1730 + index // len(uu), uu[index % len(uu)]
    key = f'{u.key}_s{seed}'
else:
    seed, u = 1729, uu[index]
    key = u.key
root = OUT / ('rebuilt_replicates' if mode == 'replicate' else 'recorded') / key
root.mkdir(parents=True, exist_ok=True)
data = root / 'input'
input_path = data / ('corrupted.h5ad' if mode == 'replicate' else 'hybrid.h5ad')
coordinates = data / 'coordinates.parquet'
ind, trans = root / 'inductive', root / 'transductive'
standard = root / 'standard'
old_norman = E / f'review_round2/norman_rebuilt/seed_replicates/seed_{seed}'

def run(python, script, args, done):
    if done.exists():
        return
    command = [str(REPO / python), str(REPO / 'scripts' / script), *map(str, args)]
    with (root / f'commands_{stage}.jsonl').open('a') as log:
        log.write(json.dumps(command)+'\n')
    subprocess.run(command, check=True)
    assert done.exists(), done

def teacher(method, out, is_trans):
    run('.venv/bin/python', 'run_leakage_safe_method.py',
        ['--method', method, '--input', input_path, '--coordinates', coordinates, '--splits', u.splits,
         '--output', out, '--seed', seed, *(['--transductive'] if is_trans else [])], out/'metadata.json')

def neural(method, is_trans):
    out = standard / method if is_trans else ind / f'{method}_inductive'
    python = f'.conda-{method}-current/bin/python'
    if is_trans:
        args = ['--corrupted', input_path, '--coordinates', coordinates, '--splits', u.splits, '--output', out, '--seed', seed]
        args += ['--epochs', 200] if method == 'scvi' else ['--n-jobs', 8]
        run(python, f'run_{method}_baseline.py', args, out/'metadata.json')
    else:
        args = ['--method', method, '--input', input_path, '--coordinates', coordinates, '--splits', u.splits,
                '--output', out, '--seed', seed]
        if method == 'magic':
            args += ['--n-jobs', 8]
        run(python, 'run_inductive_teacher.py', args, out/'metadata.json')

if stage == 'prepare':
    if mode == 'replicate':
        if u.key == 'norman_crispra':
            link(old_norman/'data/norman', data)
            for method in ['gene_median','svd_impute','graph_smooth','magic_inductive','scvi_inductive','safe_fusion']:
                link(old_norman/'norman'/method, ind/method)
            for method in ['magic','scvi']:
                link(old_norman/'baselines'/method/'norman', standard/method)
        else:
            run('.venv/bin/python', 'make_mask_replicate.py', ['--truth', u.truth, '--splits', u.splits,
                '--unit-column', u.unit_column, '--seed', seed, '--output-dir', data], coordinates)
    else:
        run('.venv/bin/python', 'build_deployment_inputs.py', ['--truth', u.truth, '--corrupted', u.corrupted,
            '--coordinates', u.coordinates, '--splits', u.splits, '--output-dir', data], data/'manifest.json')
elif stage == 'cpu':
    for is_trans, dest in [(False, ind), (True, trans)]:
        for method in ['gene_median','svd_impute','graph_smooth']:
            teacher(method, dest/method, is_trans)
        neural('magic', is_trans)
elif stage == 'gpu':
    neural('scvi', False)
    neural('scvi', True)
elif stage == 'fit':
    for method in ['magic','scvi']:
        run('.venv/bin/python', 'count_scale_contract.py', ['--contract', standard/method, '--corrupted', input_path,
            '--output', trans/method], trans/method/'metadata.json')
    for is_trans, dest in [(False, ind), (True, trans)]:
        names = ['gene_median','svd_impute','graph_smooth', *(['magic','scvi'] if is_trans else ['magic_inductive','scvi_inductive'])]
        args = [x for name in names for x in ['--teacher-contract', dest/name]]
        run('.venv/bin/python', 'run_leakage_safe_method.py', ['--method', 'safe_fusion', '--input', input_path,
            '--coordinates', coordinates, '--splits', u.splits, '--output', dest/'safe_fusion', '--seed', seed,
            *(['--transductive'] if is_trans else []), *args], dest/'safe_fusion/metadata.json')
        if mode == 'recorded':
            run('.venv/bin/python', 'calibrated_selective_fill.py', ['--corrupted', input_path, '--truth', u.truth,
                '--coordinates', coordinates, '--splits', u.splits, '--fusion-contract', dest/'safe_fusion', *args,
                '--output-dir', dest/'selector', '--fit-split', u.fit_split, '--architecture', 'mlp',
                '--budget-mode', 'apply_topk', '--budgets', '0.01', '0.05', '0.10', '--curve-points', 2, '--seed', seed],
                dest/'selector/calibration_report.json')
    if mode == 'replicate':
        contracts = {'Gene median': ind/'gene_median', 'SVD':ind/'svd_impute', 'Weighted kNN':ind/'graph_smooth',
                     'MAGIC (inductive)':ind/'magic_inductive', 'scVI (inductive)':ind/'scvi_inductive',
                     'MAGIC':standard/'magic', 'scVI':standard/'scvi'}
        ru = replace(u, key=key, corrupted=input_path, coordinates=coordinates, contracts=contracts,
                     selector_dir=ind/'selector', tie_seed=seed + (u.tie_seed-1729))
        manifest = root/'unit.json'
        manifest.write_text(json.dumps([unit_manifest_entry(ru)], indent=2)+'\n')
        link(trans, root/'teachers'/key)
        for family, slug in [('fusion','safe_fusion'), ('transductive','safe_fusion_transductive')]:
            run('.venv/bin/python', 'fusion_value_selectors.py', ['--units-manifest', manifest, '--unit-keys',key,
                '--unit',key,'--families',family,'--transductive-root',root/'teachers','--output-root',root/'scores',
                '--seed',seed], root/'scores'/key/slug/'report.json')
else:
    raise ValueError(stage)
