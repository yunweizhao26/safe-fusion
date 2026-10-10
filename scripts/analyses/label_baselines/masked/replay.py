import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from evaluate import ROOT, OUT, REF, units_for
from masked_f1_units import UNIT_FRACTIONS, fraction_name


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in ('fit_seconds', 'score_seconds')}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def main(key):
    dataset = 'Pancreas' if key.startswith('pancreas_') else 'Colon' if key.startswith('colon_') else key
    unit = next(u for u in units_for(dataset) if u.key == key)
    output = OUT / 'replay' / key
    output.mkdir(parents=True, exist_ok=False)
    if dataset in ('Pancreas', 'Colon'):
        source = REF / 'sex_zeros/masked_donor' / key
        original = source / 'selector_donor'
        plain = ROOT / ('artifacts/paper_evidence/review_round2/fusion_value/transductive' if dataset == 'Pancreas' else 'artifacts/paper_evidence/review_round3/colon_crossfit/fusion_value/transductive') / key
        fusion = source / 'safe_fusion_donor'
        teachers = [plain / 'gene_median', plain / 'svd_impute', source / 'graph_smooth_donor', plain / 'magic', source / 'scvi_donor']
        condition = ['--condition-column', 'donor', '--condition-feature-source', 'all_cells']
        extra = []
        if dataset == 'Colon':
            splits = pd.read_parquet(unit.splits)
            extra = ['--fit-cells', str(int(splits['split'].isin(['development', 'validation']).sum()))]
    else:
        source = REF / 'knockdown/masked' / key
        original = source / 'selector_condition'
        fusion = source / 'safe_fusion_condition'
        teachers = [source / name for name in ('gene_median', 'svd_impute', 'graph_smooth_condition', 'magic_inductive', 'scvi_inductive_condition')]
        condition = ['--condition-column', 'target']
        extra = []
    adata = ad.read_h5ad(unit.corrupted, backed='r')
    gene_ids = adata.var_names.astype(str).tolist()
    adata.file.close()
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/calibrated_selective_fill.py'),
               '--corrupted', str(unit.corrupted), '--truth', str(unit.truth), '--coordinates', str(unit.coordinates),
               '--splits', str(unit.splits), '--fusion-contract', str(fusion)]
    for teacher in teachers:
        command += ['--teacher-contract', str(teacher)]
    command += ['--output-dir', str(output), '--fit-split', unit.fit_split, '--architecture', 'mlp', '--budget-mode', 'apply_topk',
                '--budgets', *[str(x) for x in UNIT_FRACTIONS], '--curve-min-budget', '0.001', '--curve-max-budget', '1.0', '--curve-points', '1000',
                '--seed', '1729', *condition, *extra, '--score-gene', *gene_ids]
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    (output / 'command.sh').write_text(shlex.join(command) + '\n')
    with (output / 'selector.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    old = json.loads((original / 'calibration_report.json').read_text())
    new = json.loads((output / 'calibration_report.json').read_text())
    assert strip_timing(old) == strip_timing(new), 'Selector calibration report changed beyond fit/score timing'
    checks = []
    for fraction in UNIT_FRACTIONS:
        name = f'safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}'
        old_matrix = np.load(original / name / 'mean.npy', mmap_mode='r')
        new_matrix = np.load(output / name / 'mean.npy', mmap_mode='r')
        assert old_matrix.shape == new_matrix.shape and old_matrix.dtype == new_matrix.dtype
        for start in range(0, len(old_matrix), 512):
            assert np.array_equal(old_matrix[start:start+512], new_matrix[start:start+512]), f'Changed fill matrix {name}'
        checks.append(name)
    (output / 'verified.json').write_text(json.dumps({'passed': True, 'original_selector': str(original.relative_to(ROOT)), 'calibration_report_exact_except_timing': True, 'all_fill_matrices_exact': checks, 'test_pr_auc': old['test']['pr_auc'], 'seed': 1729, 'threads': 8}, indent=2) + '\n')
    print('verified', key, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
